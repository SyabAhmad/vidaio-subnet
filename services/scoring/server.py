from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import cv2
import numpy as np
from firerequests import FireRequests
import tempfile
import os
import glob
import random
import asyncio
from moviepy.editor import VideoFileClip
import aiohttp
import logging
import time
import math
import subprocess
from vmaf_metric import calculate_vmaf, convert_mp4_to_y4m, trim_video
from lpips_metric import calculate_lpips
from pieapp_metric import calculate_pieapp_score
from vidaio_subnet_core import CONFIG
from services.video_scheduler.video_utils import get_trim_video_path, delete_videos_with_fileid
from vidaio_subnet_core.utilities.storage_client import storage_client

# Set up logging
logger = logging.getLogger(__name__)
app = FastAPI()
fire_requests = FireRequests()

VMAF_THRESHOLD = CONFIG.score.vmaf_threshold
PIEAPP_SAMPLE_COUNT = CONFIG.score.pieapp_sample_count
PIEAPP_THRESHOLD = CONFIG.score.pieapp_threshold
VMAF_SAMPLE_COUNT = CONFIG.score.vmaf_sample_count

class SyntheticsScoringRequest(BaseModel):
    """
    Request model for scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_paths: List[str]
    uids: List[int]
    video_ids: List[str]
    uploaded_object_names: List[str]
    content_lengths: List[int]
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False

class OrganicsScoringRequest(BaseModel):
    """
    Request model for scoring. Contains URLs for distorted videos and the reference video path.
    """
    distorted_urls: List[str]
    reference_urls: List[str]
    task_types: List[str]
    uids: List[int]
    fps: Optional[float] = None
    subsample: Optional[int] = 1
    verbose: Optional[bool] = False
    progress: Optional[bool] = False

class ScoringResponse(BaseModel):
    """
    Response model for scoring. Contains the list of calculated scores for each d ofistorted video.
    """
    vmaf_scores: List[float]
    pieapp_scores: List[float]
    quality_scores: List[float]
    length_scores: List[float]
    final_scores: List[float]
    reasons: List[str]

async def download_video(video_url: str, verbose: bool) -> tuple[str, float]:
    """
    Download a video from the given URL and save it to a temporary file.

    Args:
        video_url (str): The URL of the video to download.
        verbose (bool): Whether to show download progress.

    Returns:
        tuple[str, float]: A tuple containing the path to the downloaded video file
                           and the time taken to download it.

    Raises:
        Exception: If the download fails or takes longer than the timeout.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vid_temp:
            file_path = vid_temp.name  # Path to the temporary file
        if verbose:
            print(f"Downloading video from {video_url} to {file_path}")

        timeout = aiohttp.ClientTimeout(sock_connect=0.5, total=7.5)

        start_time = time.time()  # Record start time

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(video_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download video. HTTP status: {response.status}")

                with open(file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(2 * 1024 * 1024):
                        f.write(chunk)

        end_time = time.time()  # Record end time
        download_time = end_time - start_time  # Calculate download duration

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise Exception(f"Download failed or file is empty: {file_path}")

        if verbose:
            print(f"File successfully downloaded to: {file_path}")
            print(f"Download time: {download_time:.2f} seconds")

        return file_path, download_time

    except aiohttp.ClientError as e:
        raise Exception(f"Download failed due to a network error: {e}")
    except asyncio.TimeoutError:
        raise Exception("Download timed out")

def calculate_psnr(ref_frame: np.ndarray, dist_frame: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between reference and distorted frames.

    Args:
        ref_frame (np.ndarray): The reference video frame.
        dist_frame (np.ndarray): The distorted video frame.

    Returns:
        float: The PSNR value between the reference and distorted frames.
    """
    mse = np.mean((ref_frame - dist_frame) ** 2)
    if mse == 0:
        return 1000  # Maximum PSNR value (perfect similarity)
    return 10 * np.log10((255.0**2) / mse)

def calculate_length_score(content_length):
    """
    Convert content length in seconds to a normalized length score.
    
    Args:
        content_length (float): Video duration in seconds (5-320s)
        
    Returns:
        float: Normalized length score (0-1)
    """
    return math.log(1 + content_length) / math.log(1 + 320)

def calculate_preliminary_score(quality_score, length_score, quality_weight=0.5, length_weight=0.5):
    """
    Calculate the preliminary score from quality and length scores.
    
    Args:
        quality_score (float): Normalized quality score (0-1)
        length_score (float): Normalized length score (0-1)
        quality_weight (float): Weight for quality component (default: 0.5)
        length_weight (float): Weight for length component (default: 0.5)
        
    Returns:
        float: Preliminary combined score (0-1)
    """
    return (quality_score * quality_weight) + (length_score * length_weight)

def calculate_final_score(s_pre):
    """
    Transform preliminary score into final score using exponential function.
    
    Args:
        s_pre (float): Preliminary score (0-1)
        
    Returns:
        float: Final exponentially-transformed score
    """
    return 0.1 * math.exp(6.979 * (s_pre - 0.5))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_quality_score(pieapp_score):
    sigmoid_normalized_score = sigmoid(pieapp_score)
    
    original_at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    original_at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    
    original_value = (1 - (np.log10(sigmoid_normalized_score + 1) / np.log10(3.5))) ** 2.5
    
    scaled_value = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))
    
    return scaled_value

def get_sample_frames(ref_cap, dist_cap, total_frames):
    """
    Get sample frames from both reference and distorted videos.
    
    Args:
        ref_cap: Reference video capture
        dist_cap: Distorted video capture
        total_frames: Total number of frames in the videos
        
    Returns:
        tuple: (ref_frames, dist_frames) - Lists of sampled frames
    """
    # Determine how many frames to sample
    frames_to_sample = min(PIEAPP_SAMPLE_COUNT, total_frames)
    
    # Generate a random starting point that ensures we can get consecutive frames
    # without exceeding the total number of frames
    max_start_frame = total_frames - frames_to_sample
    if max_start_frame <= 0:
        start_frame = 0
    else:
        start_frame = random.randint(0, max_start_frame)
    
    print(f"Sampling {frames_to_sample} consecutive frames starting from frame {start_frame}")
    
    # Extract frames from reference video
    ref_frames = []
    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(frames_to_sample):
        ret, frame = ref_cap.read()
        if not ret:
            break
        ref_frames.append(frame)
    
    # Extract frames from distorted video
    dist_frames = []
    dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(frames_to_sample):
        ret, frame = dist_cap.read()
        if not ret:
            break
        dist_frames.append(frame)
    
    return ref_frames, dist_frames

def get_frame_count(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return int(result.stdout.strip())

def calculate_pieapp_score_on_samples(ref_frames, dist_frames):
    """
    Calculate PIE-APP score on sampled frames without creating temporary files.
    
    Args:
        ref_frames: List of reference frames
        dist_frames: List of distorted frames
        
    Returns:
        float: Average PIE-APP score
    """
    if not ref_frames:
        print("No ref frames to process")
        return -100
    
    if not dist_frames:
        print("No dist frames to process")
        return 2.0
    
    class FrameProvider:
        def __init__(self, frames):
            self.frames = frames
            self.current_frame = 0
            self.frame_count = len(frames)
        
        def read(self):
            if self.current_frame < self.frame_count:
                frame = self.frames[self.current_frame]
                self.current_frame += 1
                return True, frame
            return False, None
        
        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return self.frame_count
            return 0
        
        def set(self, prop_id, value):
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self.current_frame = int(value) if value < self.frame_count else self.frame_count
                return True
            return False
        
        def release(self):
            # Nothing to release
            pass
        
        def isOpened(self):
            return self.frame_count > 0
    
    ref_provider = FrameProvider(ref_frames)
    dist_provider = FrameProvider(dist_frames)
    
    try:
        score = calculate_pieapp_score(ref_provider, dist_provider, frame_interval=1)
        return score
    except Exception as e:
        print(f"Error calculating PieAPP score on frames: {str(e)}")
        return -100

def upscale_video(input_path, scale_factor=2):
    """
    Upscales a video using FFmpeg by the specified scale factor.
    
    Args:
        input_path (str): Path to the input video file
        scale_factor (int): Factor by which to upscale the video (default: 2)
        
    Returns:
        str: Path to the upscaled video file
    """
    filename, extension = os.path.splitext(input_path)
    output_path = f"{filename}_upscaled{extension}"
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={new_width}:{new_height}",
        "-c:v", "libx264",
        "-preset", "medium",  
        "-crf", "18",         
        "-an",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully upscaled video to {new_width}x{new_height}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error upscaling video: {e}")
        return input_path

@app.post("/score_synthetics")
async def score_synthetics(request: SyntheticsScoringRequest) -> ScoringResponse:
    print("#################### 🤖 start scoring ####################")

    start_time = time.time()
    quality_scores = []
    length_scores = []
    final_scores = []
    vmaf_scores = []
    pieapp_scores = []
    reasons = []

    if len(request.reference_paths) != len(request.distorted_urls):
        raise HTTPException(
            status_code=400, 
            detail="Number of reference paths must match number of distorted URLs"
        )
    
    if len(request.uids) != len(request.distorted_urls):
        raise HTTPException(
            status_code=400, 
            detail="Number of UIDs must match number of distorted URLs"
        )

    for idx, (ref_path, dist_url, uid, video_id, uploaded_object_name, content_length) in enumerate(zip(
        request.reference_paths, 
        request.distorted_urls, 
        request.uids,
        request.video_ids,
        request.uploaded_object_names,
        request.content_lengths
    )):
        try:
            print(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            
            uid_start_time = time.time()  # Start time for this UID

            ref_cap = None
            dist_cap = None
            ref_y4m_path = None
            dist_path = None

            ref_cap = cv2.VideoCapture(ref_path)
            
            # Check if the reference video was actually opened successfully
            if not ref_cap.isOpened():
                # Add diagnostic information
                file_exists = os.path.exists(ref_path)
                file_size = os.path.getsize(ref_path) if file_exists else 0
                
                print(f"Error opening reference video file {ref_path}.")
                print(f"  File exists: {file_exists}")
                print(f"  File size: {file_size} bytes")
                print(f"  Current working directory: {os.getcwd()}")
                print(f"  Assigning score of 0.")
                
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append(f"error opening reference video file: {ref_path} (exists: {file_exists}, size: {file_size})")
                continue
            
            # Only log success after confirming the file was opened
            step_time = time.time() - uid_start_time
            print(f"♎️ 1. Opened reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step_time = time.time() - uid_start_time
            print(f"♎️ 2. Retrieved reference video frame count ({ref_total_frames}) in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if ref_total_frames < 10:
                print(f"Video must contain at least 10 frames. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append("reference video has fewer than 10 frames")
                ref_cap.release()
                continue

            sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
            max_start_frame = ref_total_frames - sample_size
            start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)

            print(f"Selected frame range for pieapp score {idx+1}: {start_frame} to {start_frame + sample_size - 1}")
            step_time = time.time() - uid_start_time
            print(f"♎️ 3. Selected frame range in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_frames = []
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = ref_cap.read()
                if not ret:
                    break
                ref_frames.append(frame)
            step_time = time.time() - uid_start_time
            print(f"♎️ 4. Extracted sampled frames from reference video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
            print(f"Randomly selected {VMAF_SAMPLE_COUNT} frames for VMAF score: frame list: {random_frames}")
            step_time = time.time() - uid_start_time
            print(f"♎️ 5. Selected random frames for VMAF in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
            print("The reference video has been successfully converted to Y4M format.")
            step_time = time.time() - uid_start_time
            print(f"♎️ 6. Converted reference video to Y4M in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if len(dist_url) < 10:
                print(f"Wrong dist download URL: {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("Invalid download URL: the distorted video download URL must be at least 10 characters long.")
                continue

            try:
                dist_path, download_time = await download_video(dist_url, request.verbose)
            except Exception as e:
                error_msg = f"Failed to download video from {dist_url}: {str(e)}"
                print(f"{error_msg}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to download video file from url")
                continue

            step_time = time.time() - uid_start_time
            print(f"♎️ 7. Downloaded distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            dist_cap = cv2.VideoCapture(dist_path)
            step_time = time.time() - uid_start_time
            print(f"♎️ 8. Opened distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if not dist_cap.isOpened():
                print(f"Error opening distorted video file from {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("error opening distorted video file")
                dist_cap.release()
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue

            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Distorted video has {dist_total_frames} frames.")
            step_time = time.time() - uid_start_time
            print(f"♎️ 9. Retrieved distorted video frame count in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            if dist_total_frames != ref_total_frames:
                print(
                    f"Video length mismatch for pair {idx+1}: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0."
                )
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("video length mismatch")
                dist_cap.release()
                if dist_path and os.path.exists(dist_path):
                    os.unlink(dist_path)
                continue


            # --- Downscale distorted video to original resolution for scoring ---
            downscaled_dist_path = None
            try:
                # Downscale distorted video to match reference resolution for scoring
                downscaled_dist_path = dist_path.replace('.mp4', '_downscaled_for_score.mp4')
                ref_cap_for_size = cv2.VideoCapture(ref_path)
                ref_width = int(ref_cap_for_size.get(cv2.CAP_PROP_FRAME_WIDTH))
                ref_height = int(ref_cap_for_size.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ref_cap_for_size.release()
                # Use ffmpeg for fast downscaling
                cmd = [
                    "ffmpeg", "-y", "-i", dist_path,
                    "-vf", f"scale={ref_width}:{ref_height}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
                    downscaled_dist_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Downscaled distorted video for scoring: {downscaled_dist_path}")
            except Exception as e:
                print(f"Failed to downscale distorted video for scoring: {e}")
                downscaled_dist_path = dist_path  # fallback to original

            # Calculate VMAF on downscaled video
            try:
                vmaf_start = time.time()
                vmaf_score = calculate_vmaf(ref_y4m_path, downscaled_dist_path, random_frames)
                vmaf_calc_time = time.time() - vmaf_start
                print(f"☣️☣️ VMAF calculation took {vmaf_calc_time:.2f} seconds.")

                step_time = time.time() - uid_start_time
                print(f"♎️ 10. Completed VMAF calculation in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                print(f"🎾 VMAF score is {vmaf_score}")
            except Exception as e:
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to calculate VMAF score due to video dimension mismatch")
                dist_cap.release()
                print(f"Error calculating VMAF score: {e}")
                if downscaled_dist_path and os.path.exists(downscaled_dist_path) and downscaled_dist_path != dist_path:
                    os.unlink(downscaled_dist_path)
                continue

            if vmaf_score / 100 < VMAF_THRESHOLD:
                print(f"VMAF score is too low, giving zero score, current VMAF score: {vmaf_score}")
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"VMAF score is too low, current VMAF score: {vmaf_score}")
                dist_cap.release()
                continue


            # Extract distorted frames from downscaled video for frame-based metrics
            dist_frames = []
            dist_cap_for_frames = cv2.VideoCapture(downscaled_dist_path)
            dist_cap_for_frames.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap_for_frames.read()
                if not ret:
                    break
                dist_frames.append(frame)
            dist_cap_for_frames.release()
            step_time = time.time() - uid_start_time
            print(f"♎️ 11. Extracted sampled frames from downscaled distorted video in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")

            # Calculate PieAPP score on downscaled frames
            pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
            step_time = time.time() - uid_start_time
            print(f"♎️ 12. Calculated PieAPP score in {step_time:.2f} seconds. Total time: {step_time:.2f} seconds.")
            print(f"🎾 PieAPP score is {pieapp_score}")

            if pieapp_score == -100:
                print(f"Uncertain error in pieapp calculation")
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(-100)
                reasons.append("Uncertain error in pieapp calculation")
                dist_cap.release()
                if downscaled_dist_path and os.path.exists(downscaled_dist_path) and downscaled_dist_path != dist_path:
                    os.unlink(downscaled_dist_path)
                continue

            pieapp_scores.append(pieapp_score)
            s_q = calculate_quality_score(pieapp_score)
            print(f"🏀 Quality score is {s_q}")

            dist_cap.release()

            s_l = calculate_length_score(content_length)

            s_pre = calculate_preliminary_score(s_q, s_l)

            s_f = calculate_final_score(s_pre)

            quality_scores.append(s_q)
            length_scores.append(s_l)
            final_scores.append(s_f)
            reasons.append("success")

            # Clean up downscaled video file
            if downscaled_dist_path and os.path.exists(downscaled_dist_path) and downscaled_dist_path != dist_path:
                os.unlink(downscaled_dist_path)

            step_time = time.time() - uid_start_time
            print(f"🛑 Processed one UID in {step_time:.2f} seconds.")

        except Exception as e:
            error_msg = f"Failed to process video from {dist_url}: {str(e)}"
            print(f"{error_msg}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)
            reasons.append("failed to process video")

        finally:
            # Clean up resources for this pair
            ref_cap.release()
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)

            # Delete the uploaded object
            storage_client.delete_file(uploaded_object_name)
            
            delete_videos_with_fileid(video_id)

    tmp_directory = "/tmp"
    try:
        print("🧹 Cleaning up temporary files in /tmp...")
        for file_path in glob.glob(os.path.join(tmp_directory, "*.mp4")):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        for file_path in glob.glob(os.path.join(tmp_directory, "*.y4m")):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

    processed_time = time.time() - start_time
    print(f"Completed batch scoring of {len(request.distorted_urls)} pairs within {processed_time:.2f} seconds")
    print(f"🔯🔯🔯 Calculated final scores: {final_scores} 🔯🔯🔯")
    
    return ScoringResponse(
        vmaf_scores=vmaf_scores,
        pieapp_scores=pieapp_scores,
        quality_scores=quality_scores,
        length_scores=length_scores,
        final_scores=final_scores,
        reasons=reasons
    )

@app.post("/score_organics")
async def score_organics(request: OrganicsScoringRequest) -> ScoringResponse:
    print("#################### 🤖 start scoring (organics, real metrics) ####################")
    start_time = time.time()
    vmaf_scores = []
    pieapp_scores = []
    quality_scores = []
    length_scores = []
    final_scores = []
    reasons = []

    for idx, (ref_url, dist_url, uid, task_type) in enumerate(
        zip(request.reference_urls, request.distorted_urls, request.uids, request.task_types)
    ):
        try:
            print(f"🧩 Processing pair {idx+1}/{len(request.distorted_urls)}: UID {uid} 🧩")
            uid_start_time = time.time()
            ref_cap = None
            dist_cap = None
            ref_y4m_path = None
            dist_path = None
            ref_path = None

            # Download reference video
            if len(ref_url) < 10:
                print(f"Invalid reference download URL: {ref_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("invalid reference url")
                continue
            ref_path, _ = await download_video(ref_url, request.verbose)
            ref_cap = cv2.VideoCapture(ref_path)
            if not ref_cap.isOpened():
                print(f"Error opening reference video file {ref_path}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"error opening reference video file: {ref_path}")
                continue
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if ref_total_frames < 10:
                print(f"Reference video has fewer than 10 frames. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("reference video has fewer than 10 frames")
                ref_cap.release()
                continue

            # Download distorted video
            if len(dist_url) < 10:
                print(f"Invalid distorted download URL: {dist_url}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("invalid distorted url")
                ref_cap.release()
                continue
            dist_path, _ = await download_video(dist_url, request.verbose)
            dist_cap = cv2.VideoCapture(dist_path)
            if not dist_cap.isOpened():
                print(f"Error opening distorted video file {dist_path}. Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append(f"error opening distorted video file: {dist_path}")
                ref_cap.release()
                continue
            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if dist_total_frames != ref_total_frames:
                print(f"Video length mismatch: ref({ref_total_frames}) != dist({dist_total_frames}). Assigning score of 0.")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("video length mismatch")
                ref_cap.release()
                dist_cap.release()
                continue

            # --- Downscale distorted video to original resolution for scoring ---
            downscaled_dist_path = None
            try:
                downscaled_dist_path = dist_path.replace('.mp4', '_downscaled_for_score.mp4')
                ref_cap_for_size = cv2.VideoCapture(ref_path)
                ref_width = int(ref_cap_for_size.get(cv2.CAP_PROP_FRAME_WIDTH))
                ref_height = int(ref_cap_for_size.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ref_cap_for_size.release()
                cmd = [
                    "ffmpeg", "-y", "-i", dist_path,
                    "-vf", f"scale={ref_width}:{ref_height}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
                    downscaled_dist_path
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Downscaled distorted video for scoring: {downscaled_dist_path}")
            except Exception as e:
                print(f"Failed to downscale distorted video for scoring: {e}")
                downscaled_dist_path = dist_path

            # --- VMAF calculation ---
            try:
                random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
                ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
                vmaf_score = calculate_vmaf(ref_y4m_path, downscaled_dist_path, random_frames)
                if vmaf_score is not None:
                    vmaf_scores.append(vmaf_score)
                else:
                    vmaf_score = 0.0
                    vmaf_scores.append(vmaf_score)
                print(f"🎾 VMAF score is {vmaf_score}")
            except Exception as e:
                print(f"Error calculating VMAF score: {e}")
                vmaf_scores.append(0.0)
                pieapp_scores.append(0.0)
                quality_scores.append(0.0)
                length_scores.append(0.0)
                final_scores.append(0.0)
                reasons.append("failed to calculate VMAF score")
                ref_cap.release()
                dist_cap.release()
                if downscaled_dist_path and os.path.exists(downscaled_dist_path) and downscaled_dist_path != dist_path:
                    os.unlink(downscaled_dist_path)
                continue

            # --- PieAPP calculation ---
            sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
            max_start_frame = ref_total_frames - sample_size
            start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)
            ref_frames = []
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = ref_cap.read()
                if not ret:
                    break
                ref_frames.append(frame)
            dist_frames = []
            dist_cap_for_frames = cv2.VideoCapture(downscaled_dist_path)
            dist_cap_for_frames.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap_for_frames.read()
                if not ret:
                    break
                dist_frames.append(frame)
            dist_cap_for_frames.release()
            pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
            pieapp_scores.append(pieapp_score)
            print(f"🎾 PieAPP score is {pieapp_score}")

            # --- Quality, length, final score ---
            s_q = calculate_quality_score(pieapp_score)
            s_l = 1.0  # No content_length in organics, so use 1.0
            s_pre = calculate_preliminary_score(s_q, s_l)
            s_f = calculate_final_score(s_pre)
            quality_scores.append(s_q)
            length_scores.append(s_l)
            final_scores.append(s_f)
            reasons.append("success")

            # Clean up
            ref_cap.release()
            dist_cap.release()
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)
            if downscaled_dist_path and os.path.exists(downscaled_dist_path) and downscaled_dist_path != dist_path:
                os.unlink(downscaled_dist_path)

        except Exception as e:
            print(f"Failed to process video pair: {e}. Assigning score of 0.")
            vmaf_scores.append(0.0)
            pieapp_scores.append(0.0)
            quality_scores.append(0.0)
            length_scores.append(0.0)
            final_scores.append(0.0)
            reasons.append("failed to process video pair")
            if ref_cap:
                ref_cap.release()
            if dist_cap:
                dist_cap.release()
            if ref_y4m_path and os.path.exists(ref_y4m_path):
                os.unlink(ref_y4m_path)
            if dist_path and os.path.exists(dist_path):
                os.unlink(dist_path)
            if ref_path and os.path.exists(ref_path):
                os.unlink(ref_path)
            if downscaled_dist_path and os.path.exists(downscaled_dist_path) and downscaled_dist_path != dist_path:
                os.unlink(downscaled_dist_path)
            continue

    processed_time = time.time() - start_time
    print(f"Completed batch scoring of {len(request.distorted_urls)} pairs within {processed_time:.2f} seconds")
    print(f"🔯🔯🔯 Calculated final scores: {final_scores} 🔯🔯🔯")
    return ScoringResponse(
        vmaf_scores=vmaf_scores,
        pieapp_scores=pieapp_scores,
        quality_scores=quality_scores,
        length_scores=length_scores,
        final_scores=final_scores,
        reasons=reasons
    )

if __name__ == "__main__":
    import uvicorn
    
    host = CONFIG.score.host
    port = CONFIG.score.port
    
    uvicorn.run(app, host=host, port=port)
