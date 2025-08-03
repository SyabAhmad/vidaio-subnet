import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

project_root2 = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root2))



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import os
import aiohttp
import asyncio


from vidaio_subnet_core import CONFIG
from typing import Optional
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import download_video
from loguru import logger
import traceback
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import subprocess
import random
from config.enhancement_config import ENHANCEMENT_CHAINS, AVAILABLE_METHODS


app = FastAPI()
# changes

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    scale_factor: Optional[int] = None  # Will be set based on task_type if not provided


# Load S3 credentials from .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../../.env'))
S3_BUCKET = os.getenv('BUCKET_NAME')
S3_ENDPOINT = os.getenv('BUCKET_COMPATIBLE_ENDPOINT')
S3_ACCESS_KEY = os.getenv('BUCKET_COMPATIBLE_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('BUCKET_COMPATIBLE_SECRET_KEY')

s3_client = boto3.client(
    's3',
    region_name='us-east-2',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

APP_PY_UPSCALE_URL = "http://localhost:8006/upscale-only/"  # Change if app.py runs elsewhere
    


# Helper to upload to S3
def upload_to_s3(file_path: str, object_name: str) -> str:
    try:
        s3_client.upload_file(file_path, S3_BUCKET, object_name)
        # Generate a presigned URL for the uploaded object (valid for 7 days)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': object_name},
            ExpiresIn=604800  # 7 days
        )
        return url
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload to S3")

# Call the /upscale-only/ endpoint in app.py
async def call_upscaler_endpoint(input_path: str, scale_factor: int = 4, model_type: str = "general", timeout_sec: int = 10800) -> str: # Adjusted timeout to 3 hours
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with open(input_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=os.path.basename(input_path), content_type='video/mp4')
            data.add_field('scale_factor', str(scale_factor))
            data.add_field('model_type', model_type)
            async with session.post(APP_PY_UPSCALE_URL, data=data) as resp:
                if resp.status != 200:
                    logger.error(f"Upscaler endpoint failed: {await resp.text()}")
                    raise HTTPException(status_code=500, detail="Upscaler endpoint failed")
                # Save the upscaled video to a temp file
                out_path = input_path.replace('.mp4', '_upscaled.mp4')
                with open(out_path, 'wb') as out_f:
                    out_f.write(await resp.read())
                return out_path

async def process_with_config(input_path: str, scale_factor: int) -> str:
    """Process video using enhancement config"""
    try:
        if scale_factor not in ENHANCEMENT_CHAINS:
            # Fallback to direct processing
            return await call_upscaler_endpoint(input_path, scale_factor=scale_factor)
        
        chain_config = ENHANCEMENT_CHAINS[scale_factor]
        method_name = chain_config['method']
        
        logger.info(f"🔗 Using method: {method_name} for {scale_factor}x")
        
        if method_name not in AVAILABLE_METHODS:
            logger.error(f"❌ Unknown method: {method_name}")
            raise Exception(f"Unknown method: {method_name}")
        
        method_config = AVAILABLE_METHODS[method_name]
        steps = method_config['steps']
        
        # Check what steps we have
        if len(steps) == 2 and steps[0]['type'] == 'ai' and steps[1]['type'] == 'ai':
            # Double AI steps
            logger.info("🎨 Processing: 2x AI → 2x AI")
            
            # Step 1: 2x AI
            ai_step1_path = await call_upscaler_endpoint(input_path, scale_factor=2)
            
            # Step 2: 2x AI (on the result of step 1)
            final_path = ai_step1_path.replace('_upscaled.mp4', '_double_ai_4x.mp4')
            ai_step2_path = await call_upscaler_endpoint(ai_step1_path, scale_factor=2)
            
            # Rename final result
            os.rename(ai_step2_path, final_path)
            
            # Clean up intermediate
            if os.path.exists(ai_step1_path):
                os.remove(ai_step1_path)
            
            return final_path
            
        elif len(steps) == 2 and steps[0]['type'] == 'ai' and steps[1]['type'] == 'interpolation':
            # AI + Interpolation
            logger.info("🎨 Processing: 2x AI → 2x Interpolation")
            
            # Step 1: 2x AI
            ai_path = await call_upscaler_endpoint(input_path, scale_factor=2)
            
            # Step 2: 2x Interpolation
            final_path = ai_path.replace('_upscaled.mp4', '_hybrid_4x.mp4')
            interp_cmd = [
                "ffmpeg", "-y", "-i", ai_path,
                "-vf", "scale=iw*2:ih*2:flags=lanczos",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                final_path
            ]
            
            result = subprocess.run(interp_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise Exception(f"Interpolation failed: {result.stderr}")
            
            # Clean up intermediate
            if os.path.exists(ai_path):
                os.remove(ai_path)
            
            return final_path
        
        else:
            raise Exception(f"Unsupported method configuration: {method_name}")
            
    except Exception as e:
        logger.error(f"❌ Config processing failed: {e}")
        raise e


# Mapping from task_type to scale_factor
TASK_TYPE_TO_SCALE = {
    # ✅ THE 3 REQUIRED TASKS (must complete in 35 seconds)
    "SD2HD": 2,     # SD to HD = 2x upscaling (~20 seconds)
    "HD24K": 2,     # HD to 4K = 2x upscaling (~20 seconds) 
    "SD24K": "hybrid_4x",  # SD to 4K = 4x upscaling (2x AI + 2x interpolation ~25-30 seconds)
    
    # Legacy support for your existing tests
    "2X": 2,
    "4X": "hybrid_4x",  # Also use hybrid for 4x requests
    "8X": 8,
    "10X": 10,
    "12X": 12,
    "16X": 16,
    "20X": 20,
}

@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    try:
        payload_url = request.payload_url
        task_type = request.task_type.upper() if request.task_type else ""

        # Determine processing method
        scale_factor = request.scale_factor
        use_hybrid = False
        
        if not scale_factor:
            mapped_value = TASK_TYPE_TO_SCALE.get(task_type, 2)
            if mapped_value == "hybrid_4x":
                scale_factor = 4  # User sees 4x
                use_hybrid = True
            else:
                scale_factor = mapped_value
        elif scale_factor == 4:
            # Always use hybrid for 4x requests
            use_hybrid = True
        
        # Ensure scale_factor is int for downstream use
        try:
            scale_factor = int(scale_factor) if not use_hybrid else 4
        except Exception:
            scale_factor = 2

        logger.info(f"📻 Downloading video from {payload_url} ...")
        payload_video_path: str = await download_video(payload_url)
        logger.info(f"Download video finished, Path: {payload_video_path}")

        if use_hybrid:
            # ✅ USE CONFIG-BASED PROCESSING
            logger.info("🔗 Using config-based processing for 4x")
            processed_video_path = await process_with_config(payload_video_path, 4)
            effective_scale = 4
        else:
            # ✅ PURE AI APPROACH
            processed_video_path = await call_upscaler_endpoint(payload_video_path, scale_factor=scale_factor)
            effective_scale = scale_factor

        logger.info(f"Processed video path: {processed_video_path}")
        
        # --- Direct scoring: import and call metrics ---
        import sys
        import numpy as np
        import cv2
        sys.path.append(os.path.dirname(__file__))
        try:
            from services.scoring.vmaf_metric import calculate_vmaf, convert_mp4_to_y4m
        except ImportError:
            from vidaio_subnet.services.scoring.vmaf_metric import calculate_vmaf, convert_mp4_to_y4m
        try:
            from services.scoring.lpips_metric import calculate_lpips
        except ImportError:
            from vidaio_subnet.services.scoring.lpips_metric import calculate_lpips
        try:
            from services.scoring.pieapp_metric import calculate_pieapp_score
        except ImportError:
            from vidaio_subnet.services.scoring.pieapp_metric import calculate_pieapp_score

        # Downscale processed video to reference resolution for scoring
        ref_cap = cv2.VideoCapture(payload_video_path)
        ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ref_cap.release()
        downscaled_proc_path = processed_video_path.replace('.mp4', '_downscaled_for_score.mp4')
        cmd = [
            "ffmpeg", "-y", "-i", processed_video_path,
            "-vf", f"scale={ref_width}:{ref_height}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
            downscaled_proc_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Downscaled processed video for scoring: {downscaled_proc_path}")

        # VMAF
        VMAF_SAMPLE_COUNT = 8
        random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT)) if ref_total_frames >= VMAF_SAMPLE_COUNT else list(range(ref_total_frames))
        ref_y4m_path = convert_mp4_to_y4m(payload_video_path, random_frames)
        try:
            vmaf_score = calculate_vmaf(ref_y4m_path, downscaled_proc_path, random_frames)
        except Exception as e:
            logger.warning(f"VMAF calculation failed: {e}")
            vmaf_score = None

        # PieAPP, LPIPS, PSNR on sampled frames
        PIEAPP_SAMPLE_COUNT = 8
        sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
        max_start_frame = ref_total_frames - sample_size
        start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)
        ref_cap = cv2.VideoCapture(payload_video_path)
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ref_frames = []
        for _ in range(sample_size):
            ret, frame = ref_cap.read()
            if not ret:
                break
            ref_frames.append(frame)
        ref_cap.release()
        proc_cap = cv2.VideoCapture(downscaled_proc_path)
        proc_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        proc_frames = []
        for _ in range(sample_size):
            ret, frame = proc_cap.read()
            if not ret:
                break
            proc_frames.append(frame)
        proc_cap.release()


        lpips_score = None
        pieapp_score = None
        psnr_score = None
        ssim_score = None
        logger.info(f"[DEBUG] ref_frames: {len(ref_frames) if ref_frames else 0}, proc_frames: {len(proc_frames) if proc_frames else 0}")
        if ref_frames:
            logger.info(f"[DEBUG] ref_frames[0] shape: {getattr(ref_frames[0], 'shape', None)}")
        if proc_frames:
            logger.info(f"[DEBUG] proc_frames[0] shape: {getattr(proc_frames[0], 'shape', None)}")
        if ref_frames and proc_frames and len(ref_frames) == len(proc_frames):
            try:
                lpips_vals = [calculate_lpips(rf, pf) for rf, pf in zip(ref_frames, proc_frames) if isinstance(rf, np.ndarray) and isinstance(pf, np.ndarray) and rf.shape == pf.shape]
                lpips_score = float(np.mean(lpips_vals)) if lpips_vals else None
                try:
                    ref_cap = cv2.VideoCapture(payload_video_path)
                    proc_cap = cv2.VideoCapture(downscaled_proc_path)
                    pieapp_score = calculate_pieapp_score(ref_cap, proc_cap, frame_interval=30)  # adjust interval as needed
                except Exception as e:
                    logger.warning(f"PieAPP calculation failed: {e}")
                    pieapp_score = None
                logger.info(f"[DEBUG] LPIPS values: {lpips_vals}")
            except Exception as e:
                logger.warning(f"LPIPS calculation failed: {e}")
            try:
                ref_cap = cv2.VideoCapture(payload_video_path)
                proc_cap = cv2.VideoCapture(downscaled_proc_path)
                pieapp_score = calculate_pieapp_score(ref_cap, proc_cap, frame_interval=30)  # adjust interval as needed
                logger.info(f"[DEBUG] PieAPP score: {pieapp_score}")
            except Exception as e:
                logger.warning(f"PieAPP calculation failed: {e}")
                pieapp_score = None
                logger.warning(f"PieApp Score: {pieapp_score}")
            # try:
            #     pieapp_vals = [calculate_pieapp_score(rf, pf) for rf, pf in zip(ref_frames, proc_frames) if isinstance(rf, np.ndarray) and isinstance(pf, np.ndarray) and rf.shape == pf.shape]
            #     logger.info(f"[DEBUG] PieAPP values: {pieapp_vals}")
            #     pieapp_score = float(np.mean(pieapp_vals)) if pieapp_vals else None
            # except Exception as e:
            #     logger.warning(f"PieAPP calculation failed: {e}")
            try:
                psnr_vals = [cv2.PSNR(rf, pf) for rf, pf in zip(ref_frames, proc_frames) if isinstance(rf, np.ndarray) and isinstance(pf, np.ndarray) and rf.shape == pf.shape]
                psnr_score = float(np.mean(psnr_vals)) if psnr_vals else None
                logger.info(f"[DEBUG] PSNR values: {psnr_vals}")
            except Exception as e:
                logger.warning(f"PSNR calculation failed: {e}")
            try:
                from skimage.metrics import structural_similarity as ssim
                ssim_vals = []
                for rf, pf in zip(ref_frames, proc_frames):
                    if isinstance(rf, np.ndarray) and isinstance(pf, np.ndarray) and rf.shape == pf.shape:
                        # Convert to grayscale for SSIM
                        rf_gray = cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY)
                        pf_gray = cv2.cvtColor(pf, cv2.COLOR_BGR2GRAY)
                        ssim_val = ssim(rf_gray, pf_gray, data_range=255)
                        ssim_vals.append(ssim_val)
                ssim_score = float(np.mean(ssim_vals)) if ssim_vals else None
                logger.info(f"[DEBUG] SSIM values: {ssim_vals}")
            except Exception as e:
                logger.warning(f"SSIM calculation failed: {e}")
        else:
            logger.warning(f"Frame sampling failed or frame count mismatch for scoring. ref_frames: {len(ref_frames) if ref_frames else 0}, proc_frames: {len(proc_frames) if proc_frames else 0}")

        logger.info(f"Scoring results: VMAF={vmaf_score}, PieAPP={pieapp_score}, LPIPS={lpips_score}, PSNR={psnr_score}, SSIM={ssim_score}")

        if processed_video_path is not None:
            processed_video_name = os.path.basename(processed_video_path)
            object_name: str = processed_video_name
            s3_url = upload_to_s3(processed_video_path, object_name)
            logger.info("Video uploaded to S3 successfully.")

            # Delete the local file since we've already uploaded it to S3
            if os.path.exists(processed_video_path):
                os.remove(processed_video_path)
                logger.info(f"{processed_video_path} has been deleted.")
            else:
                logger.info(f"{processed_video_path} does not exist.")

            # Schedule the file for deletion after 10 minutes (600 seconds)
            deletion_scheduled = schedule_file_deletion(object_name)
            if deletion_scheduled:
                logger.info(f"Scheduled deletion of {object_name} after 10 minutes")
            else:
                logger.warning(f"Failed to schedule deletion of {object_name}")

            # Schedule deletion of the downscaled video used for scoring
            downscaled_object_name = os.path.basename(downscaled_proc_path)
            downscaled_deletion_scheduled = schedule_file_deletion(downscaled_object_name)
            if downscaled_deletion_scheduled:
                logger.info(f"Scheduled deletion of downscaled scoring file {downscaled_object_name} after 10 minutes")
            else:
                logger.warning(f"Failed to schedule deletion of downscaled scoring file {downscaled_object_name}")

            # Optionally, delete the local downscaled file immediately (if not needed for other tasks)
            if os.path.exists(downscaled_proc_path):
                os.remove(downscaled_proc_path)
                logger.info(f"{downscaled_proc_path} has been deleted.")
            else:
                logger.info(f"{downscaled_proc_path} does not exist.")

            logger.info(f"Public S3 download link: {s3_url}")
            return {
                "uploaded_video_url": s3_url,
                "scale_factor": scale_factor,
                "vmaf": vmaf_score,
                "pieapp": pieapp_score,
                "lpips": lpips_score,
                "psnr": psnr_score,
                "ssim": ssim_score
            }

    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return {"uploaded_video_url": None}

if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)