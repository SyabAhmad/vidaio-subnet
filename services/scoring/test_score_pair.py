import sys
import os
import cv2
import random
import subprocess
from vmaf_metric import calculate_vmaf, convert_mp4_to_y4m
from pieapp_metric import calculate_pieapp_score
import numpy as np

def calculate_pieapp_score_on_samples(ref_frames, dist_frames):
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

def calculate_quality_score(pieapp_score):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    sigmoid_normalized_score = sigmoid(pieapp_score)
    original_at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    original_at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    original_value = (1 - (np.log10(sigmoid_normalized_score + 1) / np.log10(3.5))) ** 2.5
    scaled_value = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))
    return scaled_value

def calculate_preliminary_score(quality_score, length_score, quality_weight=0.5, length_weight=0.5):
    return (quality_score * quality_weight) + (length_score * length_weight)

def calculate_final_score(s_pre):
    return 0.1 * np.exp(6.979 * (s_pre - 0.5))

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_score_pair.py <reference_video.mp4> <enhanced_video.mp4>")
        sys.exit(1)
    ref_path = sys.argv[1]
    dist_path = sys.argv[2]
    # Downscale enhanced video to reference resolution for fair metric calculation
    ref_cap = cv2.VideoCapture(ref_path)
    ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_cap.release()
    downscaled_dist_path = dist_path.replace('.mp4', '_downscaled_for_score.mp4')
    cmd = [
        "ffmpeg", "-y", "-i", dist_path,
        "-vf", f"scale={ref_width}:{ref_height}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        downscaled_dist_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"Downscaled enhanced video for scoring: {downscaled_dist_path}")
    # VMAF
    VMAF_SAMPLE_COUNT = 8
    random_frames = sorted(random.sample(range(ref_total_frames), VMAF_SAMPLE_COUNT))
    ref_y4m_path = convert_mp4_to_y4m(ref_path, random_frames)
    vmaf_score = calculate_vmaf(ref_y4m_path, downscaled_dist_path, random_frames)
    print(f"VMAF score: {vmaf_score}")
    # PieAPP
    PIEAPP_SAMPLE_COUNT = 8
    sample_size = min(PIEAPP_SAMPLE_COUNT, ref_total_frames)
    max_start_frame = ref_total_frames - sample_size
    start_frame = 0 if max_start_frame <= 0 else random.randint(0, max_start_frame)
    ref_cap = cv2.VideoCapture(ref_path)
    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ref_frames = []
    for _ in range(sample_size):
        ret, frame = ref_cap.read()
        if not ret:
            break
        ref_frames.append(frame)
    ref_cap.release()
    dist_cap = cv2.VideoCapture(downscaled_dist_path)
    dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    dist_frames = []
    for _ in range(sample_size):
        ret, frame = dist_cap.read()
        if not ret:
            break
        dist_frames.append(frame)
    dist_cap.release()
    pieapp_score = calculate_pieapp_score_on_samples(ref_frames, dist_frames)
    print(f"PieAPP score: {pieapp_score}")
    # Final score
    s_q = calculate_quality_score(pieapp_score)
    s_l = 1.0
    s_pre = calculate_preliminary_score(s_q, s_l)
    s_f = calculate_final_score(s_pre)
    print(f"Quality score: {s_q}")
    print(f"Final score: {s_f}")
    # Cleanup
    if ref_y4m_path and os.path.exists(ref_y4m_path):
        os.unlink(ref_y4m_path)
    if downscaled_dist_path and os.path.exists(downscaled_dist_path) and downscaled_dist_path != dist_path:
        os.unlink(downscaled_dist_path)

if __name__ == "__main__":
    main()
