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
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

APP_PY_UPSCALE_URL = "http://localhost:8006/upscale-only/"  # Change if app.py runs elsewhere
    


# Helper to upload to S3
def upload_to_s3(file_path: str, object_name: str) -> str:
    try:
        s3_client.upload_file(file_path, S3_BUCKET, object_name)
        url = f"{S3_ENDPOINT}/{S3_BUCKET}/{object_name}"
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



# Mapping from task_type to scale_factor
TASK_TYPE_TO_SCALE = {
    "SD24K": 4,
    "2X": 2,
    "4X": 4,
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
        # Determine scale_factor: prefer explicit, else map from task_type, else default to 4
        scale_factor = request.scale_factor
        if not scale_factor:
            scale_factor = TASK_TYPE_TO_SCALE.get(task_type, 4)

        logger.info(f"📻 Downloading video from {payload_url} ...")
        payload_video_path: str = await download_video(payload_url)
        logger.info(f"Download video finished, Path: {payload_video_path}")

        # Call the upscaler endpoint with the correct scale_factor
        processed_video_path = await call_upscaler_endpoint(payload_video_path, scale_factor=scale_factor)
        processed_video_name = Path(processed_video_path).name

        logger.info(f"Processed video path: {processed_video_path}, video name: {processed_video_name}")

        if processed_video_path is not None:
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

            logger.info(f"Public S3 download link: {s3_url}")
            return {"uploaded_video_url": s3_url, "scale_factor": scale_factor}

    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return {"uploaded_video_url": None}


if __name__ == "__main__":
    
    import uvicorn
    
    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port
    
    uvicorn.run(app, host=host, port=port)