import subprocess
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import uvicorn
import docker
import sys
import re
from threading import Lock
import uuid

app = FastAPI()

class NerfstudioRequest(BaseModel):
    scene_data: str = "data/homee/test_pipeline/2floor_ball_room_iphone12/colmap/"

# Initialize Docker client
docker_client = docker.from_env()
container = None

# Global dictionary to store job progress
job_progress = {}
progress_lock = Lock()

@app.on_event("startup")
async def startup_event():
    global container
    try:
        # Start the Docker container
        container = docker_client.containers.run(
            "nerfstudio",
            command="bash -c 'while true; do sleep 30; done'",  # Keep the container running
            detach=True,
            auto_remove=True,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes={
                "/home/dennis.chuang/homee_3DGS/nerfstudio/": {
                    "bind": "/workspace",
                    "mode": "rw",
                },
            },
            ports={'7008/tcp': 7008},
            shm_size="12G"
        )
        print(f"Container {container.short_id} started.")
    except docker.errors.ImageNotFound:
        print("Error: Docker image not found.")
        raise HTTPException(status_code=500, detail="Docker image not found")
    except docker.errors.APIError as e:
        print(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Docker API error")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error starting container")

@app.on_event("shutdown")
async def shutdown_event():
    global container
    if container:
        container.stop()
        print(f"Container {container.short_id} stopped.")

def run_nerfstudio(job_id: str, scene_data: str):
    global container, job_progress
    if not container:
        raise HTTPException(status_code=500, detail="Container is not running")

    command = f"bash prepare_dataset.sh {scene_data} glomap"
    try:
        # Use stream=True to get real-time output
        exec_command = container.exec_run(command, workdir="/workspace", stream=True)
        for line in exec_command.output:
            # Decode and print each line of output
            decoded_line = line.decode().strip()
            print(decoded_line, flush=True)
            
            # Check for progress percentage in the output
            match = re.search(r'(\d+(?:\.\d+)?)%', decoded_line)
            if match:
                percentage = float(match.group(1))
                with progress_lock:
                    job_progress[job_id] = percentage

    except Exception as e:
        print(f"Error occurred: {str(e)}", flush=True)
    finally:
        # Ensure job is marked as completed
        with progress_lock:
            job_progress[job_id] = 100.0

@app.post("/api/start_nerfstudio_service")
async def start_nerfstudio_service(request: NerfstudioRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_nerfstudio, job_id, request.scene_data)
    return {"status": "PROCESSING", "job_id": job_id}

@app.get("/api/job_status/{job_id}")
async def get_job_status(job_id: str):
    return {"status": job_progress.get(job_id, "NOT_FOUND")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)