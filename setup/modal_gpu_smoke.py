import subprocess

import modal


app = modal.App("ai-project-gpu-smoke")


@app.function(gpu="T4")
def check_gpu():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@app.local_entrypoint()
def main():
    print("Remote GPU:", check_gpu.remote())
