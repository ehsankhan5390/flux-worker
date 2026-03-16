import runpod
import torch
import base64
import io
from diffusers import FluxPipeline

# ================================
# LOAD MODEL (ONLY ONCE)
# ================================

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)

pipe.enable_model_cpu_offload()

# ================================
# HANDLER
# ================================

def handler(job):

    job_input = job["input"]

    prompt = job_input.get("prompt")
    count = job_input.get("count", 1)

    images_base64 = []

    for _ in range(count):

        image = pipe(
            prompt,
            guidance_scale=3.5,
            num_inference_steps=28
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        images_base64.append(
            base64.b64encode(buf.getvalue()).decode()
        )

    return {
        "images": images_base64
    }

# ================================
# START WORKER
# ================================

runpod.serverless.start({"handler": handler})
