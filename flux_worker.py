import runpod
import torch
import base64
import io
import os
from diffusers import FluxPipeline

# Load model once when worker starts
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    token=os.environ.get("HF_TOKEN")
)

pipe.to("cuda")


def handler(job):

    prompt = job["input"]["prompt"]
    count = job["input"].get("count", 1)

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

    return {"images": images_base64}


runpod.serverless.start({
    "handler": handler
})
