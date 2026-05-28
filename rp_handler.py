"""
RunPod Serverless worker for WAN 2.2 I2V A14B video looping
with morphic frame-interpolation LoRA.

GitHub integration: push to repo, create release, RunPod builds + deploys.
"""

import runpod

# --- model cache ---
# Globals are loaded ONCE at worker startup and reused across requests.

_PIPE = None
_MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B"
_LORA_ID = "morphic/Wan2.2-frames-to-video"
_LORA_WEIGHT = "lora_interpolation_high_noise_final.safetensors"


def _get_pipe():
    """Load model once, reuse across requests."""
    global _PIPE
    if _PIPE is None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        dtype = torch.bfloat16

        vae = AutoencoderKLWan.from_pretrained(
            _MODEL_ID, subfolder="vae", torch_dtype=torch.float32
        )
        _PIPE = WanImageToVideoPipeline.from_pretrained(
            _MODEL_ID, vae=vae, torch_dtype=dtype
        )
        _PIPE.to("cuda")
        _PIPE.load_lora_weights(_LORA_ID, weight_name=_LORA_WEIGHT)
        _PIPE.fuse_lora()

    return _PIPE


def _load_image(source: str):
    """Load an image from URL or base64 data URI."""
    import io
    import base64
    from PIL import Image

    if source.startswith("data:"):
        _, encoded = source.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded)))
    elif source.startswith("http"):
        import requests as r
        resp = r.get(source, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    else:
        raise ValueError(f"Cannot parse image source: {source[:80]}...")


def handler(event):
    """
    Generate a video transitioning from start_image to end_image.

    Input:
        start_image: str (URL or base64 data URI)
        end_image: str (URL or base64 data URI)
        prompt: str
        num_frames: int (optional, default 81)
        width: int (optional, default 832)
        height: int (optional, default 480)
        num_inference_steps: int (optional, default 50)
        guidance_scale: float (optional, default 5.0)
        seed: int (optional)
    """
    import base64
    import os
    import tempfile
    import uuid

    from diffusers.utils import export_to_video
    import torch

    job_input = event["input"]

    # --- parse input ---
    prompt = job_input["prompt"]
    start_image = _load_image(job_input["start_image"])
    end_image = _load_image(job_input["end_image"])

    num_frames = job_input.get("num_frames", 81)
    width = job_input.get("width", 832)
    height = job_input.get("height", 480)
    num_inference_steps = job_input.get("num_inference_steps", 50)
    guidance_scale = job_input.get("guidance_scale", 5.0)
    seed = job_input.get("seed")

    # --- get cached model ---
    pipe = _get_pipe()

    # --- generate ---
    generator = None
    if seed is not None:
        generator = torch.Generator("cuda").manual_seed(seed)

    start_image = start_image.resize((width, height))
    end_image = end_image.resize((width, height))

    with torch.inference_mode():
        frames = pipe(
            image=start_image,
            image_end=end_image,
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

    # --- save and return ---
    output_dir = tempfile.mkdtemp(prefix="wan_")
    output_path = os.path.join(output_dir, f"{uuid.uuid4().hex}.mp4")
    export_to_video(frames, output_path, fps=16)

    with open(output_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    os.unlink(output_path)
    os.rmdir(output_dir)

    return {"video_base64": video_b64, "num_frames": num_frames, "fps": 16}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
