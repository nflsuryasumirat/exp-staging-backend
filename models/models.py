import torch
import asyncio
import numpy as np
import cv2 as cv
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
from PIL import Image
from ultralytics import YOLO

YOLO_PATH = "./best.pt"

lock = asyncio.Lock()

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")

yolo = YOLO(YOLO_PATH)

def infer_yolo(img: Image.Image) -> Image.Image:
    # classes = list(yolo.names.values())

    arr = np.array(img)
    mask = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)

    results = yolo.predict(img, device="cuda")
    for result in results:
        assert result.boxes is not None
        for box in result.boxes.xywh.tolist():
            x, y, w, h = [int(x) for x in box]
            cv.rectangle(mask, [x, y], [x+w, y+h], (0, 0, 255), -1)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9), (4, 4))
    dilated = cv.dilate(mask, kernel, iterations=5)
    masked = cv.bitwise_or(arr, dilated)

    return Image.fromarray(masked)

async def infer_pipe(
    prompt: str,
    prompt_neg: str,
    image: Image.Image,
    mask: Image.Image,
    num_images_per_prompt: int = 4,
) -> list[Image.Image]:
    async with lock:
        images = pipe(
            prompt=prompt,
            negative_prompt=prompt_neg,
            image=image,
            mask_image=mask,
            num_images_per_prompt=num_images_per_prompt,
        ).images

        return images

