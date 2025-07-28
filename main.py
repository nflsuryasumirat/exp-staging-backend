from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from typing import Any

from utils.utils import (
    decode_mask_image,
    decode_image,
    create_mask,
    combine_mask,
    img_to_b64,
)
from models.models import infer_yolo, infer_pipe
from prompts.prompts import (
    PROMPT_ADD,
    PROMPT_NEG_ADD,
    PROMPT_REMOVE,
    PROMPT_NEG_REMOVE,
)

from temp.temp import tmp_images


app = FastAPI(
    title="staging-backend",
    version="1.0.0",
    description="Virtual Staging Spike",
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class AddRequest(BaseModel):
    style: str
    room: str
    layout: str
    mask_image: str
    mask_image_neg: str
    image: str


class AddResponse(BaseModel):
    images: list[str]


@app.post("/add")
async def add(item: AddRequest):
    modifier = f"Style: {item.style}\nRoom Type: {item.room}\nLayout Type: {item.layout}\n"
    _, mask_image = decode_mask_image(item.mask_image)
    _, mask_image_neg = decode_mask_image(item.mask_image_neg)
    _, image = decode_image(item.image)

    mask = create_mask(mask_image, mask_image_neg)
    mask = mask.resize(image.size)

    final_prompt = f"{modifier}{PROMPT_ADD}"
    images = await infer_pipe(
        prompt=final_prompt,
        prompt_neg=PROMPT_NEG_ADD,
        image=image,
        mask=mask,
    )

    res: list[str] = []
    for image in images:
        res.append(img_to_b64(image))

    return { "images": res }


class RemoveRequest(BaseModel):
    mask_image: str
    mask_image_neg: str
    image: str
    auto: bool


@app.post("/remove")
async def remove(item: RemoveRequest):
    _, mask_image = decode_mask_image(item.mask_image)
    _, mask_image_neg = decode_mask_image(item.mask_image_neg)
    _, image = decode_image(item.image)
    use_yolo = item.auto

    mask = create_mask(mask_image, mask_image_neg)
    mask = mask.resize(image.size)

    if use_yolo:
        yolo_mask = infer_yolo(image)
        mask = combine_mask(yolo_mask, mask)

    images = await infer_pipe(
        prompt=PROMPT_REMOVE,
        prompt_neg=PROMPT_NEG_REMOVE,
        image=image,
        mask=mask,
    )

    res: list[str] = []
    for image in images:
        res.append(img_to_b64(image))

    return {"images": res}

