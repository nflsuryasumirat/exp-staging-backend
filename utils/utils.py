import base64
import cv2 as cv
import numpy as np
from io import BytesIO
from PIL import Image, ImageChops

# FIXME: image not loading properly (color_mask -> mask is not good)
def decode_mask_image(img: str) -> tuple[str, Image.Image]:
    img_type, img_b64 = img.split(",")
    img_bytes = base64.b64decode(img_b64)
    img_bytes = BytesIO(img_bytes)
    ret_img = Image.open(img_bytes).convert("L", dither=None)

    return img_type, ret_img


def decode_image(img: str) -> tuple[str, Image.Image]:
    img_type, img_b64 = img.split(",")
    img_bytes = base64.b64decode(img_b64)
    img_bytes = BytesIO(img_bytes)
    ret_img = Image.open(img_bytes).convert("RGB")

    return img_type, ret_img

def create_mask(pos: Image.Image, neg: Image.Image) -> Image.Image:
    pos_arr = np.array(pos)
    _, pos_arr = cv.threshold(pos_arr, 0, 255, cv.THRESH_BINARY)

    neg_arr = np.array(neg)
    _, neg_arr = cv.threshold(neg_arr, 0, 255, cv.THRESH_BINARY)

    mask_arr = cv.subtract(pos_arr, neg_arr, dtype=cv.CV_8U)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9), (4, 4))
    dilated = cv.morphologyEx(mask_arr, cv.MORPH_CLOSE, kernel, iterations=3)

    mask = Image.fromarray(dilated)
    return mask

def combine_mask(img1: Image.Image, img2: Image.Image) -> Image.Image:
    return ImageChops.add(img1.convert("L"), img2.convert("L"))

def img_to_b64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")
