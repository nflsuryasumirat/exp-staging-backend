from PIL import Image


def tmp_images() -> list[Image.Image]:
    img1 = Image.open("image.jpeg").convert("RGB")
    img2 = Image.open("mask.jpeg").convert("RGB")
    img3 = Image.open("mask_image.jpeg").convert("RGB")
    img4 = Image.open("mask_image_neg.jpeg").convert("RGB")
    return [img1, img2, img3, img4]

def tmp_images_b(l: list[Image.Image]) -> list[Image.Image]:
    return l
