from PIL import Image
from numpy import asarray, rint, clip, uint8, float32

def load_image(filepath, size):
	img = Image.open(filepath).convert("RGB").resize(size, Image.LANCZOS)
	array = asarray(img, dtype=uint8).transpose(2, 0, 1) / 255
	return array.astype(float32)

def to_pil_image(array):
	srgb = clip(rint(array * 255), 0, 255).astype(uint8)
	return Image.fromarray(srgb.transpose(1, 2, 0), "RGB")

def save_image(array, filepath):
	to_pil_image(array).save(filepath)
