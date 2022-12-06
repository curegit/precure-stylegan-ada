from PIL import Image
from numpy import rint, asarray, uint8, float32

def load_image(filepath, size):
	img = Image.open(filepath).convert("RGB").resize(size, Image.LANCZOS)
	return (asarray(img, dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)

def load_image_uint8(filepath, size):
	img = Image.open(filepath).convert("RGB").resize(size, Image.LANCZOS)
	return asarray(img, dtype=uint8).transpose(2, 0, 1)

def uint8_to_float(array):
	return (array / 255).astype(float32)

def to_pil_image(array):
	srgb = rint(array * 255).clip(0, 255).astype(uint8)
	return Image.fromarray(srgb.transpose(1, 2, 0), "RGB")

def save_image(array, filepath):
	to_pil_image(array).save(filepath)
