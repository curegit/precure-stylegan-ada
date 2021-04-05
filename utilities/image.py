from PIL import Image
from numpy import rint, clip, asarray, uint8, float32

def load_image(filepath, size):
	img = Image.open(filepath).convert("RGB").resize(size, Image.LANCZOS)
	return (asarray(img, dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)

def to_pillow_image(array):
	srgb = clip(rint(array * 255), 0, 255).astype(uint8)
	return Image.fromarray(srgb.transpose(1, 2, 0), "RGB")

def save_image(array, filepath):
	to_pillow_image(array).save(filepath)
