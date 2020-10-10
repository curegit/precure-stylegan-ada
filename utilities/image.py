from PIL import Image
from numpy import asarray, vectorize, rint, clip, uint8, float32

def gamma_forward(u):
	return 12.92 * u if u <= 0.0031308 else 1.055 * u ** (1 / 2.4) - 0.055

def gamma_reverse(u):
	return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4

def rgb_to_srgb(array):
	return vectorize(gamma_forward)(array)

def srgb_to_rgb(array):
	return vectorize(gamma_reverse)(array)

def load_image(filepath, size):
	img = Image.open(filepath).convert("RGB").resize(size, Image.LANCZOS)
	array = asarray(img, dtype=uint8).transpose(2, 0, 1) / 255
	return srgb_to_rgb(array).astype(float32)

def array_to_image(array):
	srgb = clip(rint(rgb_to_srgb(array) * 255), 0, 255).astype(uint8)
	return Image.fromarray(srgb.transpose(1, 2, 0), "RGB")

def save_image(array, filepath):
	array_to_image(array).save(filepath)
