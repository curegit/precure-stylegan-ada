from chainer.dataset import DatasetMixin
from utilities.stdio import eprint
from utilities.image import load_image
from utilities.filesys import glob_recursively

class ImageDataset(DatasetMixin):

	extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"]

	def __init__(self, directory, resolution, preload=False):
		super().__init__()
		self.resolution = resolution
		self.preload = preload
		self.image_files = sum([glob_recursively(directory, e) for e in ImageDataset.extensions], [])
		self.loaded_images = [load_image(i, resolution) for i in self.image_files] if preload else []
		if not self.image_files:
			eprint("No images found in the directory!")
			raise RuntimeError("empty dataset")

	def __len__(self):
		return len(self.image_files)

	def get_example(self, index):
		return self.loaded_images[index] if self.preload else load_image(self.image_files[index], self.resolution)
