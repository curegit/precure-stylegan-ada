from os.path import lexists, isdir
from numpy import eye, float32
from chainer.dataset import DatasetMixin
from utilities.stdio import eprint
from utilities.image import load_image
from utilities.filesys import glob_recursively

class ImageDataset(DatasetMixin):

	extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"]

	def __init__(self, directory, resolution):
		super().__init__()
		self.resolution = resolution
		self.preloaded = False
		if not lexists(directory):
			eprint(f"Invalid dataset: {directory}")
			eprint("No such directory!")
			raise RuntimeError("Input error")
		if not isdir(directory):
			eprint(f"Invalid dataset: {directory}")
			eprint("Specified path is not a correct directory!")
			raise RuntimeError("Input error")
		self.image_files = sum([glob_recursively(directory, e, robust_letter_case=True) for e in ImageDataset.extensions], [])
		if not self.image_files:
			eprint(f"Invalid dataset: {directory}")
			eprint("No images found in the directory!")
			raise RuntimeError("Input error")

	def __len__(self):
		return len(self.image_files)

	def preload(self, callback=None):
		self.preloaded = True
		self.loaded_images = []
		for i in self.image_files:
			self.loaded_images.append(load_image(i, self.resolution))
			if callback:
				callback()

	def get_example(self, index):
		return self.loaded_images[index] if self.preloaded else load_image(self.image_files[index], self.resolution)

class MulticategoryImageDataset(DatasetMixin):

	def __init__(self, directories, resolution):
		super().__init__()
		self.datasets = [ImageDataset(d, resolution) for d in directories]

	def __len__(self):
		return sum(len(dataset) for dataset in self.datasets)

	def preload(self, callback=None):
		for dataset in self.datasets:
			dataset.preload(callback)

	def get_example(self, index):
		for i, dataset in enumerate(self.datasets):
			if index < len(dataset):
				condition = eye(len(self.datasets), dtype=float32)[i]
				return dataset.get_example(index), condition
			else:
				index -= len(dataset)
