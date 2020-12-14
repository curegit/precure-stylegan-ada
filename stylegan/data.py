from glob import glob
from os.path import isfile
from chainer.dataset import DatasetMixin
from utilities.stdio import eprint
from utilities.image import load_image
from utilities.filesys import build_filepath

class ImageDataset(DatasetMixin):

	def __init__(self, directory, resolution, preload=False):
		super().__init__()
		self.resolution = resolution
		self.preload = preload
		self.images = []
		for e in ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff"]:
			self.images += [f for f in glob(build_filepath(directory, "**/*", e), recursive=True) if isfile(f)]
		if not self.images:
			eprint("")
			assert False
		if preload:
			self.list = [load_image(i, resolution) for i in self.images]

	def __len__(self):
		return len(self.images)

	def get_example(self, index):
		return self.list[index] if self.preload else load_image(self.images[index], self.resolution)
