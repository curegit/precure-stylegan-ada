from chainer import Chain, ChainList
from stylegan.manipulations.noise import AdditiveNoise, Cutout
from stylegan.manipulations.pixel import Mirror, Rotation, Shift
from stylegan.manipulations.color import ColorAffineTransformation
from stylegan.manipulations.geometric import AffineTransformation
from stylegan.manipulations.filtering import Filtering

class AugmentationPipeline(Chain):

	def __init__(self, pixel=1.0, geometric=1.0, color=1.0, filtering=1.0, noise=1.0):
		super().__init__()
		self.probability = 1.0
		with self.init_scope():
			self.manipulations = ChainList(
				Mirror(probability_multiplier=pixel),
				Rotation(probability_multiplier=pixel),
				Shift(probability_multiplier=pixel),
				AffineTransformation(probability_multiplier=geometric),
				ColorAffineTransformation(probability_multiplier=color),
				Filtering(probability_multiplier=filtering),
				AdditiveNoise(probability_multiplier=noise),
				Cutout(probability_multiplier=noise),
			)

	def __call__(self, x):
		for f in self.manipulations:
			x = f(x, self.probability)
		return x
