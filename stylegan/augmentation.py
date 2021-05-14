from chainer import Chain, ChainList
from stylegan.manipulations.noise import AdditiveNoise, Cutout
from stylegan.manipulations.pixel import Mirror, Rotation, Shift
from stylegan.manipulations.color import ColorAffineTransformation
from stylegan.manipulations.geometric import AffineTransformation

class AugmentationPipeline(Chain):

	def __init__(self, probability=0.5):
		super().__init__()
		self.probability = probability
		with self.init_scope():
			self.manipulations = ChainList(*[
				Mirror(),
				Rotation(),
				Shift(),
				AffineTransformation(),
				ColorAffineTransformation(),
				AdditiveNoise(),
				Cutout()])

	def __call__(self, x):
		for f in self.manipulations:
			x = f(x, self.probability)
		return x
