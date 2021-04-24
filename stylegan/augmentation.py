from chainer import Chain, ChainList
from stylegan.manipulations.pixel import Mirror, Rotation, Shift

class AugmentationPipeline(Chain):

	def __init__(self, probability=0.0):
		super().__init__()
		self.probability = probability
		with self.init_scope():
			self.manipulations = ChainList(*[
				Mirror(),
				Rotation(),
				Shift()])

	def __call__(self, x):
		for f in self.manipulations:
			x = f(x, self.probability)
		return x
