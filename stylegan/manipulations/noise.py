from chainer import Variable
from chainer.functions import gaussian, broadcast_to, where
from stylegan.manipulations.base import Manipulation

class AdditiveNoise(Manipulation):

	def __init__(self, sd=0.05, probability_multiplier=1.0):
		super().__init__()
		self.sd = sd
		self.probability_multiplier = probability_multiplier

	def __call__(self, x, p):
		p *= self.probability_multiplier
		if p <= 0:
			return x
		batch = x.shape[0]
		normal = self.xp.random.normal(scale=self.sd, size=batch).astype(self.xp.float32)
		half_normal = self.xp.absolute(normal)
		mean = Variable(self.xp.zeros(shape=x.shape, dtype=self.xp.float32))
		ln_var = broadcast_to(Variable(self.xp.log(half_normal ** 2)).reshape(batch, 1, 1, 1), x.shape)
		noise_added = x + gaussian(mean, ln_var)
		return self.random_where(p, noise_added, x)

class Cutout(Manipulation):

	def __init__(self, width=0.5, height=0.5, fill=0.5, probability_multiplier=1.0):
		super().__init__()
		self.width = width
		self.height = height
		self.fill = fill
		self.probability_multiplier = probability_multiplier

	def __call__(self, x, p):
		p *= self.probability_multiplier
		if p <= 0:
			return x
		batch, _, height, width = x.shape
		horizontal_center = self.xp.random.uniform(size=(batch, 1)) * width
		vertical_center = self.xp.random.uniform(size=(batch, 1)) * height
		w2, h2 = self.width * width / 2, self.height * height / 2
		left, right = horizontal_center - w2, horizontal_center + w2
		top, bottom = vertical_center - h2, vertical_center + h2
		horizontal_indices = self.xp.tile(self.xp.arange(0.5, width), (batch, 1))
		vertical_indices = self.xp.tile(self.xp.arange(0.5, height), (batch, 1))
		w = ((left <= horizontal_indices) & (horizontal_indices < right)).reshape(batch, 1, 1, width)
		h = ((top <= vertical_indices) & (vertical_indices < bottom)).reshape(batch, 1, height, 1)
		cut = Variable(self.xp.tile(w & h, (1, 3, 1, 1)))
		hole = Variable(self.xp.full(x.shape, self.fill, dtype=self.xp.float32))
		cutout = where(cut, hole, x)
		return self.random_where(p, cutout, x)
