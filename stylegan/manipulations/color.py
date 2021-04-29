from chainer.functions import where
from stylegan.manipulations.base import Manipulation

class GammaTransformation(Manipulation):

	def __call__(self, x, p):
		batch = x.shape[0]
		gamma = self.xp.random.uniform(low=0.6, high=1.4, size=batch).astype(self.xp.float32)
		gt = where(0 < x.array, x ** gamma.reshape(batch, 1, 1, 1), x)
		return self.random_where(p, gt, x)
