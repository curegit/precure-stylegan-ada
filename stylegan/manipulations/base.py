from chainer import Variable, Link
from chainer.functions import broadcast_to, where

class Manipulation(Link):

	def random_where(self, p, x, y, axis=0):
		axes = axis if axis is tuple else tuple([axis])
		shape = tuple([n if i in axes else 1 for i, n in enumerate(x.shape)])
		condition = self.xp.random.uniform(size=shape).astype(self.xp.float32) < p
		return where(broadcast_to(Variable(condition), x.shape), x, y)
