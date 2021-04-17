from chainer import Variable, Link
from chainer.functions import flip, where, broadcast_to

class Manipulation(Link):

	def random_where(self, p, x, y, axis=0):
		axes = axis if axis is tuple else tuple([axis])
		shape = tuple([n if i in axes else 1 for i, n in enumerate(x.shape)])
		condition = Variable(self.xp.random.uniform(size=shape).astype(self.xp.float32)) < p
		return where(broadcast_to(condition, x.shape), x, y)

class Mirror(Manipulation):

	def __call__(self, x, p):
		mirrored = self.random_where(0.5, x, flip(x, axis=3))
		return self.random_where(p, mirrored, x)

class Rotation(Manipulation):

	def __call__(self, x, p):
		transposed = x.transpose(0, 1, 3, 2)
		rot90 = flip(transposed, axis=2)
		rot180 = flip(flip(x, axis=2), axis=3)
		rot270 = flip(transposed, axis=3)
		rotated = self.random_where(0.5, self.random_where(0.5, x, rot90), self.random_where(0.5, rot180, rot270))
		return self.random_where(p, rotated, x)
