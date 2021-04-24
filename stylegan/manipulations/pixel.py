from chainer import Variable, Link
from chainer.functions import flip, broadcast_to, pad, where

class Manipulation(Link):

	def random_uniform(self, shape, broadcast=None):
		samples = Variable(self.xp.random.uniform(size=shape))
		return broadcast_to(samples, broadcast) if broadcast else samples

	def random_where(self, p, x, y, axis=0):
		axes = axis if axis is tuple else tuple([axis])
		shape = tuple([n if i in axes else 1 for i, n in enumerate(x.shape)])
		condition = self.xp.random.uniform(size=shape).astype(self.xp.float32) < p
		return where(broadcast_to(Variable(condition), x.shape), x, y)

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

class Shift(Manipulation):

	def  __call__(self, x, p):
		batch, _, height, width = x.shape
		max_h, max_w = int(height * 0.125), int(width * 0.125)
		padded = pad(x, ((0, 0), (0, 0), (max_h, max_h), (max_w, max_w)), mode="symmetric")
		shift = self.xp.random.uniform(size=(4, batch)) * self.xp.array([[0], [0], [max_h * 2], [max_w * 2]])
		b, c, h, w = self.xp.indices(x.shape) + shift.reshape(4, batch, 1, 1, 1).astype("int")
		return self.random_where(p, padded[b, c, h, w], x)
