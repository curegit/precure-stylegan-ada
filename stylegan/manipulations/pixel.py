from chainer.functions import flip, pad
from stylegan.manipulations.base import Manipulation

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

	def __init__(self, horizontal=0.125, vertical=0.125):
		super().__init__()
		self.horizontal = horizontal
		self.vertical = vertical

	def __call__(self, x, p):
		batch, _, height, width = x.shape
		sh, sw = round(height * self.vertical), round(width * self.horizontal)
		padded = pad(x, ((0, 0), (0, 0), (sh, sh), (sw, sw)), mode="symmetric")
		shift = self.xp.random.uniform(size=(2, batch)) * self.xp.array([[sh * 2 + 1], [sw * 2 + 1]])
		indices_shift = self.xp.concatenate((self.xp.zeros(shape=(2, batch)), shift))
		indices_shift_int = self.xp.trunc(indices_shift).astype(self.xp.int).reshape(4, batch, 1, 1, 1)
		b, c, h, w = self.xp.indices(x.shape) + indices_shift_int
		shifted = padded[b, c, h, w]
		return self.random_where(p, shifted, x)
