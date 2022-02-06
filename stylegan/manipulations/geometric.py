from math import sqrt, log, sin, cos, pi
from numpy import array, eye, stack, float32
from numpy.random import uniform, normal, lognormal
from chainer.functions import sum, pad
from stylegan.manipulations.base import Manipulation

indentity = eye(3, dtype=float32)

def translation(x, y):
	return array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float32)

def rotation(theta):
	return array([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]], dtype=float32)

def scaling(s, t):
	return array([[s, 0, 0], [0, t, 0], [0, 0, 1]], dtype=float32)

def inverse_translation(x, y):
	return translation(-x, -y)

def inverse_rotation(theta):
	return rotation(-theta)

def inverse_scaling(s, t):
	return scaling(1 / s, 1 / t)

class AffineTransformation(Manipulation):

	def __init__(self, translation=0.125, rotation=360, scale=0.2, probability_multiplier=1.0):
		super().__init__()
		self.translation = translation
		self.rotation = rotation / 360 * pi
		self.scale = scale * log(2)
		self.probability_multiplier = probability_multiplier

	def __call__(self, x, p=1.0):
		p *= self.probability_multiplier
		if p <= 0:
			return x
		batch, _, height, width = x.shape
		centering = inverse_translation(-height / 2 + 0.5, -width / 2 + 0.5)
		affine = indentity @ centering
		condition = uniform(size=batch) < p
		scale = lognormal(sigma=self.scale, size=batch)
		isotropic_scaling = stack([inverse_scaling(s, s) if c else indentity for c, s in zip(condition, scale)])
		affine = affine @ isotropic_scaling
		condition = uniform(size=batch) < 1 - sqrt(1 - p)
		theta = uniform(low=-self.rotation, high=self.rotation, size=batch)
		pre_rotation = stack([inverse_rotation(t) if c else indentity for c, t in zip(condition, theta)])
		affine = affine @ pre_rotation
		condition = uniform(size=batch) < p
		scale = lognormal(sigma=self.scale, size=batch)
		anisotropic_scaling = stack([inverse_scaling(1 / s, s) if c else indentity for c, s in zip(condition, scale)])
		affine = affine @ anisotropic_scaling
		condition = uniform(size=batch) < 1 - sqrt(1 - p)
		theta = uniform(low=-self.rotation, high=self.rotation, size=batch)
		post_rotation = stack([inverse_rotation(t) if c else indentity for c, t in zip(condition, theta)])
		affine = affine @ post_rotation
		condition = uniform(size=batch) < p
		th = height * normal(scale=self.translation, size=batch)
		tw = width * normal(scale=self.translation, size=batch)
		translation = stack([inverse_translation(h, w) if c else indentity for c, h, w in zip(condition, th, tw)])
		affine = affine @ translation
		inverse_centering = inverse_translation(height / 2 - 0.5, width / 2 - 0.5)
		affine = affine @ inverse_centering
		affine = self.xp.asarray(affine)
		indices = self.xp.indices((height, width), dtype=self.xp.float32)
		ones = self.xp.ones((1, height, width), dtype=self.xp.float32)
		coordinate = self.xp.concatenate((indices, ones)).transpose(1, 2, 0)
		resampling_coordinate = affine.reshape(batch, 1, 3, 3) @ coordinate.reshape(1, height * width, 3, 1)
		resampling_grid = resampling_coordinate.reshape(batch, height, width, 3).transpose(3, 0, 1, 2)[0:2]
		return self.lanczos_resampling(x, resampling_grid)

	def lanczos_resampling(self, x, grid):
		batch, channels, height, width = x.shape
		i2 = self.xp.floor(grid).astype(self.xp.int32)
		i1, i3, i4 = i2 - 1, i2 + 1, i2 + 2
		indices = self.xp.stack((i1, i2, i3, i4))
		t = (indices - grid).transpose(1, 0, 2, 3, 4)
		dirty_weight = self.xp.sinc(t) * self.xp.sinc(t / 2)
		wh, ww = dirty_weight / dirty_weight.sum(axis=1, keepdims=True)
		weight = wh.reshape(4, 1, batch, 1, height, width) * ww.reshape(1, 4, batch, 1, height, width)
		lower = i1.min(axis=(1, 2, 3))
		upper = i4.max(axis=(1, 2, 3))
		top = abs(min(int(lower[0]), 0))
		left = abs(min(int(lower[1]), 0))
		bottom = max(int(upper[0]) - height + 1, 0)
		right = max(int(upper[1]) - width + 1, 0)
		ih, iw = indices.transpose(1, 0, 2, 3, 4)
		h = ih.reshape(4, 1, batch, 1, height, width) + top
		w = iw.reshape(1, 4, batch, 1, height, width) + left
		j, i, b, c, _, _ = self.xp.indices((1, 1, batch, channels, 1, 1))
		padded_x = pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode="symmetric")
		neighbor = padded_x.reshape(1, 1, *padded_x.shape)[j, i, b, c, h, w]
		return sum(neighbor * weight, axis=(0, 1))
