from math import sqrt, log, sin, cos, pi
from numpy import array, eye, outer, stack, float32
from numpy.random import uniform, normal, lognormal
from chainer.functions import concat
from stylegan.manipulations.base import Manipulation

identity = eye(4, dtype=float32)

def translation(x, y, z):
	return array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float32)

def rotation(axis, theta):
	norm = sqrt(sum(a ** 2 for a in axis))
	x, y, z = (a / norm for a in axis)
	s, c, d = sin(theta), cos(theta), 1 - cos(theta)
	return array([[c + d * x ** 2, x * y * d - z * s, z * x * d + y * s, 0], [x * y * d + z * s, c + d * y ** 2, y * z *d - x * s, 0], [z * x * d - y * s, y * z * d + x * s, c + d * z ** 2, 0], [0, 0, 0, 1]], dtype=float32)

def scaling(s, t, u):
	return array([[s, 0, 0, 0], [0, t, 0, 0], [0, 0, u, 0], [0, 0, 0, 1]], dtype=float32)

def householder(axis):
	norm = sqrt(sum(a ** 2 for a in axis))
	v = array([*axis, 0], dtype=float32) / norm
	return identity - 2 * outer(v, v)


class ColorAffineTransformation(Manipulation):

	def __init__(self, brightness=0.1, contrast=0.5, hue_rotation=360, saturation=1.0, probability_multiplier=1.0):
		super().__init__()
		self.brightness = brightness
		self.contrast = contrast * log(2)
		self.hue_rotation = hue_rotation / 360 * pi
		self.saturation = saturation * log(2)
		self.probability_multiplier = probability_multiplier

	def __call__(self, x, p=1.0):
		p *= self.probability_multiplier
		if p <= 0:
			return x
		batch, channels, height, width = x.shape
		centering = translation(-0.5, -0.5, -0.5)
		affine = centering @ identity
		condition = uniform(size=batch) < p
		brightness = normal(scale=self.brightness, size=batch)
		brightness_adjustment = stack([translation(b, b, b) if c else identity for c, b in zip(condition, brightness)])
		affine = brightness_adjustment @ affine
		condition = uniform(size=batch) < p
		contrast = lognormal(sigma=self.contrast, size=batch)
		contrast_adjustment = stack([scaling(s, s, s) if c else identity for c, s in zip(condition, contrast)])
		affine = contrast_adjustment @ affine
		condition = uniform(size=batch) < p / 2
		luminance_flip = stack([householder([1, 1, 1]) if c else identity for c in condition])
		affine = luminance_flip @ affine
		condition = uniform(size=batch) < p
		theta = uniform(low=-self.hue_rotation, high=self.hue_rotation, size=batch)
		hue_rotation = stack([rotation([1, 1, 1], t) if c else identity for c, t in zip(condition, theta)])
		affine = hue_rotation @ affine
		condition = uniform(size=batch) < p
		saturation = lognormal(sigma=self.saturation, size=batch)
		v = array([1, 1, 1, 0], dtype=float32) / sqrt(3)
		o = outer(v, v)
		saturation_adjustment = stack([o + (identity - o) * array([s, s, s, 1], dtype=float32) if c else identity for c, s in zip(condition, saturation)])
		affine = saturation_adjustment @ affine
		inverse_centering = translation(0.5, 0.5, 0.5)
		affine = inverse_centering @ affine
		affine = affine[:, 0:3].reshape(batch, 1, 3, 4)
		affine = self.xp.asarray(affine)
		ones = self.xp.ones((batch, height * width, 1, 1), dtype=self.xp.float32)
		h1 = x.transpose(0, 2, 3, 1).reshape(batch, height * width, channels, 1)
		h2 = concat((h1, ones), axis=2)
		h3 = (affine @ h2).reshape(batch, height, width, channels)
		return h3.transpose(0, 3, 1, 2)
