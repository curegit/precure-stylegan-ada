from math import sqrt as root
from numpy import array, pad as padded, float32
from chainer import Parameter, Link, Chain
from chainer.functions import sqrt, sum, convolution_2d, resize_images, broadcast_to, depth2space, pad
from chainer.initializers import Zero, One, Normal
from stylegan.links.common import GaussianDistribution, EqualizedLinear, LeakyRelu

class StyleAffineTransform(Chain):

	def __init__(self, size, channels):
		super().__init__()
		with self.init_scope():
			self.s = EqualizedLinear(size, channels, initial_bias=One(), gain=1)

	def __call__(self, w):
		return self.s(w)

class WeightModulatedConvolution2D(Link):

	def __init__(self, in_channels, out_channels, pointwise=False, demod=True, gain=root(2)):
		super().__init__()
		self.demod = demod
		self.ksize = 1 if pointwise else 3
		self.pad = 0 if pointwise else 1
		self.c = gain * root(1 / (in_channels * self.ksize ** 2))
		with self.init_scope():
			self.w = Parameter(shape=(out_channels, in_channels, self.ksize, self.ksize), initializer=Normal(1.0))
			self.b = Parameter(shape=out_channels, initializer=Zero())

	def __call__(self, x, y):
		out_channels = self.b.shape[0]
		batch, in_channels, height, width = x.shape
		modulated_w = self.w * y.reshape(batch, 1, in_channels, 1, 1)
		w = modulated_w / sqrt(sum(modulated_w ** 2, axis=(2, 3, 4), keepdims=True) + 1e-08) if self.demod else modulated_w
		grouped_w = w.reshape(batch * out_channels, in_channels, self.ksize, self.ksize)
		grouped_x = x.reshape(1, batch * in_channels, height, width)
		padded_grouped_x = pad(grouped_x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="edge")
		h = convolution_2d(padded_grouped_x, grouped_w, stride=1, pad=0, groups=batch)
		return h.reshape(batch, out_channels, height, width) + self.b.reshape(1, out_channels, 1, 1)

class NoiseAdder(Chain):

	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.g = GaussianDistribution()
			self.s = Parameter(shape=1, initializer=Zero())

	def __call__(self, x):
		b, _, h, w = x.shape
		return x + self.s * self.g(b, 1, h, w)

class BicubicUpsampler(Link):

	def __init__(self, b=0.0, c=0.5):
		super().__init__()
		self.b, self.c = b, c
		s = array([self.kernel(i + 0.25) for i in range(-2, 2)])
		e = array([self.kernel(i + 0.75) for i in range(-2, 2)])
		k1 = padded(s.reshape(1, 4) * s.reshape(4, 1), ((0, 1), (0, 1)))
		k2 = padded(e.reshape(1, 4) * s.reshape(4, 1), ((0, 1), (1, 0)))
		k3 = padded(s.reshape(1, 4) * e.reshape(4, 1), ((1, 0), (0, 1)))
		k4 = padded(e.reshape(1, 4) * e.reshape(4, 1), ((1, 0), (1, 0)))
		self.w = array([[k1], [k2], [k3], [k4]], dtype=float32)

	def __call__(self, x):
		batch, channels, height, width = x.shape
		h1 = x.reshape(batch * channels, 1, height, width)
		h2 = pad(h1, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="symmetric")
		h3 = convolution_2d(h2, self.xp.array(self.w))
		h4 = depth2space(h3, 2)
		return h4.reshape(batch, channels, height * 2, width * 2)

	def kernel(self, x):
		b, c = self.b, self.c
		if abs(x) < 1:
			return 1 / 6 * ((12 - 9 * b - 6 * c) * abs(x) ** 3 + (-18 + 12 * b + 6 * c) * abs(x) ** 2 + (6 - 2 * b))
		elif 1 <= abs(x) < 2:
			return 1 / 6 * ((-b - 6 * c) * abs(x) ** 3 + (6 * b + 30 * c) * abs(x) ** 2 + (-12 * b - 48 * c) * abs(x) + (8 * b + 24 * c))
		else:
			return 0.0

class ToRGB(Chain):

	def __init__(self, in_channels):
		super().__init__()
		with self.init_scope():
			self.w = WeightModulatedConvolution2D(in_channels, 3, pointwise=True, demod=False, gain=1)

	def __call__(self, x, y):
		return self.w(x, y)

class InitialSkipArchitecture(Chain):

	def __init__(self, size, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.c1 = Parameter(shape=(in_channels, 4, 4), initializer=Normal(1.0))
			self.s1 = StyleAffineTransform(size, in_channels)
			self.w1 = WeightModulatedConvolution2D(in_channels, out_channels)
			self.n1 = NoiseAdder()
			self.a1 = LeakyRelu()
			self.s2 = StyleAffineTransform(size, out_channels)
			self.trgb = ToRGB(out_channels)

	def __call__(self, w):
		batch = w.shape[0]
		h1 = self.c1.reshape(1, *self.c1.shape)
		h2 = broadcast_to(h1, (batch, *self.c1.shape))
		h3 = self.w1(h2, self.s1(w))
		h4 = self.n1(h3)
		h5 = self.a1(h4)
		return h5, self.trgb(h5, self.s2(w))

class SkipArchitecture(Chain):

	def __init__(self, size, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.up = BicubicUpsampler()
			self.s1 = StyleAffineTransform(size, in_channels)
			self.w1 = WeightModulatedConvolution2D(in_channels, out_channels)
			self.n1 = NoiseAdder()
			self.a1 = LeakyRelu()
			self.s2 = StyleAffineTransform(size, out_channels)
			self.w2 = WeightModulatedConvolution2D(out_channels, out_channels)
			self.n2 = NoiseAdder()
			self.a2 = LeakyRelu()
			self.s3 = StyleAffineTransform(size, out_channels)
			self.trgb = ToRGB(out_channels)
			self.skip = BicubicUpsampler(1, 0)

	def __call__(self, x, y, w):
		h1 = self.up(x)
		h2 = self.w1(h1, self.s1(w))
		h3 = self.n1(h2)
		h4 = self.a1(h3)
		h5 = self.w2(h4, self.s2(w))
		h6 = self.n2(h5)
		h7 = self.a2(h6)
		return h7, self.skip(y) + self.trgb(h7, self.s3(w))
