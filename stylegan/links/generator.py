from math import sqrt as root
from chainer import Parameter, Link, Chain
from chainer.links import Scale
from chainer.functions import sqrt, sum, convolution_2d, resize_images, broadcast_to
from chainer.initializers import Zero, One, Normal
from stylegan.links.common import Constant, GaussianDistribution, EqualizedLinear, EqualizedConvolution2D, LeakyRelu

class ToRGB(Chain):

	def __init__(self, in_channels):
		super().__init__()
		with self.init_scope():
			self.c = EqualizedConvolution2D(in_channels, 3, ksize=1, stride=1, pad=0, gain=1)

	def __call__(self, x):
		return self.c(x)

class Upsampler(Link):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		height, width = x.shape[2:]
		return resize_images(x, (height * 2, width * 2), align_corners=False)

class StyleAffineTransform(Chain):

	def __init__(self, latent_size):
		super().__init__()
		with self.init_scope():
			self.s = EqualizedLinear(latent_size, latent_size, initial_bias=One(), gain=1)

	def __call__(self, w):
		return self.s(w)

class WeightDemodulatedConvolution2D(Link):

	def __init__(self, in_channels, out_channels, gain=root(2)):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.c = gain * root(1 / (in_channels * 3 ** 2))
		with self.init_scope():
			self.w = Parameter(shape=(out_channels, in_channels, 3, 3), initializer=Normal(1.0))
			self.b = Parameter(shape=out_channels, initializer=Zero())

	def __call__(self, x, y):
		batch, _, height, width = x.shape
		mod_w = y.reshape((batch, 1, self.in_channels, 1, 1)) * self.w
		demod_w = mod_w / sqrt(sum(mod_w ** 2, axis=(2, 3, 4), keepdims=True) + 1e-8)
		w = demod_w.reshape((batch * self.out_channels, self.in_channels, 3, 3))
		group_x = self.c * x.reshape((1, batch * self.in_channels, height, width))
		h = convolution_2d(group_x, w, self.b, stride=1, pad=1, groups=batch)
		return h.reshape((batch, self.out_channels, height, width))

class NoiseAdder(Chain):

	def __init__(self, channels):
		super().__init__()
		with self.init_scope():
			self.g = GaussianDistribution()
			self.s = Scale(W_shape=channels)

	def __call__(self, x):
		b, c, h, w = x.shape
		n = broadcast_to(self.g((b, 1, h, w)), x.shape)
		return x + self.s(n)

class InitialSkipArchitecture(Chain):

	def __init__(self, in_channels, out_channels, latent_size):
		super().__init__()
		self.in_channels = in_channels
		with self.init_scope():
			self.c1 = Constant(1)
			self.s1 = StyleAffineTransform(latent_size, latent_size)
			self.w1 = WeightDemodulatedConvolution2D(in_channels, out_channels)
			self.n1 = NoiseAdder(out_channels)
			self.a1 = LeakyRelu()
			self.trgb = ToRGB(out_channels)

	def __call__(self, w):
		batch = w.shape[0]
		h1 = self.c1((batch, self.in_channels, 4, 4))
		h2 = self.w1(h1, self.s1(w))
		h3 = self.n1(h2)
		h4 = self.a1(h3)
		return h4, self.trgb(h4)

class SkipArchitecture(Chain):

	def __init__(self, in_channels, out_channels, latent_size):
		super().__init__()
		with self.init_scope():
			self.up = Upsampler()
			self.s1 = StyleAffineTransform(latent_size, latent_size)
			self.w1 = WeightDemodulatedConvolution2D(in_channels, out_channels)
			self.n1 = NoiseAdder(out_channels)
			self.a1 = LeakyRelu()
			self.s2 = StyleAffineTransform(latent_size, latent_size)
			self.w2 = WeightDemodulatedConvolution2D(out_channels, out_channels)
			self.n2 = NoiseAdder(out_channels)
			self.a2 = LeakyRelu()
			self.trgb = ToRGB(out_channels)
			self.skip = Upsampler()

	def __call__(self, x, y, w):
		h1 = self.up(x)
		h2 = self.w1(h1, self.s1(w))
		h3 = self.n1(h2)
		h4 = self.a1(h3)
		h5 = self.w2(h4, self.s2(w))
		h6 = self.n2(h5)
		h7 = self.a2(h6)
		return h7, self.skip(y) + self.trgb(h7)
