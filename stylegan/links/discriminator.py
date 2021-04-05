from math import sqrt as root
from chainer import Link, Chain, Sequential
from chainer.functions import sqrt, mean, average_pooling_2d, concat, broadcast_to, flatten
from stylegan.links.common import EqualizedLinear, EqualizedConvolution2D, LeakyRelu

class FromRGB(Chain):

	def __init__(self, out_channels):
		super().__init__()
		with self.init_scope():
			self.c = EqualizedConvolution2D(3, out_channels, ksize=1, stride=1, pad=0, gain=1)
			self.a = LeakyRelu()

	def __call__(self, x):
		return self.a(self.c(x))

class Downsampler(Link):

	def __call__(self, x):
		return average_pooling_2d(x, ksize=2, stride=2)

class MiniBatchStandardDeviation(Link):

	def __init__(self, group_size=4):
		super().__init__()
		self.group_size = group_size

	def __call__(self, x):
		batch, channels, height, width = x.shape
		group = batch // self.group_size
		y = x.reshape(group, self.group_size, channels, height, width)
		y = y - mean(y, axis=1, keepdims=True)
		y = mean(sqrt(y ** 2), axis=1, keepdims=True)
		y = mean(y, axis=(2, 3, 4), keepdims=True)
		y = broadcast_to(y, (group, self.group_size, 1, height, width))
		dev = y.reshape(batch, 1, height, width)
		return concat((x, dev), axis=1)

class ResidualBlock(Chain):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.c1 = EqualizedConvolution2D(in_channels, in_channels, ksize=3, stride=1, pad=1)
			self.a1 = LeakyRelu()
			self.c2 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
			self.a2 = LeakyRelu()
			self.down = Downsampler()
			self.skip = Sequential(Downsampler(), EqualizedConvolution2D(in_channels, out_channels, ksize=1, stride=1, pad=0, nobias=True))

	def __call__(self, x):
		h = self.a2(self.c2(self.a1(self.c1(x))))
		return (self.skip(x) + self.down(h)) / root(2)

class OutputBlock(Chain):

	def __init__(self, in_channels):
		super().__init__()
		with self.init_scope():
			self.mbstd = MiniBatchStandardDeviation()
			self.c1 = EqualizedConvolution2D(in_channels + 1, in_channels, ksize=3, stride=1, pad=1)
			self.a1 = LeakyRelu()
			self.c2 = EqualizedConvolution2D(in_channels, in_channels, ksize=4, stride=1, pad=0)
			self.a2 = LeakyRelu()
			self.fc = EqualizedLinear(in_channels, 1, gain=1)

	def __call__(self, x):
		h1 = self.a1(self.c1(self.mbstd(x)))
		h2 = self.a2(self.c2(h1))
		return flatten(self.fc(h2))
