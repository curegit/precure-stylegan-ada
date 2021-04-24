from math import sqrt as root
from numpy import array, sum, float32
from chainer import Link, Chain, Sequential
from chainer.functions import sqrt, mean, average_pooling_2d, convolution_2d, concat, broadcast_to, flatten, pad
from stylegan.links.common import EqualizedLinear, EqualizedConvolution2D, LeakyRelu
from utilities.math import sinc

class FromRGB(Chain):

	def __init__(self, out_channels):
		super().__init__()
		with self.init_scope():
			self.c = EqualizedConvolution2D(3, out_channels, ksize=1, stride=1, pad=0, gain=1)
			self.a = LeakyRelu()

	def __call__(self, x):
		return self.a(self.c(x))

class Downsampler(Link):

	def __init__(self, lanczos=True, n=2):
		super().__init__()
		self.lanczos = lanczos
		if lanczos:
			self.n = n
			ys = array([self.lanczos_kernel(i + 0.5, n) for i in range(-n, n)])
			ys = ys / sum(ys)
			k = ys.reshape(1, n * 2) * ys.reshape(n * 2, 1)
			self.w = array([[k]], dtype=float32)

	def __call__(self, x):
		if self.lanczos:
			p = self.n - 1
			batch, channels, height, width = x.shape
			h1 = x.reshape(batch * channels, 1, height, width)
			h2 = pad(h1, ((0, 0), (0, 0), (p, p), (p, p)), mode="symmetric")
			h3 = convolution_2d(h2, self.xp.array(self.w), stride=2)
			return h3.reshape(batch, channels, height // 2, width // 2)
		else:
			return average_pooling_2d(x, ksize=2, stride=2)

	def lanczos_kernel(self, x, n):
		return 0.0 if abs(x) > n else sinc(x) * sinc(x / n)

class MiniBatchStandardDeviation(Link):

	def __init__(self, group_size=None):
		super().__init__()
		self.group_size = group_size

	def __call__(self, x):
		batch, channels, height, width = x.shape
		group_size = min(batch, self.group_size or batch)
		groups = batch // group_size
		grouped_x = x.reshape(groups, group_size, channels, height, width)
		variance = mean((grouped_x - mean(grouped_x, axis=1, keepdims=True)) ** 2, axis=1, keepdims=True)
		deviation = mean(sqrt(variance + 1e-08), axis=(2, 3, 4), keepdims=True)
		deviation_map = broadcast_to(deviation, (groups, group_size, 1, height, width)).reshape(batch, 1, height, width)
		return concat((x, deviation_map), axis=1)

class ResidualBlock(Chain):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.c1 = EqualizedConvolution2D(in_channels, in_channels, ksize=3, stride=1, pad=1)
			self.a1 = LeakyRelu()
			self.c2 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
			self.a2 = LeakyRelu()
			self.down = Downsampler(lanczos=False)
			self.skip = Sequential(EqualizedConvolution2D(in_channels, out_channels, ksize=1, stride=1, pad=0, nobias=True), Downsampler())

	def __call__(self, x):
		h = self.a2(self.c2(self.a1(self.c1(x))))
		return (self.skip(x) + self.down(h)) / root(2)

class OutputBlock(Chain):

	def __init__(self, in_channels, group_size=None):
		super().__init__()
		with self.init_scope():
			self.mbstd = MiniBatchStandardDeviation(group_size)
			self.c1 = EqualizedConvolution2D(in_channels + 1, in_channels, ksize=3, stride=1, pad=1)
			self.a1 = LeakyRelu()
			self.c2 = EqualizedConvolution2D(in_channels, in_channels, ksize=4, stride=1, pad=0)
			self.a2 = LeakyRelu()
			self.fc = EqualizedLinear(in_channels, 1, gain=1)

	def __call__(self, x):
		h1 = self.a1(self.c1(self.mbstd(x)))
		h2 = self.a2(self.c2(h1))
		return flatten(self.fc(h2))
