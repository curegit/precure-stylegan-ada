from chainer import Link, Chain, Sequential
from chainer.functions import sqrt, mean, resize_images, concat, broadcast_to, flatten
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

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		height, width = x.shape[2:]
		return resize_images(x, (height // 2, width // 2), align_corners=False)

class MiniBatchStandardDeviation(Link):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		m = broadcast_to(mean(x, axis=0, keepdims=True), x.shape)
		sd = sqrt(mean((x - m) ** 2, axis=0, keepdims=True) + 1e-8)
		dev_channel = broadcast_to(mean(sd), (x.shape[0], 1, x.shape[2], x.shape[3]))
		return concat((x, dev_channel), axis=1)

class ResidualBlock(Chain):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.c1 = EqualizedConvolution2D()
			self.a1 = LeakyRelu()
			self.c2 = EqualizedConvolution2D()
			self.a2 = LeakyRelu()
			self.down = Downsampler()
			self.skip = Sequential(Downsampler(), EqualizedConvolution2D())

	def __call__(self, x):
		y = self.a2(self.c2(self.a1(self.c1(x))))
		return self.skip(x) + self.down(y)

class OutputBlock(Chain):

	def __init__(self, in_channels):
		super().__init__()
		with self.init_scope():
			self.mbstd = MiniBatchStandardDeviation()
			self.c1 = EqualizedConvolution2D(in_channels + 1, in_channels, ksize=3, stride=1, pad=1)
			self.a1 = LeakyRelu()
			self.c2 = EqualizedConvolution2D(in_channels, in_channels, ksize=4, stride=1, pad=0)
			self.a2 = LeakyRelu()
			self.fc1 = EqualizedLinear(in_channels, in_channels)
			self.a3 = LeakyRelu()
			self.fc2 = EqualizedLinear(in_channels, 1, gain=1)

	def __call__(self, x):
		h1 = self.a1(self.c1(self.mbstd(x)))
		h2 = self.a2(self.c2(h1))
		h3 = self.fc2(self.a3(self.fc1(h2)))
		return flatten(h3)
