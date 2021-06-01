from math import sqrt as root
from numpy import array, sinc, float32
from chainer import Chain
from chainer.functions import sqrt, mean, average_pooling_2d, convolution_2d, concat, broadcast_to, pad
from stylegan.layers.basic import LeakyRelu, EqualizedLinear, EqualizedConvolution2D

class Downsampler():

	def __init__(self, lanczos=True, n=2):
		self.lanczos = lanczos
		if lanczos:
			self.n = n
			kernel = lambda x: sinc(x) * sinc(x / n)
			ys = array([kernel(i + 0.5) for i in range(-n, n)])
			k = ys.reshape(1, n * 2) * ys.reshape(n * 2, 1)
			self.w = array([[k / k.sum()]], dtype=float32)

	def __call__(self, x):
		if self.lanczos:
			p = self.n - 1
			batch, channels, height, width = x.shape
			h1 = x.reshape(batch * channels, 1, height, width)
			h2 = pad(h1, ((0, 0), (0, 0), (p, p), (p, p)), mode="symmetric")
			h3 = convolution_2d(h2, x.xp.asarray(self.w), stride=2)
			return h3.reshape(batch, channels, height // 2, width // 2)
		else:
			return average_pooling_2d(x, ksize=2, stride=2)

class MinibatchStandardDeviation():

	def __init__(self, group_size=None):
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

class FromRGB(Chain):

	def __init__(self, out_channels):
		super().__init__()
		with self.init_scope():
			self.conv = EqualizedConvolution2D(3, out_channels, ksize=1, stride=1, pad=0)
			self.act = LeakyRelu()

	def __call__(self, x):
		return self.act(self.conv(x))

class ResidualBlock(Chain):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		with self.init_scope():
			self.conv1 = EqualizedConvolution2D(in_channels, in_channels, ksize=3, stride=1, pad=1)
			self.act1 = LeakyRelu()
			self.conv2 = EqualizedConvolution2D(in_channels, out_channels, ksize=3, stride=1, pad=1)
			self.act2 = LeakyRelu()
			self.pool = Downsampler(lanczos=False)
			self.conv3 = EqualizedConvolution2D(in_channels, out_channels, ksize=1, stride=1, pad=0, nobias=True, gain=1)
			self.down = Downsampler()

	def __call__(self, x):
		h = self.pool(self.act2(self.conv2(self.act1(self.conv1(x)))))
		skip = self.down(self.conv3(x))
		return (h + skip) / root(2)

class OutputBlock(Chain):

	def __init__(self, in_channels, conditional=False, group_size=None):
		super().__init__()
		with self.init_scope():
			self.mbstd = MinibatchStandardDeviation(group_size)
			self.conv1 = EqualizedConvolution2D(in_channels + 1, in_channels, ksize=3, stride=1, pad=1)
			self.act1 = LeakyRelu()
			self.conv2 = EqualizedConvolution2D(in_channels, in_channels, ksize=4, stride=1, pad=0)
			self.act2 = LeakyRelu()
			self.linear = EqualizedLinear(in_channels, in_channels if conditional else 1, gain=1)

	def __call__(self, x):
		h1 = self.act1(self.conv1(self.mbstd(x)))
		h2 = self.act2(self.conv2(h1))
		return self.linear(h2)
