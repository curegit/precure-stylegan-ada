from numpy import array, pad as padded, float32
from chainer import Parameter, Link, Chain
from chainer.functions import sqrt, sum, convolution_2d, broadcast_to, depth2space, pad
from chainer.initializers import Zero, One, Normal
from stylegan.layers.basic import GaussianDistribution, LeakyRelu, EqualizedLinear

class LearnableConstant(Link):

	def __init__(self, channels):
		super().__init__()
		with self.init_scope():
			self.c = Parameter(shape=(channels, 4, 4), initializer=Normal(1.0))

	def __call__(self, batch):
		return broadcast_to(self.c, (batch, *self.c.shape))


class BicubicUpsampler:

	def __init__(self, b=0.0, c=0.5):
		self.b, self.c = b, c
		ys1 = array([self.kernel(i + 0.25) for i in range(-2, 2)])
		ys2 = array([self.kernel(i + 0.75) for i in range(-2, 2)])
		k1 = padded(ys1.reshape(1, 4) * ys1.reshape(4, 1), ((0, 1), (0, 1)))
		k2 = padded(ys2.reshape(1, 4) * ys1.reshape(4, 1), ((0, 1), (1, 0)))
		k3 = padded(ys1.reshape(1, 4) * ys2.reshape(4, 1), ((1, 0), (0, 1)))
		k4 = padded(ys2.reshape(1, 4) * ys2.reshape(4, 1), ((1, 0), (1, 0)))
		self.w = array([[k1], [k2], [k3], [k4]], dtype=float32)

	def __call__(self, x):
		batch, channels, height, width = x.shape
		h1 = x.reshape(batch * channels, 1, height, width)
		h2 = pad(h1, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="symmetric")
		h3 = convolution_2d(h2, x.xp.asarray(self.w))
		h4 = depth2space(h3, 2)
		return h4.reshape(batch, channels, height * 2, width * 2)

	def kernel(self, x):
		if abs(x) < 1:
			return 1 / 6 * ((12 - 9 * self.b - 6 * self.c) * abs(x) ** 3 + (-18 + 12 * self.b + 6 * self.c) * abs(x) ** 2 + (6 - 2 * self.b))
		elif 1 <= abs(x) < 2:
			return 1 / 6 * ((-self.b - 6 * self.c) * abs(x) ** 3 + (6 * self.b + 30 * self.c) * abs(x) ** 2 + (-12 * self.b - 48 * self.c) * abs(x) + (8 * self.b + 24 * self.c))
		else:
			return 0.0


class StyleAffineTransformation(Chain):

	def __init__(self, size, channels):
		super().__init__()
		with self.init_scope():
			self.linear = EqualizedLinear(size, channels, initial_bias=One(), gain=1)

	def __call__(self, w):
		return self.linear(w)


class WeightModulatedConvolution(Link):

	def __init__(self, in_channels, out_channels, pointwise=False, demod=True):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.demod = demod
		self.ksize = 1 if pointwise else 3
		self.pad = 0 if pointwise else 1
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
		padded_grouped_x = pad(grouped_x, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode="edge") if self.pad else grouped_x
		h = convolution_2d(padded_grouped_x, grouped_w, stride=1, pad=0, groups=batch)
		return h.reshape(batch, out_channels, height, width) + self.b.reshape(1, out_channels, 1, 1)


class NoiseAdder(Link):

	def __init__(self, id=0):
		super().__init__()
		with self.init_scope():
			self.id = id
			self.sampler = GaussianDistribution(self)
			self.s = Parameter(shape=1, initializer=Zero())

	def __call__(self, x, coefficient=1.0, fixed=None):
		if coefficient == 0.0:
			return x
		else:
			batch, channels, height, width = x.shape
			scale = self.s if coefficient == 1.0 else coefficient * self.s
			if fixed is None:
				return x + scale * self.sampler(batch, 1, height, width)
			else:
				return x + scale * self.sampler.deterministic(1, 1, height, width, seed=(fixed + self.id))


class ToRGB(Chain):

	def __init__(self, in_channels):
		super().__init__()
		with self.init_scope():
			self.wmconv = WeightModulatedConvolution(in_channels, 3, pointwise=True, demod=False)

	def __call__(self, x, y):
		return self.wmconv(x, y)


class InitialSkipArchitecture(Chain):

	def __init__(self, size, in_channels, out_channels, level=1):
		super().__init__()
		with self.init_scope():
			self.const = LearnableConstant(in_channels)
			self.style1 = StyleAffineTransformation(size, in_channels)
			self.wmconv = WeightModulatedConvolution(in_channels, out_channels)
			self.noise = NoiseAdder(level * 10 + 1)
			self.act = LeakyRelu()
			self.style2 = StyleAffineTransformation(size, out_channels)
			self.torgb = ToRGB(out_channels)

	def __call__(self, w, noise=1.0, fixed=None):
		h1 = self.const(w.shape[0])
		h2 = self.wmconv(h1, self.style1(w))
		h3 = self.noise(h2, coefficient=noise, fixed=fixed)
		h4 = self.act(h3)
		return h4, self.torgb(h4, self.style2(w))

	@property
	def channels(self):
		yield self.wmconv.in_channels
		yield self.wmconv.out_channels


class SkipArchitecture(Chain):

	def __init__(self, size, in_channels, out_channels, level=2):
		super().__init__()
		with self.init_scope():
			self.up = BicubicUpsampler()
			self.style1 = StyleAffineTransformation(size, in_channels)
			self.wmconv1 = WeightModulatedConvolution(in_channels, out_channels)
			self.noise1 = NoiseAdder(level * 10 + 1)
			self.act1 = LeakyRelu()
			self.style2 = StyleAffineTransformation(size, out_channels)
			self.wmconv2 = WeightModulatedConvolution(out_channels, out_channels)
			self.noise2 = NoiseAdder(level * 10 + 2)
			self.act2 = LeakyRelu()
			self.style3 = StyleAffineTransformation(size, out_channels)
			self.torgb = ToRGB(out_channels)
			self.skip = BicubicUpsampler(1, 0)

	def __call__(self, x, y, w, noise=1.0, fixed=None):
		h1 = self.up(x)
		h2 = self.wmconv1(h1, self.style1(w))
		h3 = self.noise1(h2, coefficient=noise, fixed=fixed)
		h4 = self.act1(h3)
		h5 = self.wmconv2(h4, self.style2(w))
		h6 = self.noise2(h5, coefficient=noise, fixed=fixed)
		h7 = self.act2(h6)
		return h7, self.skip(y) + self.torgb(h7, self.style3(w))

	@property
	def channels(self):
		yield self.wmconv1.in_channels
		yield self.wmconv1.out_channels
		yield self.wmconv2.out_channels
