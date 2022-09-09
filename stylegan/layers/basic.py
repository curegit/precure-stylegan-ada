from math import sqrt, log
from numpy import float32
from numpy.random import Generator, PCG64
from chainer import Variable, Parameter, Link
from chainer.functions import gaussian, leaky_relu, linear, convolution_2d, broadcast_to
from chainer.initializers import Zero, Normal

class GaussianDistribution():

	def __init__(self, link, mean=0.0, sd=1.0):
		self.link = link
		self.mean = mean
		self.sd = sd
		self.ln_var = log(sd ** 2)

	def __call__(self, *shape):
		mean = self.link.xp.array(self.mean, dtype=self.link.xp.float32)
		ln_var = self.link.xp.array(self.ln_var, dtype=self.link.xp.float32)
		return gaussian(broadcast_to(mean, shape), broadcast_to(ln_var, shape))

	def deterministic(self, *shape, seed=0):
		x = Generator(PCG64(seed)).normal(size=shape, loc=self.mean, scale=self.sd).astype(float32)
		return Variable(self.link.xp.asarray(x))

class LeakyRelu():

	def __init__(self, a=0.2):
		self.a = a

	def __call__(self, x):
		return leaky_relu(x, self.a)

class EqualizedLinear(Link):

	def __init__(self, in_size, out_size, nobias=False, initial_bias=Zero(), gain=sqrt(2)):
		super().__init__()
		self.c = gain / sqrt(in_size)
		with self.init_scope():
			self.w = Parameter(shape=(out_size, in_size), initializer=Normal(1.0))
			self.b = None if nobias else Parameter(shape=out_size, initializer=initial_bias)

	def __call__(self, x):
		return linear(self.c * x, self.w, self.b)

class EqualizedConvolution2D(Link):

	def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0, nobias=False, initial_bias=Zero(), gain=sqrt(2)):
		super().__init__()
		self.stride = stride
		self.pad = pad
		self.c = gain / sqrt(in_channels * ksize ** 2)
		with self.init_scope():
			self.w = Parameter(shape=(out_channels, in_channels, ksize, ksize), initializer=Normal(1.0))
			self.b = None if nobias else Parameter(shape=out_channels, initializer=initial_bias)

	def __call__(self, x):
		return convolution_2d(self.c * x, self.w, self.b, self.stride, self.pad)
