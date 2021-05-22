from math import sqrt
from chainer import Parameter, Link
from chainer.functions import leaky_relu, linear, convolution_2d
from chainer.initializers import Zero, Normal

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
		return linear(x, self.c * self.w, self.b)

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
		return convolution_2d(x, self.c * self.w, self.b, self.stride, self.pad)
