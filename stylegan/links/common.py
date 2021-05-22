from math import sqrt, log
from chainer import Link, Chain
from chainer.links import Linear, Convolution2D
from chainer.functions import gaussian, leaky_relu, broadcast_to
from chainer.initializers import Normal

class EqualizedLinear(Chain):

	def __init__(self, in_size, out_size, initial_bias=None, gain=sqrt(2)):
		super().__init__()
		self.c = gain * sqrt(1 / in_size)
		with self.init_scope():
			self.linear = Linear(in_size, out_size, initialW=Normal(1.0), initial_bias=initial_bias)

	def __call__(self, x):
		return self.linear(self.c * x)

class EqualizedConvolution2D(Chain):

	def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad=0, nobias=False, initial_bias=None, gain=sqrt(2)):
		super().__init__()
		self.c = gain * sqrt(1 / (in_channels * ksize ** 2))
		with self.init_scope():
			self.conv = Convolution2D(in_channels, out_channels, ksize, stride, pad, nobias=nobias, initialW=Normal(1.0), initial_bias=initial_bias)

	def __call__(self, x):
		return self.conv(self.c * x)

class LeakyRelu(Link):

	def __init__(self, a=0.2):
		super().__init__()
		self.a = a

	def __call__(self, x):
		return leaky_relu(x, self.a)
