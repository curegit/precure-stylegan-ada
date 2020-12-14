from numpy import load, save
from chainer import Variable, Chain, ChainList, Sequential
from chainer.backend import CpuDevice
from chainer.functions import sqrt, mean
from chainer.serializers import load_hdf5, save_hdf5
from stylegan.links.common import GaussianDistribution, EqualizedLinear, LeakyRelu
from stylegan.links.generator import InitialSkipArchitecture, SkipArchitecture
from stylegan.links.discriminator import FromRGB, ResidualBlock, OutputBlock

class Network(Chain):

	def load_model(self, filepath=None):
		if filepath is not None:
			load_hdf5(filepath, self)
		return self

	def save_model(self, filepath):
		device = self.device
		self.to_device(CpuDevice())
		save_hdf5(filepath, self)
		return self.to_device(device)

	def load_variable(self, filepath):
		x = Variable(load(filepath))
		x.to_device(self.device)
		return x

	def save_variable(self, filepath, x):
		x.to_device(CpuDevice())
		save(filepath, x.array)
		x.to_device(self.device)
		return x

class Mapper(Network):

	def __init__(self, size, depth):
		super().__init__()
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(size, size), LeakyRelu()).repeat(depth)

	def __call__(self, z):
		return self.mlp(z / sqrt(mean(z ** 2, axis=1, keepdims=True) + 1e-8))

class Synthesizer(Network):

	def __init__(self, size, levels=7, first_channels=512, last_channels=16, double_last=True):
		super().__init__()
		in_channels = [first_channels if i == 1 else min(first_channels, last_channels * 2 ** (levels - i + 1)) for i in range(1, levels + 1)]
		out_channels = [last_channels * (2 if double_last else 1) if i == levels else min(first_channels, last_channels * 2 ** (levels - i)) for i in range(1, levels + 1)]
		skips = [SkipArchitecture(size, i, o) for i, o in zip(in_channels[1:], out_channels[1:])]
		with self.init_scope():
			self.init = InitialSkipArchitecture(size, in_channels[0], out_channels[0])
			self.skips = ChainList(*skips)

	def __call__(self, ws):
		h, rgb = self.init(ws[0])
		for s, w in zip(self.skips, ws[1:]):
			h, rgb = s(h, rgb, w)
		return rgb

class Generator(Network):

	def __init__(self, latent_size=512, depth=8, levels=7, first_channels=512, last_channels=16, double_last=True):
		super().__init__()
		self.latent_size = latent_size
		self.levels = levels
		self.resolution = (2 * 2 ** levels, 2 * 2 ** levels)
		with self.init_scope():
			self.sampler = GaussianDistribution()
			self.mapper = Mapper(latent_size, depth)
			self.synthesizer = Synthesizer(latent_size, levels, first_channels, last_channels, double_last)

	def __call__(self, z):
		return self.synthesizer([self.mapper(z)] * self.levels)

	def generate_latents(self, batch):
		return self.sampler((batch, self.latent_size))

class Discriminator(Network):

	def __init__(self, levels=7, first_channels=16, last_channels=512):
		super().__init__()
		in_channels = [first_channels if i == levels else min(first_channels * 2 ** (levels - i), last_channels) for i in range(levels, 1, -1)]
		out_channels = [last_channels if i == 2 else min(first_channels * 2 ** (levels - i + 1), last_channels) for i in range(levels, 1, -1)]
		blocks = [ResidualBlock(i, o) for i, o in zip(in_channels, out_channels)]
		with self.init_scope():
			self.frgb = FromRGB(first_channels)
			self.blocks = Sequential(*blocks)
			self.output = OutputBlock(last_channels)

	def __call__(self, x):
		return self.output(self.blocks(self.frgb(x)))
