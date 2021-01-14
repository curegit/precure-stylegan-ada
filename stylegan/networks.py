from random import randint
from chainer import Chain, ChainList, Sequential
from chainer.functions import sqrt, mean
from chainer.serializers import load_hdf5, save_hdf5
from stylegan.links.common import GaussianDistribution, EqualizedLinear, LeakyRelu
from stylegan.links.generator import InitialSkipArchitecture, SkipArchitecture
from stylegan.links.discriminator import FromRGB, ResidualBlock, OutputBlock

class Network(Chain):

	def load_state(self, filepath):
		load_hdf5(filepath, self)

	def save_state(self, filepath):
		save_hdf5(filepath, self)

class Mapper(Chain):

	def __init__(self, size, depth):
		super().__init__()
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(size, size), LeakyRelu()).repeat(depth)

	def __call__(self, z):
		return self.mlp(z / sqrt(mean(z ** 2, axis=1, keepdims=True) + 1e-8))

class Synthesizer(Chain):

	def __init__(self, size, levels, first_channels, last_channels, double_last):
		super().__init__()
		in_channels = [first_channels] * levels
		out_channels = [last_channels] * levels
		for i in range(1, levels):
			channels = min(first_channels, last_channels * 2 ** i)
			in_channels[-i] = channels
			out_channels[-i - 1] = channels
		if double_last:
			out_channels[-1] *= 2
		with self.init_scope():
			self.init = InitialSkipArchitecture(size, in_channels[0], out_channels[0])
			self.skips = ChainList(*[SkipArchitecture(size, i, o) for i, o in zip(in_channels[1:], out_channels[1:])])

	def __call__(self, ws):
		h, rgb = self.init(ws[0])
		for s, w in zip(self.skips, ws[1:]):
			h, rgb = s(h, rgb, w)
		return rgb

class Generator(Network):

	def __init__(self, size=512, depth=8, levels=7, first_channels=512, last_channels=16, double_last=True):
		super().__init__()
		self.size = size
		self.levels = levels
		self.resolution = (2 * 2 ** levels, 2 * 2 ** levels)
		with self.init_scope():
			self.sampler = GaussianDistribution()
			self.mapper = Mapper(size, depth)
			self.synthesizer = Synthesizer(size, levels, first_channels, last_channels, double_last)

	def __call__(self, z, *zs, random_mix=None):
		w = self.mapper(z)
		ws = [w] * self.levels
		stop = self.levels
		if self.levels > 1 and random_mix is not None:
			mix_level = randint(1, self.levels - 1)
			mix_w = self.mapper(random_mix)
			ws[mix_level:stop] = [mix_w] * (stop - mix_level)
			stop = mix_level
		for i, z in zip(range(1, stop), zs):
			if z is not Ellipsis:
				ws[i:stop] = [self.mapper(z)] * (stop - i)
		return self.synthesizer(ws)

	def generate_latents(self, batch):
		return self.sampler(batch, self.size)

class Discriminator(Network):

	def __init__(self, levels=7, first_channels=16, last_channels=512):
		super().__init__()
		in_channels = [first_channels] * (levels - 1)
		out_channels = [last_channels] * (levels - 1)
		for i in range(1, levels - 1):
			channels = min(first_channels * 2 ** i, last_channels)
			in_channels[i] = channels
			out_channels[i - 1] = channels
		with self.init_scope():
			self.frgb = FromRGB(first_channels)
			self.blocks = Sequential(*[ResidualBlock(i, o) for i, o in zip(in_channels, out_channels)])
			self.output = OutputBlock(last_channels)

	def __call__(self, x):
		return self.output(self.blocks(self.frgb(x)))
