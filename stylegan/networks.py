from chainer import Chain, ChainList, Sequential
from chainer.functions import sqrt, mean
from stylegan.links.common import GaussianDistribution, EqualizedLinear, LeakyRelu
from stylegan.links.generator import InitialSkipArchitecture, SkipArchitecture
from stylegan.links.discriminator import FromRGB, ResidualBlock, OutputBlock

class Mapper(Chain):

	def __init__(self, latent_size, depth):
		super().__init__()
		with self.init_scope():
			self.mlp = Sequential(EqualizedLinear(latent_size, latent_size), LeakyRelu()).repeat(depth)

	def __call__(self, z):
		norm_z = z / sqrt(mean(z ** 2, axis=1, keepdims=True) + 1e-8)
		return self.mlp(norm_z)

class Synthesizer(Chain):

	def __init__(self, latent_size, levels=7, first_channels=512, last_channels=16):
		super().__init__()
		in_channels = [first_channels if i == 1 else min(first_channels, last_channels * 2 ** (levels - i + 1)) for i in range(1, levels + 1)]
		out_channels = [last_channels if i == levels else min(first_channels, last_channels * 2 ** (levels - i)) for i in range(1, levels + 1)]
		skips = [SkipArchitecture(latent_size, i, o) for i, o in zip(in_channels[1:], out_channels[1:])]
		with self.init_scope():
			self.init = InitialSkipArchitecture(latent_size, in_channels[0], out_channels[0])
			self.skips = ChainList(*skips)

	def __call__(self, ws):
		h, rgb = self.init(ws[0])
		for s, w in zip(self.skips, ws[1:]):
			h, rgb = s(h, rgb, w)
		return rgb

class Generator(Chain):

	def __init__(self, latent_size=512, levels=7, first_channels=512, last_channels=16):
		super().__init__()
		self.latent_size = latent_size
		self.levels = levels
		self.resolution = (2 * 2 ** levels, 2 * 2 ** levels)
		with self.init_scope():
			self.sampler = GaussianDistribution()
			self.mapper = Mapper(latent_size, levels)
			self.synthesizer = Synthesizer(latent_size, levels, first_channels, last_channels)

	def __call__(self, z):
		return self.synthesizer([self.mapper(z)] * levels)

	def generate_latents(self, batch):
		return self.sampler((batch, self.latent_size))

class Discriminator(Chain):

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
