from chainer import Chain, Sequential

from stylegan.links.discriminator import FromRGB, ResidualBlock, OutputBlock

class Generator(Chain):

	def __init__():
		pass

class Discriminator(Chain):

	def __init__(self, max_stage=7, first_channels=16, last_channels=512):
		super().__init__()
		in_channels = [first_channels if i == max_stage else min(first_channels * 2 ** (max_stage - i), last_channels) for i in range(max_stage, 1, -1)]
		out_channels = [last_channels if i == 2 else min(first_channels * 2 ** (max_stage - i + 1), last_channels) for i in range(max_stage, 1, -1)]
		blocks = [ResidualBlock(i, o) for i, o in zip(in_channels, out_channels)]
		with self.init_scope():
			self.frgb = FromRGB(first_channels)
			self.blocks = Sequential(*blocks)
			self.output = OutputBlock(last_channels)

	def __call__(self, x):
		return self.output(self.blocks(self.frgb(x)))
