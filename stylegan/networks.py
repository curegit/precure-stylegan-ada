from math import sqrt as root
from random import randint
from h5py import File as HDF5File
from chainer import Variable, Chain, ChainList, Sequential
from chainer.functions import sqrt, mean, concat
from chainer.serializers import HDF5Serializer, HDF5Deserializer
from stylegan.layers.basic import LeakyRelu, EqualizedLinear
from stylegan.layers.generator import InitialSkipArchitecture, SkipArchitecture
from stylegan.layers.discriminator import FromRGB, ResidualBlock, OutputBlock
from utilities.math import identity, lerp

class Mapper(Chain):

	def __init__(self, size, depth, conditional=False):
		super().__init__()
		with self.init_scope():
			self.mlp = Sequential(
				EqualizedLinear(size * 2 if conditional else size, size), LeakyRelu(),
				*([EqualizedLinear(size, size), LeakyRelu()] * (depth - 1)))

	def __call__(self, z, c=None):
		h1 = z / sqrt(mean(z ** 2, axis=1, keepdims=True) + 1e-08)
		h2 = h1 if c is None else concat((h1, c / sqrt(mean(c ** 2, axis=1, keepdims=True) + 1e-08)), axis=1)
		return self.mlp(h2)

class Synthesizer(Chain):

	def __init__(self, size, levels, first_channels, last_channels):
		super().__init__()
		in_channels = [first_channels] * levels
		out_channels = [last_channels] * levels
		for i in range(1, levels):
			channels = min(first_channels, last_channels * 2 ** i)
			in_channels[-i] = channels
			out_channels[-i - 1] = channels
		with self.init_scope():
			self.init = InitialSkipArchitecture(size, in_channels[0], out_channels[0])
			self.skips = ChainList(*[SkipArchitecture(size, i, o) for i, o in zip(in_channels[1:], out_channels[1:])])

	def __call__(self, ws):
		h, rgb = self.init(ws[0])
		for s, w in zip(self.skips, ws[1:]):
			h, rgb = s(h, rgb, w)
		return rgb

class Generator(Chain):

	def __init__(self, size=512, depth=8, levels=7, first_channels=512, last_channels=64, categories=1):
		super().__init__()
		self.size = size
		self.depth = depth
		self.levels = levels
		self.first_channels = first_channels
		self.last_channels = last_channels
		self.categories = categories
		self.resolution = (2 * 2 ** levels, 2 * 2 ** levels)
		self.labels = [f"Category {i}" for i in range(categories)]
		with self.init_scope():
			self.mapper = Mapper(size, depth, categories > 1)
			self.synthesizer = Synthesizer(size, levels, first_channels, last_channels)
			if categories > 1:
				self.embedder = EqualizedLinear(categories, size, gain=1)

	def __call__(self, z, c=None, random_mix=None, psi=1.0, mean_w=None):
		z, *zs = z if z is tuple or z is list else [z]
		if c is not None:
			c = self.embedder(c)
		truncation_trick = identity
		if psi != 1.0:
			if mean_w is None:
				mean_w = self.calculate_mean_w()
			truncation_trick = lambda w: lerp(mean_w, w, psi)
		w = truncation_trick(self.mapper(z, c))
		ws = [w] * self.levels
		stop = self.levels
		if self.levels > 1 and random_mix is not None:
			mix_level = randint(1, self.levels - 1)
			mix_w = truncation_trick(self.mapper(random_mix, c))
			ws[mix_level:stop] = [mix_w] * (stop - mix_level)
			stop = mix_level
		for i, z in zip(range(1, stop), zs):
			if z is not Ellipsis:
				ws[i:stop] = [truncation_trick(self.mapper(z, c))] * (stop - i)
		return ws, self.synthesizer(ws)

	def generate_latents(self, batch):
		return Variable(self.xp.random.normal(size=(batch, self.size)).astype(self.xp.float32))

	def generate_conditions(self, batch, category=None):
		if category is None:
			return Variable(self.xp.eye(self.categories, dtype=self.xp.float32)[self.xp.random.randint(low=0, high=self.categories, size=batch)])
		else:
			return Variable(self.xp.eye(self.categories, dtype=self.xp.float32)[[category] * batch])

	def generate_masks(self, batch):
		return Variable(self.xp.random.normal(size=(batch, 3, *self.resolution)).astype(self.xp.float32)) / root(self.resolution[0] * self.resolution[1])

	def calculate_mean_w(self, n=50000):
		return mean(self.mapper(self.generate_latents(n)), axis=0)

	def embed_labels(self, labels):
		for i, l in enumerate(labels):
			self.labels[i] = str(l)

	def save(self, filepath):
		with HDF5File(filepath, "w") as hdf5:
			hdf5.attrs["size"] = self.size
			hdf5.attrs["depth"] = self.depth
			hdf5.attrs["levels"] = self.levels
			hdf5.attrs["first_channels"] = self.first_channels
			hdf5.attrs["last_channels"] = self.last_channels
			hdf5.attrs["categories"] = self.categories
			hdf5.attrs["labels"] = self.labels
			HDF5Serializer(hdf5).save(self)

	@staticmethod
	def load(filepath):
		with HDF5File(filepath, "r") as hdf5:
			size = int(hdf5.attrs["size"])
			depth = int(hdf5.attrs["depth"])
			levels = int(hdf5.attrs["levels"])
			first_channels = int(hdf5.attrs["first_channels"])
			last_channels = int(hdf5.attrs["last_channels"])
			categories = int(hdf5.attrs["categories"])
			generator = Generator(size, depth, levels, first_channels, last_channels, categories)
			generator.embed_labels(hdf5.attrs["labels"])
			HDF5Deserializer(hdf5).load(generator)
			return generator

class Discriminator(Chain):

	def __init__(self, levels=7, first_channels=16, last_channels=512, categories=1, depth=8, group_size=None):
		super().__init__()
		in_channels = [first_channels] * (levels - 1)
		out_channels = [last_channels] * (levels - 1)
		for i in range(1, levels - 1):
			channels = min(first_channels * 2 ** i, last_channels)
			in_channels[i] = channels
			out_channels[i - 1] = channels
		with self.init_scope():
			self.main = Sequential(
				FromRGB(first_channels),
				*[ResidualBlock(i, o) for i, o in zip(in_channels, out_channels)],
				OutputBlock(last_channels, categories > 1, group_size))
			if categories > 1:
				self.embedder = EqualizedLinear(categories, last_channels, gain=1)
				self.mapper = Sequential(EqualizedLinear(last_channels, last_channels), LeakyRelu()).repeat(depth)

	def __call__(self, x, c=None):
		if c is not None:
			embedded = self.embedder(c)
			normalized = embedded / sqrt(mean(embedded ** 2, axis=1, keepdims=True) + 1e-08)
			c1 = self.mapper(normalized)
		h = self.main(x)
		batch, channels = h.shape
		return h.reshape(batch) if c is None else (h * c1).sum(axis=1) / root(channels)
