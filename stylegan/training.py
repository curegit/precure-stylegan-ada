from chainer.training import Trainer, extensions, make_extension
from chainer.iterators import MultiprocessIterator
from random import random
from chainer import grad
from chainer.reporter import report
from chainer.training import Updater, Trainer
from chainer.functions import sum, batch_l2_norm_squared, softplus

from chainer.iterators import SerialIterator

from utilities.stdio import eprint

from h5py import File as HDF5File

from chainer.serializers import HDF5Serializer, HDF5Deserializer

class OptimizerSet():

	def __init__(self, mapper_optimizer, synthesizer_optimizer, discriminator_optimizer):
		self.mapper_optimizer = mapper_optimizer
		self.synthesizer_optimizer = synthesizer_optimizer
		self.discriminator_optimizer = discriminator_optimizer

	def setup(self, generator, discriminator):
		self.mapper_optimizer.setup(generator.mapper)
		self.synthesizer_optimizer.setup(generator.synthesizer)
		self.discriminator_optimizer.setup(discriminator)

	def update_generator(self):
		self.mapper_optimizer.update()
		self.synthesizer_optimizer.update()

	def update_discriminator(self):
		self.discriminator_optimizer.update()

	def load_state(self, filepath):
		with HDF5File(filepath, "r") as hdf5:
			HDF5Deserializer(hdf5["mapper"]).load(self.mapper_optimizer)
			HDF5Deserializer(hdf5["synthesizer"]).load(self.synthesizer_optimizer)
			HDF5Deserializer(hdf5["discriminator"]).load(self.discriminator_optimizer)

	def save_state(self, filepath):
		with HDF5File(filepath, "w") as hdf5:
			HDF5Serializer(hdf5.create_group("mapper")).save(self.mapper_optimizer)
			HDF5Serializer(hdf5.create_group("synthesizer")).save(self.synthesizer_optimizer)
			HDF5Serializer(hdf5.create_group("discriminator")).save(self.discriminator_optimizer)

class CustomUpdater(Updater):

	def __init__(self, generator, discriminator, iterator, optimizers, mixing=0.5, gamma=10, lsgan=False):
		super().__init__()
		self.generator = generator
		self.discriminator = discriminator
		self.iterator = iterator
		self.optimizers = optimizers
		self.mixing = mixing
		self.gamma = gamma
		self.lsgan = lsgan

	def update_core(self):
		batch = self.iterator.next()
		batchsize = len(batch)

		# Train discriminator
		x_real = self.discriminator.wrap_array(batch)
		y_real = self.discriminator(x_real)
		gradient = grad([y_real], [x_real], enable_double_backprop=True)[0]
		gradient_norm = sum(batch_l2_norm_squared(gradient)) / batchsize
		loss_grad = self.gamma * gradient_norm / 2
		z = self.generator.generate_latent(batchsize)
		#mix_z = self.generator.generate_latent(batchsize) if self.mixing > random() else None
		x_fake = self.generator(z) # TODO
		y_fake = self.discriminator(x_fake)
		loss_dis = ((sum((y_real - 1) ** 2) + sum(y_fake ** 2)) / 2 if self.lsgan else (sum(softplus(-y_real)) + sum(softplus(y_fake)))) / batchsize
		loss_dis += loss_grad
		x_fake.unchain_backward()
		self.discriminator.cleargrads()
		loss_dis.backward()
		self.optimizers.update_discriminator()
		#self.discriminator_optimizer.update()

		# Train generator
		z = self.generator.generate_latent(batchsize)
		#mix_z = self.generator.generate_latent(batchsize) if self.mixing > random() else None
		x_fake = self.generator(z) # TODO
		y_fake = self.discriminator(x_fake)
		loss_gen = (sum((y_fake - 1) ** 2) / 2 if self.lsgan else sum(softplus(-y_fake))) / batchsize
		self.generator.cleargrads()
		loss_gen.backward()
		self.optimizers.update_generator()
		#self.mapper_optimizer.update()
		#self.generator_optimizer.update()

		#report({"alpha": self.alpha})
		report({"loss (gen)": loss_gen})
		report({"loss (dis)": loss_dis})
		report({"loss (grad)": loss_grad})
		#self.alpha = min(1.0, self.alpha + self.delta)

"""
# Define extension to output images in progress
def save_middle_images(generator, stage, directory, number, batch, mix, force=True, save_latent=True):
	@make_extension()
	def func(trainer):
		c = 0
		mixing = mix > random()
		while c < number:
			n = min(number - c, batch)
			z = generator.generate_latent(n)
			mix_z = generator.generate_latent(n) if mixing else None
			y = generator(z, stage, trainer.updater.alpha, mix_z)
			z.to_cpu()
			y.to_cpu()
			for i in range(n):
				path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{trainer.updater.alpha:.3f}_{c + i + 1}", "png")
				path = path if force else altfilepath(path)
				save_image(y.array[i], path)
				if save_latent:
					path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{trainer.updater.alpha:.3f}_{c + i + 1}", "npy")
					path = path if force else altfilepath(path)
					save_array(z.array[i], path)
			c += n
	return func

# Define extension to save models in progress
def save_middle_models(generator, discriminator, stage, directory, device, force=True):
	@make_extension()
	def func(trainer):
		generator.to_cpu()
		discriminator.to_cpu()
		path = filepath(directory, f"gen_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, generator)
		path = filepath(directory, f"dis_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, discriminator)
		if device >= 0:
			generator.to_gpu(device)
			discriminator.to_gpu(device)
	return func

# Define extension to save optimizers in progress
def save_middle_optimizers(mapper_optimizer, generator_optimizer, discriminator_optimizer, stage, directory, force=True):
	@make_extension()
	def func(trainer):
		path = filepath(directory, f"mopt_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, mapper_optimizer)
		path = filepath(directory, f"gopt_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, generator_optimizer)
		path = filepath(directory, f"dopt_{stage}_{trainer.updater.iteration}", "hdf5")
		path = path if force else altfilepath(path)
		serializers.save_hdf5(path, discriminator_optimizer)
	return func
"""

class CustomTrainer(Trainer):

	def __init__(self, updater, epoch, dest):
		super().__init__(updater, (epoch, "epoch"), dest)
		self.dest = dest
		#i = updater.iterator.batch_size
		self.generator = updater.generator
		self.discriminator = updater.discriminator
		self.number = 100

"""
	@make_extension()
	def save_middle_images(trainer):
		trainer.generator
		number = self.number or
		for i in range(0, number, batch):
			n = min(number - i, batch)


		c = 0
		mixing = mix > random()
		while c < number:
			n = min(number - c, batch)
			z = generator.generate_latent(n)
			mix_z = generator.generate_latent(n) if mixing else None
			y = generator(z, stage, trainer.updater.alpha, mix_z)
			z.to_cpu()
			y.to_cpu()
			for i in range(n):
				path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{trainer.updater.alpha:.3f}_{c + i + 1}", "png")
				path = path if force else altfilepath(path)
				save_image(y.array[i], path)
				if save_latent:
					path = filepath(directory, f"{stage}_{trainer.updater.iteration}_{trainer.updater.alpha:.3f}_{c + i + 1}", "npy")
					path = path if force else altfilepath(path)
					save_array(z.array[i], path)
			c += n
"""
