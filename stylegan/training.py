import random
import numpy as np
from os.path import basename
from h5py import File as HDF5File
from chainer import grad, Variable
from chainer.reporter import report
from chainer.training import StandardUpdater, Trainer
from chainer.training.extensions import PrintReport, LogReport, PlotReport, ProgressBar
from chainer.functions import sum, batch_l2_norm_squared, softplus
from chainer.serializers import HDF5Serializer, HDF5Deserializer
from utilities.iter import range_batch
from utilities.image import save_image
from utilities.filesys import build_filepath

class OptimizerTriple():

	def __init__(self, mapper_optimizer, synthesizer_optimizer, discriminator_optimizer):
		self.mapper_optimizer = mapper_optimizer
		self.synthesizer_optimizer = synthesizer_optimizer
		self.discriminator_optimizer = discriminator_optimizer

	def __iter__(self):
		yield "mapper", self.mapper_optimizer
		yield "synthesizer", self.synthesizer_optimizer
		yield "discriminator", self.discriminator_optimizer

	def setup(self, generator, discriminator):
		self.mapper_optimizer.setup(generator.mapper)
		self.synthesizer_optimizer.setup(generator.synthesizer)
		self.discriminator_optimizer.setup(discriminator)

	def update_generator(self):
		self.mapper_optimizer.update()
		self.synthesizer_optimizer.update()

	def update_discriminator(self):
		self.discriminator_optimizer.update()

	def load_states(self, filepath):
		with HDF5File(filepath, "r") as hdf5:
			HDF5Deserializer(hdf5["mapper"]).load(self.mapper_optimizer)
			HDF5Deserializer(hdf5["synthesizer"]).load(self.synthesizer_optimizer)
			HDF5Deserializer(hdf5["discriminator"]).load(self.discriminator_optimizer)

	def save_states(self, filepath):
		with HDF5File(filepath, "w") as hdf5:
			HDF5Serializer(hdf5.create_group("mapper")).save(self.mapper_optimizer)
			HDF5Serializer(hdf5.create_group("synthesizer")).save(self.synthesizer_optimizer)
			HDF5Serializer(hdf5.create_group("discriminator")).save(self.discriminator_optimizer)

class CustomUpdater(StandardUpdater):

	def __init__(self, generator, discriminator, iterator, optimizers, mixing_rate=0.5, gamma=10, lsgan=False):
		super().__init__(iterator, dict(optimizers))
		self.generator = generator
		self.discriminator = discriminator
		self.optimizers = optimizers
		self.mixing_rate = mixing_rate
		self.gamma = gamma
		self.lsgan = lsgan

	def update_core(self):
		self.update_discriminator()
		self.update_generator()

	def update_generator(self):
		self.generator.cleargrads()
		z = self.generator.generate_latents(self.batch_size)
		mix = self.generator.generate_latents(self.batch_size) if random.random() < self.mixing_rate else None
		x_fake = self.generator(z, random_mix=mix)
		y_fake = self.discriminator(x_fake)
		loss = (sum((y_fake - 1) ** 2) / 2 if self.lsgan else sum(softplus(-y_fake))) / self.iterator.batch_size
		loss.backward()
		self.optimizers.update_generator()
		report({"loss (G)": loss})

	def update_discriminator(self):
		self.discriminator.cleargrads()
		x_real = Variable(np.array(self.iterator.next()))
		#self.get_iterator("main").
		#x_real.to_device(self.discriminator.device)
		y_real = self.discriminator(x_real)
		gradient = grad([y_real], [x_real], enable_double_backprop=True)[0]
		gradient_norm = sum(batch_l2_norm_squared(gradient)) / self.iterator.batch_size
		penalty = self.gamma * gradient_norm / 2
		z = self.generator.generate_latents(self.iterator.batch_size)
		mix = self.generator.generate_latents(self.iterator.batch_size) if random.random() < self.mixing_rate else None
		x_fake = self.generator(z, random_mix=mix)
		y_fake = self.discriminator(x_fake)
		x_fake.unchain_backward()
		loss = ((sum((y_real - 1) ** 2) + sum(y_fake ** 2)) / 2 if self.lsgan else (sum(softplus(-y_real)) + sum(softplus(y_fake)))) / self.iterator.batch_size
		loss += penalty
		loss.backward()
		self.optimizers.update_discriminator()
		report({"loss (D)": loss, "penalty (D)": penalty})

	@property
	def iterator(self):
		return self.get_iterator("main")

	@property
	def batch_size(self):
		return self.iterator.batch_size
'''
	@staticmethod
	def convert(self):
		self.discriminator.

	@staticmethod
	def adversarial_loss(x, y):

	@staticmethod
	def least_square_loss(x, y, a, b, c):
'''

class CustomTrainer(Trainer):

	def __init__(self, updater, epoch, dest, overwrite=False):
		super().__init__(updater, (epoch, "epoch"), dest)
		self.overwrite = overwrite
		self.number = 16

	def enable_reports(self, interval):
		#entries = ["epoch", "iteration", "loss (G)", "loss (D)", "penalty (D)"]
		filename = basename(build_filepath(self.out, "report", "log", self.overwrite))
		log = LogReport(trigger=(interval, "iteration"), filename=filename)
		print = PrintReport(["epoch", "iteration", "loss (G)", "loss (D)", "penalty (D)"], log)
		filename = basename(build_filepath(self.out, "plot", "png", self.overwrite))
		plot = PlotReport(["loss (G)", "loss (D)", "penalty (D)"], "iteration", trigger=(interval, "iteration"), filename=filename)
		self.extend(log)
		self.extend(print)
		self.extend(plot)

	def enable_progress_bar(self, interval=1):
		self.extend(ProgressBar(update_interval=interval))

	def hook_state_save(self, interval):
		self.extend(CustomTrainer.save_model_states, trigger=(interval, "iteration"))
		self.extend(CustomTrainer.save_optimizer_states, trigger=(interval, "iteration"))

	def hook_image_generation(self, interval, number=None):
		self.number = self.number if number is None else number
		self.extend(CustomTrainer.save_middle_images, trigger=(interval, "iteration"))

	@staticmethod
	def save_model_states(trainer):
		iteration = trainer.updater.iteration
		filepath = build_filepath(trainer.out, f"gen_{iteration}", "hdf5", trainer.overwrite)
		trainer.updater.generator.save_state(filepath)
		filepath = build_filepath(trainer.out, f"dis_{iteration}", "hdf5", trainer.overwrite)
		trainer.updater.discriminator.save_state(filepath)

	@staticmethod
	def save_optimizer_states(trainer):
		iteration = trainer.updater.iteration
		filepath = build_filepath(trainer.out, f"opt_{iteration}", "hdf5", trainer.overwrite)
		trainer.updater.optimizers.save_states(filepath)

	@staticmethod
	def save_middle_images(trainer):
		for i, n in range_batch(trainer.number, trainer.updater.iterator.batch_size):
			z = trainer.updater.generator.generate_latents(n)
			y = trainer.updater.generator(z)
			z.to_cpu()
			y.to_cpu()
			for j in range(n):
				filename = f"{trainer.updater.iteration}_{i + j + 1}"
				np.save(build_filepath(trainer.out, filename, "npy", trainer.overwrite), z.array[j])
				save_image(y.array[j], build_filepath(trainer.out, filename, "png", trainer.overwrite))
