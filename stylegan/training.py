import random
import numpy as np
from os.path import basename
from h5py import File as HDF5File
from chainer import grad, Variable
from chainer.reporter import report
from chainer.training import StandardUpdater, Trainer
from chainer.training.extensions import PrintReport, LogReport, PlotReport, ProgressBar
from chainer.functions import sqrt, sum, mean, batch_l2_norm_squared, softplus, stack
from chainer.serializers import HDF5Serializer, HDF5Deserializer
from utilities.iter import range_batch
from utilities.image import save_image
from utilities.filesys import build_filepath

class OptimizerTriple():

	def __init__(self, mapper_optimizer, synthesizer_optimizer, discriminator_optimizer):
		self.mapper_optimizer = mapper_optimizer
		self.synthesizer_optimizer = synthesizer_optimizer
		self.discriminator_optimizer = discriminator_optimizer
		self.path_length = 0.0

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
			self.path_length = float(hdf5["path_length"].value)
			HDF5Deserializer(hdf5["mapper"]).load(self.mapper_optimizer)
			HDF5Deserializer(hdf5["synthesizer"]).load(self.synthesizer_optimizer)
			HDF5Deserializer(hdf5["discriminator"]).load(self.discriminator_optimizer)

	def save_states(self, filepath):
		with HDF5File(filepath, "w") as hdf5:
			hdf5.create_dataset("path_length", data=self.path_length)
			HDF5Serializer(hdf5.create_group("mapper")).save(self.mapper_optimizer)
			HDF5Serializer(hdf5.create_group("synthesizer")).save(self.synthesizer_optimizer)
			HDF5Serializer(hdf5.create_group("discriminator")).save(self.discriminator_optimizer)

class CustomUpdater(StandardUpdater):

	def __init__(self, generator, discriminator, iterator, optimizers, mixing_rate=0.5, gamma=10, decay=0.01, pl_weight=2, lsgan=False):
		super().__init__(iterator, dict(optimizers))
		self.generator = generator
		self.discriminator = discriminator
		self.iterator = iterator
		self.optimizers = optimizers
		self.mixing_rate = mixing_rate
		self.gamma = gamma
		self.decay = decay
		self.lsgan = lsgan

	def next_latents(self):
		return self.generator.generate_latents(self.iterator.batch_size)

	def next_real_images(self):
		return Variable(self.discriminator.xp.array(self.iterator.next()))

	def update_core(self):
		self.update_discriminator()
		self.update_generator()

	def update_generator(self):
		self.generator.cleargrads()
		z = self.next_latents()
		mix = self.next_latents() if random.random() < self.mixing_rate else None
		ws, x_fake = self.generator(z, random_mix=mix)
		y_fake = self.discriminator(x_fake)

		lerp = lambda a, b, t: a + (b - a) * t
		p = CustomUpdater.path_length_f(ws, x_fake, self.generator.generate_masks(ws[0].shape[0]))
		self.optimizers.path_length = lerp(self.optimizers.path_length, p.item(), self.decay)
		penalty = (p - self.optimizers.path_length) ** 2

		loss_func = CustomUpdater.generator_ls_loss if self.lsgan else CustomUpdater.generator_adversarial_loss
		loss = loss_func(y_fake) + pl_weight * penalty
		loss.backward()
		self.optimizers.update_generator()
		report({"loss (G)": loss})

	def update_discriminator(self):
		self.discriminator.cleargrads()
		x_real = self.next_real_images()
		y_real = self.discriminator(x_real)
		z = self.next_latents()
		mix = self.next_latents() if random.random() < self.mixing_rate else None
		ws, x_fake = self.generator(z, random_mix=mix)
		y_fake = self.discriminator(x_fake)
		x_fake.unchain_backward()
		penalty = CustomUpdater.gradient_penalty(x_real, y_real, gamma=self.gamma)
		loss_func = CustomUpdater.discriminator_ls_loss if self.lsgan else CustomUpdater.discriminator_adversarial_loss
		loss = loss_func(y_real, y_fake) + penalty
		loss.backward()
		self.optimizers.update_discriminator()
		report({"loss (D)": loss, "penalty (D)": penalty})

	@staticmethod
	def generator_adversarial_loss(fake):
		return sum(softplus(-fake)) / fake.shape[0]

	@staticmethod
	def generator_ls_loss(fake):
		return sum((fake - 1) ** 2) / 2 / fake.shape[0]

	@staticmethod
	def discriminator_adversarial_loss(real, fake):
		return (sum(softplus(-real)) + sum(softplus(fake))) / real.shape[0]

	@staticmethod
	def discriminator_ls_loss(real, fake):
		return (sum((real - 1) ** 2) + sum(fake ** 2)) / 2 / real.shape[0]

	@staticmethod
	def gradient_penalty(x, y, gamma=10):
		gradient = grad([y], [x], enable_double_backprop=True)[0]
		squared_norm = sum(batch_l2_norm_squared(gradient)) / x.shape[0]
		return gamma * squared_norm / 2

	@staticmethod
	def path_length_f(ws, x, mask):
		levels = len(ws)
		batch, size = ws[0].shape
		gradients = grad([x * mask], ws, enable_double_backprop=True)
		gradient = stack(gradients).transpose(1, 0, 2).reshape(batch * levels, size)
		path_lengths = batch_l2_norm_squared(gradient).reshape(batch, levels)
		return mean(sqrt(mean(path_lengths, axis=1)))

class CustomTrainer(Trainer):

	def __init__(self, updater, epoch, dest, overwrite=False):
		super().__init__(updater, (epoch, "epoch"), dest)
		self.overwrite = overwrite
		self.number = 16

	def enable_reports(self, interval):
		entries = ["epoch", "iteration", "loss (G)", "loss (D)", "penalty (D)"]
		filename = basename(build_filepath(self.out, "report", "log", self.overwrite))
		log_report = LogReport(trigger=(interval, "iteration"), filename=filename)
		print_report = PrintReport(entries, log_report)
		entries = ["loss (G)", "loss (D)", "penalty (D)"]
		filename = basename(build_filepath(self.out, "plot", "png", self.overwrite))
		plot_report = PlotReport(entries, "iteration", trigger=(interval, "iteration"), filename=filename)
		self.extend(log_report)
		self.extend(print_report)
		self.extend(plot_report)

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
		trainer.updater.generator.save_weights(filepath)
		filepath = build_filepath(trainer.out, f"dis_{iteration}", "hdf5", trainer.overwrite)
		trainer.updater.discriminator.save_weights(filepath)

	@staticmethod
	def save_optimizer_states(trainer):
		iteration = trainer.updater.iteration
		filepath = build_filepath(trainer.out, f"opt_{iteration}", "hdf5", trainer.overwrite)
		trainer.updater.optimizers.save_states(filepath)

	@staticmethod
	def save_middle_images(trainer):
		for i, n in range_batch(trainer.number, trainer.updater.iterator.batch_size):
			z = trainer.updater.generator.generate_latents(n)
			ws, y = trainer.updater.generator(z)
			z.to_cpu()
			y.to_cpu()
			for j in range(n):
				filename = f"{trainer.updater.iteration}_{i + j + 1}"
				np.save(build_filepath(trainer.out, filename, "npy", trainer.overwrite), z.array[j])
				save_image(y.array[j], build_filepath(trainer.out, filename, "png", trainer.overwrite))
