import random
import numpy as np
from os.path import basename
from h5py import File as HDF5File
from chainer import grad, Variable
from chainer.reporter import report
from chainer.training import StandardUpdater, Trainer
from chainer.training.extensions import PrintReport, LogReport, PlotReport, ProgressBar
from chainer.functions import sqrt, sign, softplus, mean, batch_l2_norm_squared, stack
from chainer.optimizers import Adam
from chainer.serializers import HDF5Serializer, HDF5Deserializer
from utilities.iter import range_batch
from utilities.math import lerp
from utilities.image import save_image
from utilities.filesys import build_filepath

class AdamTriple():

	def __init__(self, alphas, beta1, beta2):
		self.mapper_optimizer = Adam(alphas[0], beta1, beta2, eps=1e-08)
		self.synthesizer_optimizer = Adam(alphas[1], beta1, beta2, eps=1e-08)
		self.discriminator_optimizer = Adam(alphas[2], beta1, beta2, eps=1e-08)

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

class CustomUpdater(StandardUpdater):

	def __init__(self, generator, averaged_generator, discriminator, iterator, optimizers, averaging_images=10000, lsgan=False):
		super().__init__(iterator, dict(optimizers))
		self.generator = generator
		self.averaged_generator = averaged_generator
		self.discriminator = discriminator
		self.iterator = iterator
		self.optimizers = optimizers
		self.averaging_images = averaging_images
		self.lsgan = lsgan
		self.style_mixing_rate = 0.0
		self.r1_regularization_interval = 0
		self.averaged_path_length = 0.0
		self.path_length_regularization_interval = 0
		self.augumentation_probability = 0.0

	def enable_style_mixing(self, mixing_rate=0.5):
		self.style_mixing_rate = mixing_rate

	def enable_r1_regularization(self, gamma=10, interval=16):
		self.r1_regularization_gamma = gamma
		self.r1_regularization_interval = interval

	def enable_path_length_regularization(self, decay=0.99, weight=2, interval=8):
		self.path_length_decay = decay
		self.path_length_penalty_weight = weight
		self.path_length_regularization_interval = interval

	def next_latents(self):
		return self.generator.generate_latents(self.iterator.batch_size)

	def next_real_images(self):
		return Variable(self.discriminator.xp.array(self.iterator.next()))

	def update_core(self):
		self.update_generator()
		self.average_generator()
		self.update_discriminator()

	def update_generator(self):
		self.generator.cleargrads()
		z = self.next_latents()
		mix = self.next_latents() if random.random() < self.style_mixing_rate else None
		ws, x_fake = self.generator(z, random_mix=mix)
		y_fake = self.discriminator(x_fake)
		if self.path_length_regularization_interval and self.iteration % self.path_length_regularization_interval == 0:
			masks = self.generator.generate_masks(self.iterator.batch_size)
			path_length = CustomUpdater.path_length(ws, x_fake, masks)
			self.averaged_path_length = lerp(path_length.item(), self.averaged_path_length, self.path_length_decay)
			penalty = self.path_length_penalty_weight * (path_length - self.averaged_path_length) ** 2
		else:
			penalty = 0.0
		if self.lsgan:
			loss = CustomUpdater.generator_least_squares_loss(y_fake)
		else:
			loss = CustomUpdater.generator_logistic_loss(y_fake)
		(loss + penalty).backward()
		self.optimizers.update_generator()
		report({"loss (G)": loss, "penalty (G)": penalty, "path length": self.averaged_path_length})

	def average_generator(self):
		decay = 0.5 ** (self.iterator.batch_size / self.averaging_images)
		for raw, averaged in zip(self.generator.params(), self.averaged_generator.params()):
			averaged.copydata(lerp(raw, averaged, decay))

	def update_discriminator(self):
		self.discriminator.cleargrads()
		x_real = self.next_real_images()
		y_real = self.discriminator(x_real)
		rt = mean(sign(y_real - 0.5 if self.lsgan else y_real))
		z = self.next_latents()
		mix = self.next_latents() if random.random() < self.style_mixing_rate else None
		_, x_fake = self.generator(z, random_mix=mix)
		y_fake = self.discriminator(x_fake)
		x_fake.unchain_backward()
		if self.r1_regularization_interval and self.iteration % self.r1_regularization_interval == 0:
			penalty = CustomUpdater.gradient_penalty(x_real, y_real, self.r1_regularization_gamma)
		else:
			penalty = 0.0
		if self.lsgan:
			loss = CustomUpdater.discriminator_least_squares_loss(y_real, y_fake)
		else:
			loss = CustomUpdater.discriminator_logistic_loss(y_real, y_fake)
		(loss + penalty).backward()
		self.optimizers.update_discriminator()
		report({"loss (D)": loss, "penalty (D)": penalty, "overfitting": rt, "augumentation": self.augumentation_probability})

	def load_states(self, filepath):
		with HDF5File(filepath, "r") as hdf5:
			self.averaged_path_length = float(hdf5["averaged_path_length"][()])
			self.augumentation_probability = float(hdf5["augumentation_probability"][()])
			HDF5Deserializer(hdf5["generator"]).load(self.generator)
			HDF5Deserializer(hdf5["averaged_generator"]).load(self.averaged_generator)
			HDF5Deserializer(hdf5["discriminator"]).load(self.discriminator)
			for key, optimizer in dict(self.optimizers).items():
				HDF5Deserializer(hdf5["optimizers"][key]).load(optimizer)

	def save_states(self, filepath):
		with HDF5File(filepath, "w") as hdf5:
			hdf5.create_dataset("averaged_path_length", data=self.averaged_path_length)
			hdf5.create_dataset("augumentation_probability", data=self.augumentation_probability)
			HDF5Serializer(hdf5.create_group("generator")).save(self.generator)
			HDF5Serializer(hdf5.create_group("averaged_generator")).save(self.averaged_generator)
			HDF5Serializer(hdf5.create_group("discriminator")).save(self.discriminator)
			optimizer_group = hdf5.create_group("optimizers")
			for key, optimizer in dict(self.optimizers).items():
				HDF5Serializer(optimizer_group.create_group(key)).save(optimizer)

	@staticmethod
	def generator_logistic_loss(fake):
		return mean(softplus(-fake))

	@staticmethod
	def generator_least_squares_loss(fake):
		return mean((fake - 1) ** 2)

	@staticmethod
	def discriminator_logistic_loss(real, fake):
		return mean(softplus(-real)) + mean(softplus(fake))

	@staticmethod
	def discriminator_least_squares_loss(real, fake):
		return mean((real - 1) ** 2) + mean(fake ** 2)

	@staticmethod
	def gradient_penalty(x, y, gamma):
		gradient = grad([y], [x], enable_double_backprop=True)[0]
		squared_norm = mean(batch_l2_norm_squared(gradient))
		return gamma * squared_norm / 2

	@staticmethod
	def path_length(ws, x, mask):
		levels, batch, size = len(ws), *(ws[0].shape)
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
		filename = basename(build_filepath(self.out, "report", "log", self.overwrite))
		log_report = LogReport(trigger=(interval, "iteration"), filename=filename)
		entries = ["iteration", "loss (G)", "loss (D)", "penalty (G)", "penalty (D)", "overfitting"]
		print_report = PrintReport(entries, log_report)
		entries = ["loss (G)", "loss (D)", "penalty (G)", "penalty (D)", "overfitting"]
		filename = basename(build_filepath(self.out, "plot", "png", self.overwrite))
		plot_report = PlotReport(entries, "iteration", trigger=(interval, "iteration"), filename=filename)
		self.extend(log_report)
		self.extend(print_report)
		self.extend(plot_report)

	def enable_progress_bar(self, interval=1):
		self.extend(ProgressBar(update_interval=interval))

	def hook_state_save(self, interval):
		self.extend(CustomTrainer.save_states, trigger=(interval, "iteration"))
		self.extend(CustomTrainer.save_generator, trigger=(interval, "iteration"))

	def hook_image_generation(self, interval, number=None):
		self.number = self.number if number is None else number
		self.extend(CustomTrainer.save_images, trigger=(interval, "iteration"))

	@property
	def iteration(self):
		return self.updater.iteration

	@property
	def batch_size(self):
		return self.updater.iterator.batch_size

	@staticmethod
	def save_states(trainer):
		filepath = build_filepath(trainer.out, f"snapshot-{trainer.iteration}", "hdf5", trainer.overwrite)
		trainer.updater.save_states(filepath)

	@staticmethod
	def save_generator(trainer):
		filepath = build_filepath(trainer.out, f"generator-{trainer.iteration}", "hdf5", trainer.overwrite)
		trainer.updater.averaged_generator.save_weights(filepath)

	@staticmethod
	def save_images(trainer):
		for i, n in range_batch(trainer.number, trainer.batch_size):
			z = trainer.updater.averaged_generator.generate_latents(n)
			_, y = trainer.updater.averaged_generator(z)
			z.to_cpu()
			y.to_cpu()
			for j in range(n):
				filename = f"{trainer.iteration}-{i + j + 1}"
				np.save(build_filepath(trainer.out, filename, "npy", trainer.overwrite), z.array[j])
				save_image(y.array[j], build_filepath(trainer.out, filename, "png", trainer.overwrite))
