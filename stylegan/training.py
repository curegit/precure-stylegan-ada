import random
import os.path
import numpy as np
from h5py import File as HDF5File
from chainer import grad, Variable
from chainer.reporter import report
from chainer.training import StandardUpdater, Trainer
from chainer.training.extensions import PrintReport, LogReport, PlotReport, ProgressBar
from chainer.functions import sqrt, sign, softplus, sum, mean, batch_l2_norm_squared, stack
from chainer.optimizers import Adam
from chainer.serializers import HDF5Serializer, HDF5Deserializer
from stylegan.networks import Generator, Discriminator
from utilities.iter import range_batch, iter_batch
from utilities.math import identity, sgn, clamp, lerp
from utilities.image import save_image
from utilities.filesys import mkdirs, build_filepath

class AdamSet():

	def __init__(self, alpha, beta1, beta2, conditional=False):
		self.conditional = conditional
		self.mapper_optimizer = Adam(alpha / 100, beta1, beta2, eps=1e-08)
		self.synthesizer_optimizer = Adam(alpha, beta1, beta2, eps=1e-08)
		self.discriminator_optimizer = Adam(alpha, beta1, beta2, eps=1e-08)
		if conditional:
			self.generator_embedder_optimizer = Adam(alpha, beta1, beta2, eps=1e-08)
			self.discriminator_embedder_optimizer = Adam(alpha, beta1, beta2, eps=1e-08)
			self.condition_mapper_optimizer = Adam(alpha / 100, beta1, beta2, eps=1e-08)

	def __iter__(self):
		yield "mapper", self.mapper_optimizer
		yield "synthesizer", self.synthesizer_optimizer
		yield "discriminator", self.discriminator_optimizer
		if self.conditional:
			yield "generator_embedder", self.generator_embedder_optimizer
			yield "discriminator_embedder", self.discriminator_embedder_optimizer
			yield "condition_mapper", self.condition_mapper_optimizer

	def setup(self, generator, discriminator):
		self.mapper_optimizer.setup(generator.mapper)
		self.synthesizer_optimizer.setup(generator.synthesizer)
		self.discriminator_optimizer.setup(discriminator.main)
		if self.conditional:
			self.generator_embedder_optimizer.setup(generator.embedder)
			self.discriminator_embedder_optimizer.setup(discriminator.embedder)
			self.condition_mapper_optimizer.setup(discriminator.mapper)

	def update_generator(self):
		self.mapper_optimizer.update()
		self.synthesizer_optimizer.update()
		if self.conditional:
			self.generator_embedder_optimizer.update()

	def update_discriminator(self):
		self.discriminator_optimizer.update()
		if self.conditional:
			self.discriminator_embedder_optimizer.update()
			self.condition_mapper_optimizer.update()

class CustomUpdater(StandardUpdater):

	def __init__(self, generator, averaged_generator, discriminator, iterator, optimizers, averaging_images=10000, lsgan=False):
		super().__init__(iterator, dict(optimizers))
		self.generator = generator
		self.averaged_generator = averaged_generator
		self.discriminator = discriminator
		self.iterator = iterator
		self.optimizers = optimizers
		self.conditional = generator.conditional
		self.averaging_images = averaging_images
		self.lsgan = lsgan
		self.group = None
		self.style_mixing_rate = 0.0
		self.r1_regularization_interval = 0
		self.averaged_path_length = 0.0
		self.path_length_regularization_interval = 0
		self.augumentation = False
		self.augumentation_pipeline = identity
		self.augumentation_probability = 0.0

	def enable_gradient_accumulation(self, group_size):
		self.group = group_size

	def enable_style_mixing(self, mixing_rate=0.5):
		self.style_mixing_rate = mixing_rate

	def enable_r1_regularization(self, gamma=10, interval=16):
		self.r1_regularization_gamma = gamma
		self.r1_regularization_interval = interval

	def enable_path_length_regularization(self, decay=0.99, weight=2, interval=8):
		self.path_length_decay = decay
		self.path_length_penalty_weight = weight
		self.path_length_regularization_interval = interval

	def enable_adaptive_augumentation(self, pipeline, target=0.6, limit=0.8, delta_images=500000, initial_probability=None):
		self.augumentation = True
		self.augumentation_pipeline = pipeline
		self.overfitting_target = target
		self.augumentation_limit = limit
		self.augumentation_delta_images = delta_images
		self.augumentation_probability = initial_probability or self.augumentation_probability

	def freeze_generator(self, levels=[]):
		for i, s in self.generator.synthesizer.blocks:
			if i in levels:
				s.freeze()

	def freeze_discriminator(self, levels=[]):
		for i, s in self.discriminator.blocks:
			if i in levels:
				s.freeze()

	def generate_latents(self, n):
		return self.generator.generate_latents(n)

	def generate_conditions(self, n):
		return self.generator.generate_conditions(n)

	def next_latent_groups(self):
		for _, n in range_batch(self.batch_size, self.group_size):
			yield self.generate_latents(n), (self.generate_conditions(n) if self.conditional else None)

	def next_real_groups(self):
		for group in iter_batch(self.iterator.next(), self.group_size):
			if self.conditional:
				xs, cs = zip(*group)
				x = self.discriminator.xp.array(list(xs))
				c = self.discriminator.xp.array(list(cs))
				yield Variable(x), Variable(c)
			else:
				x = self.discriminator.xp.array(list(group))
				yield Variable(x), None

	def update_core(self):
		if self.augumentation:
			self.apply_augumentation()
		self.update_generator()
		self.average_generator()
		rt = self.update_discriminator()
		if self.augumentation:
			self.adapt_augumentation(rt)

	def update_generator(self):
		accumulated_loss = 0.0
		accumulated_penalty = 0.0
		accumulated_path_length = 0.0
		self.generator.cleargrads()
		for z, c in self.next_latent_groups():
			group_size = z.shape[0]
			mix = self.generate_latents(group_size) if random.random() < self.style_mixing_rate else None
			ws, x_fake = self.generator(z, c, random_mix=mix)
			y_fake = self.discriminator(self.augumentation_pipeline(x_fake), c)
			penalty = 0.0
			if self.path_length_regularization():
				masks = self.generator.generate_masks(group_size)
				path_length = CustomUpdater.path_length(ws, x_fake, masks)
				averaged_path_length = lerp(mean(path_length).item(), self.averaged_path_length, self.path_length_decay)
				weight = self.path_length_penalty_weight * self.path_length_regularization_interval
				penalty = weight * sum((path_length - averaged_path_length) ** 2) / self.batch_size
				accumulated_penalty += penalty.item()
				accumulated_path_length += sum(path_length).item() / self.batch_size
			if self.lsgan:
				loss = CustomUpdater.generator_least_squares_loss(y_fake) / self.batch_size
			else:
				loss = CustomUpdater.generator_logistic_loss(y_fake) / self.batch_size
			accumulated_loss += loss.item()
			(loss + penalty).backward()
		self.optimizers.update_generator()
		report({"loss (G)": accumulated_loss, "penalty (G)": accumulated_penalty})
		if self.path_length_regularization():
			self.averaged_path_length = lerp(accumulated_path_length, self.averaged_path_length, self.path_length_decay)
			report({"path length": self.averaged_path_length})

	def average_generator(self):
		decay = 0.5 ** (self.batch_size / self.averaging_images)
		for raw, averaged in zip(self.generator.params(), self.averaged_generator.params()):
			averaged.copydata(lerp(raw, averaged, decay))

	def update_discriminator(self):
		accumulated_loss = 0.0
		accumulated_penalty = 0.0
		accumulated_rt = 0.0
		self.discriminator.cleargrads()
		for (x_real, c_real), (z, c) in zip(self.next_real_groups(), self.next_latent_groups()):
			group_size = z.shape[0]
			y_real = self.discriminator(self.augumentation_pipeline(x_real), c_real)
			accumulated_rt += sum(sign(y_real - 0.5 if self.lsgan else y_real)).item() / self.batch_size
			mix = self.generate_latents(group_size) if random.random() < self.style_mixing_rate else None
			_, x_fake = self.generator(z, c, random_mix=mix)
			y_fake = self.discriminator(self.augumentation_pipeline(x_fake), c)
			x_fake.unchain_backward()
			penalty = 0.0
			if self.r1_regularization():
				weight = self.r1_regularization_interval * self.r1_regularization_gamma
				penalty = weight * CustomUpdater.gradient_penalty(x_real, y_real) / self.batch_size
				accumulated_penalty += penalty.item()
			if self.lsgan:
				loss = CustomUpdater.discriminator_least_squares_loss(y_real, y_fake) / self.batch_size
			else:
				loss = CustomUpdater.discriminator_logistic_loss(y_real, y_fake) / self.batch_size
			accumulated_loss += loss.item()
			(loss + penalty).backward()
		self.optimizers.update_discriminator()
		report({"loss (D)": accumulated_loss, "penalty (D)": accumulated_penalty, "overfitting": accumulated_rt})
		return accumulated_rt

	def adapt_augumentation(self, overfitting):
		delta = self.batch_size / self.augumentation_delta_images
		direction = sgn(overfitting - self.overfitting_target)
		probability = self.augumentation_probability + delta * direction
		self.augumentation_probability = clamp(0.0, probability, self.augumentation_limit)
		report({"augumentation": self.augumentation_probability})

	def apply_augumentation(self):
		self.augumentation_pipeline.probability = self.augumentation_probability

	def r1_regularization(self):
		return self.r1_regularization_interval and self.iteration % self.r1_regularization_interval == 0

	def path_length_regularization(self):
		return self.path_length_regularization_interval and self.iteration % self.path_length_regularization_interval == 0

	def transfer(self, filepath, generator_levels=[], discriminator_levels=[]):
		with HDF5File(filepath, "r") as hdf5:
			generator_params = Generator.read_params(hdf5["generator"])
			discriminator_kws = ["levels", "first_channels", "last_channels", "categories", "depth"]
			discriminator_params = {k: v for k, v in generator_params.items() if k in discriminator_kws}
			source_generator = Generator(**generator_params)
			source_discriminator = Discriminator(**discriminator_params)
			HDF5Deserializer(hdf5["generator"]).load(source_generator)
			HDF5Deserializer(hdf5["discriminator"]).load(source_discriminator)

	def load_states(self, filepath):
		with HDF5File(filepath, "r") as hdf5:
			self.averaged_path_length = float(hdf5["averaged_path_length"][()])
			self.augumentation_probability = float(hdf5["augumentation_probability"][()])
			HDF5Deserializer(hdf5["generator"]).load(self.generator)
			HDF5Deserializer(hdf5["averaged_generator"]).load(self.averaged_generator)
			HDF5Deserializer(hdf5["discriminator"]).load(self.discriminator)
			for key, optimizer in self.optimizers:
				HDF5Deserializer(hdf5["optimizers"][key]).load(optimizer)

	def save_states(self, filepath):
		with HDF5File(filepath, "w") as hdf5:
			hdf5.create_dataset("averaged_path_length", data=self.averaged_path_length)
			hdf5.create_dataset("augumentation_probability", data=self.augumentation_probability)
			HDF5Serializer(hdf5.create_group("generator")).save(self.generator)
			self.generator.embed_params(hdf5["generator"])
			HDF5Serializer(hdf5.create_group("averaged_generator")).save(self.averaged_generator)
			self.averaged_generator.embed_params(hdf5["averaged_generator"])
			HDF5Serializer(hdf5.create_group("discriminator")).save(self.discriminator)
			optimizer_group = hdf5.create_group("optimizers")
			for key, optimizer in self.optimizers:
				HDF5Serializer(optimizer_group.create_group(key)).save(optimizer)

	@property
	def batch_size(self):
		return self.iterator.batch_size

	@property
	def group_size(self):
		return self.batch_size if self.group is None else self.group

	@staticmethod
	def generator_logistic_loss(fake):
		return sum(softplus(-fake))

	@staticmethod
	def generator_least_squares_loss(fake):
		return sum((fake - 1) ** 2)

	@staticmethod
	def discriminator_logistic_loss(real, fake):
		return sum(softplus(-real)) + sum(softplus(fake))

	@staticmethod
	def discriminator_least_squares_loss(real, fake):
		return sum((real - 1) ** 2) + sum(fake ** 2)

	@staticmethod
	def gradient_penalty(x, y):
		gradient = grad([y], [x], enable_double_backprop=True)[0]
		squared_norm = sum(batch_l2_norm_squared(gradient))
		return squared_norm / 2

	@staticmethod
	def path_length(ws, x, mask):
		levels, batch, size = len(ws), *(ws[0].shape)
		gradients = grad([x * mask], ws, enable_double_backprop=True)
		gradient = stack(gradients).transpose(1, 0, 2).reshape(batch * levels, size)
		path_lengths = batch_l2_norm_squared(gradient).reshape(batch, levels)
		return sqrt(mean(path_lengths, axis=1))

class CustomTrainer(Trainer):

	def __init__(self, updater, epoch, dest, overwrite=False):
		super().__init__(updater, (epoch, "epoch"), dest)
		self.overwrite = overwrite
		self.number = 16
		self.images_out = os.path.join(dest, "images")
		self.states_out = os.path.join(dest, "checkpoints")
		mkdirs(self.images_out)
		mkdirs(self.states_out)

	def enable_reports(self, interval):
		filename = os.path.basename(build_filepath(self.out, "report", "log", self.overwrite))
		log_report = LogReport(trigger=(interval, "iteration"), filename=filename)
		entries = ["iteration", "loss (G)", "loss (D)", "penalty (G)", "penalty (D)", "overfitting"]
		print_report = PrintReport(entries, log_report)
		entries = ["loss (G)", "loss (D)", "penalty (G)", "penalty (D)", "overfitting"]
		filename = os.path.basename(build_filepath(self.out, "curves", "png", self.overwrite))
		plot_report = PlotReport(entries, "iteration", trigger=(interval, "iteration"), filename=filename)
		self.extend(log_report)
		self.extend(print_report)
		self.extend(plot_report)

	def enable_progress_bar(self, interval=1):
		self.extend(ProgressBar(update_interval=interval))

	def hook_state_save(self, interval):
		self.extend(CustomTrainer.save_snapshot, trigger=(interval, "iteration"))
		self.extend(CustomTrainer.save_generator, trigger=(interval, "iteration"))

	def hook_image_generation(self, interval, number=None):
		self.number = self.number if number is None else number
		self.extend(CustomTrainer.save_images, trigger=(interval, "iteration"))

	@property
	def iteration(self):
		return self.updater.iteration

	@property
	def batch_size(self):
		return self.updater.group_size

	@property
	def conditional(self):
		return self.updater.conditional

	@staticmethod
	def save_snapshot(trainer):
		filepath = build_filepath(trainer.states_out, f"snapshot-{trainer.iteration}", "hdf5", trainer.overwrite)
		trainer.updater.save_states(filepath)

	@staticmethod
	def save_generator(trainer):
		filepath = build_filepath(trainer.states_out, f"generator-{trainer.iteration}", "hdf5", trainer.overwrite)
		trainer.updater.averaged_generator.save(filepath)

	@staticmethod
	def save_images(trainer):
		for i, n in range_batch(trainer.number, trainer.batch_size):
			z = trainer.updater.averaged_generator.generate_latents(n)
			c = trainer.updater.averaged_generator.generate_conditions(n) if trainer.conditional else None
			_, y = trainer.updater.averaged_generator(z, c)
			z.to_cpu()
			y.to_cpu()
			for j in range(n):
				filename = f"{trainer.iteration}-{i + j + 1}"
				np.save(build_filepath(trainer.images_out, filename + "-latent", "npy", trainer.overwrite), z.array[j])
				save_image(y.array[j], build_filepath(trainer.images_out, filename, "png", trainer.overwrite))
