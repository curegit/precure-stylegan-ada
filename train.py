from chainer import global_config
from chainer.iterators import SerialIterator
from stylegan.dataset import ImageDataset, MulticategoryImageDataset
from stylegan.networks import Generator, Discriminator
from stylegan.training import AdamSet, CustomUpdater, CustomTrainer
from stylegan.augmentation import AugmentationPipeline
from interface.args import dump_json, CustomArgumentParser
from interface.argtypes import uint, natural, ufloat, positive, rate
from interface.stdout import chainer_like_tqdm, print_model_args, print_data_classes, print_training_args
from utilities.stdio import eprint
from utilities.filesys import mkdirs, build_filepath

def main(args):
	global_config.train = True
	global_config.autotune = True
	global_config.cudnn_deterministic = False
	print("Initializing models...")
	categories = len(args.dataset)
	generator = Generator(args.size, args.depth, args.levels, *args.channels, categories)
	discriminator = Discriminator(args.levels, args.channels[1], args.channels[0], categories, args.depth, args.group)
	averaged_generator = generator.copy("copy")
	generator.to_device(args.device)
	discriminator.to_device(args.device)
	averaged_generator.to_device(args.device)
	print_model_args(generator)
	optimizers = AdamSet(args.alpha, args.betas[0], args.betas[1], categories > 1)
	optimizers.setup(generator, discriminator)
	print("Preparing dataset...")
	if categories > 1:
		dataset = MulticategoryImageDataset(args.dataset, generator.resolution)
	else:
		dataset = ImageDataset(args.dataset[0], generator.resolution)
	print(f"Dataset size: {len(dataset)} images")
	if categories > 1:
		generator.embed_labels(args.labels)
	print_data_classes(generator)
	if args.preload and args.nobar:
		dataset.preload()
	elif args.preload:
		with chainer_like_tqdm("dataset", len(dataset)) as bar:
			dataset.preload(lambda: bar.update())
	print("Setting up training....")
	print_training_args(args)
	iterator = SerialIterator(dataset, args.batch, repeat=True, shuffle=True)
	updater = CustomUpdater(generator, averaged_generator, discriminator, iterator, optimizers, args.ema, args.lsgan)
	if args.accum is not None:
		updater.enable_gradient_accumulation(args.accum)
	updater.enable_style_mixing(args.mix)
	if args.gamma > 0:
		updater.enable_r1_regularization(args.gamma, args.r1)
	if args.weight > 0:
		updater.enable_path_length_regularization(args.decay, args.weight, args.pl)
	if args.ada:
		print("Enabling ADA...")
		pipeline = AugmentationPipeline(args.pixel, args.geometric, args.color, args.filtering, args.noise)
		pipeline.to_device(args.device)
		updater.enable_adaptive_augumentation(pipeline, args.target, args.limit, args.delta)
	if args.snapshot is not None:
		print("Loading snapshot....")
		updater.load_states(args.snapshot)
	mkdirs(args.dest)
	dump_json(args, build_filepath(args.dest, "arguments", "json", args.force))
	trainer = CustomTrainer(updater, args.epoch, args.dest, args.force)
	if args.save != 0:
		trainer.hook_state_save(args.save)
		trainer.hook_image_generation(args.save, args.number)
	if args.print != 0:
		trainer.enable_reports(args.print)
	if not args.nobar:
		trainer.enable_progress_bar(1)
	trainer.run()
	print("Saving results...")
	averaged_generator.save(build_filepath(args.dest, "generator", "hdf5", args.force))
	updater.save_states(build_filepath(args.dest, "snapshot", "hdf5", args.force))

def check_args(args):
	if len(args.dataset) == 1 and args.labels:
		eprint("Unconditional model cannot have labels!")
		raise RuntimeError("Label error")
	if len(args.dataset) > 1 and args.labels and len(args.labels) != len(args.dataset):
		eprint("You must provide the same number of data classes and labels!")
		raise RuntimeError("Label error")
	if args.labels and len(args.labels) != len(set(args.labels)):
		eprint("Labels are not unique!")
		raise RuntimeError("Label error")
	if args.accum is None:
		if args.group == 0:
			return args
		if args.batch % args.group == 0:
			return args
		eprint("Batch size is not divisible by group size!")
	else:
		if args.group == 0:
			return args
		if args.accum % args.group == 0:
			if (args.batch % args.accum) % args.group == 0:
				return args
			eprint("Last accumulation size is not divisible by group size!")
		else:
			eprint("Accumulation size is not divisible by group size!")
	raise RuntimeError("Incompatible grouping configuration")

def parse_args():
	parser = CustomArgumentParser("")
	group = parser.add_argument_group("training arguments", "")
	group.add_argument("dataset", metavar="DATASET_DIR", nargs="+", help="dataset directory which stores images")
	group.add_argument("-p", "--preload", action="store_true", help="preload entire dataset into the memory")

	group.add_argument("-s", "--snapshot", metavar="FILE", help="snapshot")
	group.add_argument("-b", "--batch", type=natural, default=16, help="batch")
	group.add_argument("-k", "--accum", dest="accum", type=natural, help="partial batch size")
	group.add_argument("-g", "--group", dest="group", type=uint, default=0, help="set 0 to use entire batch")

	group.add_argument("-e", "--epoch", type=natural, default=1, help="")
	group.add_argument("-r", "--r1-gamma", dest="gamma", type=ufloat, default=10, help="")
	group.add_argument("-t", "--r1-interval", dest="r1", type=natural, default=16, help="")
	group.add_argument("-y", "--pl-decay", dest="decay", type=rate, default=0.99, help="")
	group.add_argument("-w", "--pl-weight", dest="weight", type=ufloat, default=2, help="")
	group.add_argument("-l", "--pl-interval", dest="pl", type=natural, default=8, help="")

	group.add_argument("-A", "--ada", action="store_true", help="")
	group.add_argument("-T", "--target", metavar="RATE", type=rate, default=0.6, help="")
	group.add_argument("-M", "--limit", metavar="RATE", type=rate, default=0.8, help="")
	group.add_argument("-D", "--delta", metavar="N", type=natural, default=500000, help="")

	group.add_argument("-I", "--pixel", metavar="P", type=rate, default=1.0, help="")
	group.add_argument("-G", "--geometric", metavar="P", type=rate, default=1.0, help="")
	group.add_argument("-C", "--color", metavar="P", type=rate, default=1.0, help="")
	group.add_argument("-F", "--filtering", metavar="P", type=rate, default=1.0, help="")
	group.add_argument("-N", "--noise", metavar="P", type=rate, default=1.0, help="")

	group.add_argument("-L", "--lsgan", "--least-squares", action="store_true", help="")
	group.add_argument("-i", "--mixing-rate", metavar="RATE", dest="mix", type=rate, default=0.5, help="")
	group.add_argument("-a", "--ema-images", metavar="N", dest="ema", type=natural, default=10000, help="")

	group.add_argument("-n", "--number", type=uint, default=32, help="the number of middle images to generate")

	group.add_argument("-R", "--alpha", metavar="ALPHA", type=positive, default=0.002, help="Adam's coefficient of learning rates of mapper, generator, and discriminator")
	group.add_argument("-B", "--betas", metavar=("BETA1", "BETA2"), type=rate, nargs=2, default=(0.0, 0.99), help="Adam's exponential decay rates of the 1st and 2nd order moments")
	parser.add_argument("-O", "--no-progress-bar", dest="nobar", action="store_true", help="")
	parser.add_argument("-U", "--print-interval", metavar="ITER", dest="print", type=uint, default=1000, help="")
	parser.add_argument("-S", "--save-interval", metavar="ITER", dest="save", type=uint, default=2000, help="")
	return parser.add_output_args(default_dest="results").add_model_args().add_evaluation_args(include_batch=False).parse_args()

if __name__ == "__main__":
	try:
		main(check_args(parse_args()))
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
