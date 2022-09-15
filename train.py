from os.path import basename, realpath
from chainer import global_config
from chainer.iterators import SerialIterator
from stylegan.dataset import ImageDataset, MulticategoryImageDataset
from stylegan.networks import Generator, Discriminator
from stylegan.training import AdamSet, CustomUpdater, CustomTrainer
from stylegan.augmentation import AugmentationPipeline
from interface.args import dump_json, CustomArgumentParser
from interface.argtypes import uint, natural, ufloat, positive, rate
from interface.stdout import chainer_like_tqdm, print_model_args, print_parameter_counts, print_cnn_architecture, print_data_classes, print_training_args
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
	print_parameter_counts(generator, discriminator)
	print_cnn_architecture(generator, discriminator)
	optimizers = AdamSet(args.alpha, args.betas[0], args.betas[1], categories > 1)
	optimizers.setup(generator, discriminator)
	print("Preparing a dataset...")
	if categories > 1:
		dataset = MulticategoryImageDataset(args.dataset, generator.resolution)
	else:
		dataset = ImageDataset(args.dataset[0], generator.resolution)
	print(f"Dataset size: {len(dataset)} images")
	if args.labels:
		generator.embed_labels(args.labels)
		averaged_generator.embed_labels(args.labels)
	print_data_classes(generator)
	if args.preload and args.nobar:
		dataset.preload()
	elif args.preload:
		with chainer_like_tqdm("dataset", len(dataset)) as bar:
			dataset.preload(lambda: bar.update())
	print("Setting up training...")
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
		print(f"- pixel: {args.pixel * 100}%")
		print(f"- geometric: {args.geometric * 100}%")
		print(f"- color: {args.color * 100}%")
		print(f"- filtering: {args.filtering * 100}%")
		print(f"- noise: {args.noise * 100}%")
		pipeline = AugmentationPipeline(args.pixel, args.geometric, args.color, args.filtering, args.noise)
		pipeline.to_device(args.device)
		updater.enable_adaptive_augumentation(pipeline, args.target, args.limit, args.delta)
	if args.snapshot is not None:
		print("Loading a snapshot...")
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
	print("Training started")
	trainer.run()
	print("Saving results...")
	gen_path = build_filepath(args.dest, "generator", "hdf5", args.force)
	averaged_generator.save(gen_path)
	print(f"Saved: {gen_path}")
	dis_path = build_filepath(args.dest, "snapshot", "hdf5", args.force)
	updater.save_states(dis_path)
	print(f"Saved: {dis_path}")

def check_args(args):
	if len(args.dataset) == 1 and args.labels:
		eprint("Unconditional model cannot have labels!")
		raise RuntimeError("Label error")
	if len(args.dataset) > 1 and args.labels and len(args.labels) != len(args.dataset):
		eprint("You must provide the same number of data classes and labels!")
		raise RuntimeError("Label error")
	if args.labels:
		for l in args.labels:
			if not l:
				eprint("Empty strings are not allowed for labels!")
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
	eprint("Incompatible grouping configuration")
	raise RuntimeError("Conflict error")

def preprocess_args(args):
	if args.labels is not None and len(args.labels) == 0:
		args.labels = [basename(realpath(d)) for d in args.dataset]
	return args

def parse_args():
	parser = CustomArgumentParser("Train a conditional or unconditional StyleGAN 2.0 model")
	parser.add_output_args(default_dest="results").add_model_args()
	parser.add_argument("dataset", metavar="DATASET_DIR", nargs="+", help=f"dataset directory that includes real images (specify multiple directories to train conditional models, one directory per image class), supported file extensions: {', '.join(ImageDataset.extensions)}")
	parser.add_argument("-p", "--preload", action="store_true", help="preload entire dataset into the RAM")
	parser.add_argument("-l", "--labels", metavar="CLASS", nargs="*", help="embed data class labels into output generators (provide CLASS as many as dataset directories), dataset directory names are automatically used if no CLASS arguments are given")
	group = parser.add_argument_group("training arguments")
	group.add_argument("-s", "--snapshot", metavar="HDF5_FILE", help="load weights and parameters from a snapshot (for resuming)")
	group.add_argument("-e", "--epoch", metavar="N", type=natural, default=1, help="training duration in epoch (note that training duration will not be serialized in snapshot)")
	group.add_argument("-b", "--batch", metavar="N", type=natural, default=16, help="batch size, affecting not only memory usage, but also training result")
	group.add_argument("-k", "--accum", metavar="N", dest="accum", type=natural, help="enable the gradient accumulation and specify its partial batch size")
	group.add_argument("-g", "--group", metavar="N", dest="group", type=uint, default=0, help="group size of the minibatch standard deviation (set 0 to use entire batch)")
	group.add_argument("-m", "--mixing-rate", metavar="RATE", dest="mix", type=rate, default=0.5, help="application rate of the mixing regularization")
	group.add_argument("-r", "--r1-gamma", metavar="GAMMA", dest="gamma", type=ufloat, default=10.0, help="coefficient of R1 regularization (set 0 to disable)")
	group.add_argument("-i", "--r1-interval", metavar="ITER", dest="r1", type=natural, default=16, help="apply R1 regularization every ITER iteration (lazy regularization)")
	group.add_argument("-w", "--pl-weight", metavar="W", dest="weight", type=ufloat, default=2.0, help="coefficient of the path length regularization (set 0 to disable)")
	group.add_argument("-y", "--pl-decay", metavar="RATE", dest="decay", type=rate, default=0.99, help="decay rate of the path length regularization")
	group.add_argument("-t", "--pl-interval", metavar="ITER", dest="pl", type=natural, default=8, help="apply the path length regularization every ITER iteration (lazy regularization)")
	group.add_argument("-L", "--lsgan", "--least-squares", action="store_true", help="use the least squares loss function instead of the logistic loss function")
	group.add_argument("-E", "--ema-images", metavar="N", dest="ema", type=natural, default=10000, help="period of the exponential moving average for generator weights (enlarge the value to take longer)")
	group.add_argument("-A", "--alpha", "--lr", metavar="ALPHA", type=positive, default=0.002, help="Adam's learning rate (shared both generator and discriminator)")
	group.add_argument("-B", "--betas", metavar=("BETA1", "BETA2"), type=rate, nargs=2, default=(0.0, 0.99), help="Adam's exponential decay rates of the 1st and 2nd order moments")
	group = parser.add_argument_group("augmentation arguments")
	group.add_argument("-a", "--ada", action="store_true", help="enable the adaptive discriminator augmentation")
	group.add_argument("-T", "--target", metavar="RATE", type=rate, default=0.6, help="target value of the discriminator overfitting heuristic indicator")
	group.add_argument("-M", "--limit", metavar="RATE", type=rate, default=0.8, help="upper limit of the augmentation probability")
	group.add_argument("-D", "--delta", metavar="N", type=natural, default=500000, help="control the amount of an augmentation probability update (use a smaller value for bigger updates)")
	group.add_argument("-I", "--pixel", "--integer", metavar="P", type=ufloat, default=1.0, help="application rate multiplier of the by-pixel transformation augmentation")
	group.add_argument("-G", "--geometric", metavar="P", type=ufloat, default=1.0, help="application rate multiplier of the general geometric transformation augmentation")
	group.add_argument("-C", "--color", metavar="P", type=ufloat, default=1.0, help="application rate multiplier of the color transformation augmentation")
	group.add_argument("-F", "--filtering", metavar="P", type=ufloat, default=1.0, help="application rate multiplier of the filtering augmentation")
	group.add_argument("-N", "--noise", metavar="P", type=ufloat, default=1.0, help="application rate multiplier of the noise augmentation")
	parser.add_argument("-J", "--no-progress-bar", dest="nobar", action="store_true", help="don't show progress bars")
	parser.add_argument("-P", "--print-interval", metavar="ITER", dest="print", type=uint, default=1000, help="print statistics every ITER iteration")
	parser.add_argument("-S", "--save-interval", metavar="ITER", dest="save", type=uint, default=2000, help="save snapshots, statistics and middle images every ITER iteration")
	parser.add_argument("-n", "--number", metavar="N", type=uint, default=32, help="the number of middle images to generate each save-time")
	return parser.add_evaluation_args(include_batch=False, include_noise=False).parse_args()

if __name__ == "__main__":
	try:
		main(check_args(preprocess_args(parse_args())))
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
