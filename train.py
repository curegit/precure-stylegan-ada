from chainer import global_config
from chainer.iterators import SerialIterator
from stylegan.dataset import ImageDataset, MulticategoryImageDataset
from stylegan.networks import Generator, Discriminator
from stylegan.training import AdamSet, CustomUpdater, CustomTrainer
from stylegan.augmentation import AugmentationPipeline
from interface.args import dump_json, CustomArgumentParser
from interface.argtypes import uint, natural, ufloat, positive, rate
from utilities.stdio import eprint
from utilities.filesys import mkdirs, build_filepath

from tqdm import tqdm
bf = "{desc} [{bar}] {percentage:5.1f}%"
def chainer_like_tqdm(desc, total):
	return tqdm(desc=desc, total=total, bar_format=bf, miniters=1, ascii=".#", ncols=70)

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
	optimizers = AdamSet(args.alpha, args.betas[0], args.betas[1], categories > 1)
	optimizers.setup(generator, discriminator)

	if categories > 1:
		dataset = MulticategoryImageDataset(args.dataset, generator.resolution)
	else:
		dataset = ImageDataset(args.dataset[0], generator.resolution)
	if args.preload:
		with chainer_like_tqdm("dataset", len(dataset)) as bar:
			dataset.preload(lambda: bar.update())
	iterator = SerialIterator(dataset, args.batch, repeat=True, shuffle=True)
	updater = CustomUpdater(generator, averaged_generator, discriminator, iterator, optimizers, categories > 1, args.ema, args.lsgan)
	if args.accum is not None:
		updater.enable_gradient_accumulation(args.accum)
	updater.enable_style_mixing(args.mix)
	updater.enable_r1_regularization(args.gamma, args.r1)
	updater.enable_path_length_regularization(args.decay, args.weight, args.pl)
	if args.ada:
		pipeline = AugmentationPipeline(args.pixel, args.geometric, args.color, args.filtering, args.noise)
		pipeline.to_device(args.device)
		updater.enable_adaptive_augumentation(pipeline, args.target, args.limit, args.delta)
	if args.snapshot is not None:
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
	averaged_generator.save(build_filepath(args.dest, "generator", "hdf5", args.force))
	updater.save_states(build_filepath(args.dest, "snapshot", "hdf5", args.force))

def parse_args():
	parser = CustomArgumentParser("")
	group = parser.add_argument_group("training arguments", "")
	group.add_argument("dataset", metavar="DATASET_DIR", nargs="+", help="dataset directory which stores images")
	group.add_argument("-p", "--preload", action="store_true", help="preload entire dataset into the memory")

	group.add_argument("-s", "--snapshot", metavar="FILE", help="snapshot")
	group.add_argument("-k", "--gradient-accum", dest="accum", type=natural, help="partial batch size")
	group.add_argument("-g", "--group-size", dest="group", type=uint, default=4, help="set 0 to use entire batch")

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

	group.add_argument("-R", "--alpha", metavar="ALPHA", type=positive, default=0.002, help="Adam's coefficient of learning rates of mapper, generator, and discriminator")
	group.add_argument("-B", "--betas", metavar=("BETA1", "BETA2"), type=rate, nargs=2, default=(0.0, 0.99), help="Adam's exponential decay rates of the 1st and 2nd order moments")
	parser.add_argument("-O", "--no-progress-bar", dest="nobar", action="store_true", help="")
	parser.add_argument("-U", "--print-interval", metavar="ITER", dest="print", type=uint, default=1000, help="")
	parser.add_argument("-S", "--save-interval", metavar="ITER", dest="save", type=uint, default=2000, help="")
	return parser.add_output_args(default_dest="results").add_model_args().add_evaluation_args().parse_args()

if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
