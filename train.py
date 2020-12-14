from chainer import global_config
from chainer.iterators import SerialIterator
from chainer.optimizers import SGD, Adam
from stylegan.data import ImageDataset
from stylegan.networks import Generator, Discriminator
from stylegan.training import OptimizerSet, CustomUpdater, CustomTrainer
from interface.args import CustomArgumentParser
from interface.argtypes import uint, natural, ufloat, positive, rate
from utilities.filesys import mkdirs, build_filepath

global_config.train = True
global_config.autotune = True
global_config.cudnn_deterministic = False

parser = CustomArgumentParser("")
parser.add_argument("dataset", metavar="DATASET_DIR", help="dataset directory which stores images")
parser.add_argument("-e", "--epoch", type=natural, default=1, help="")
parser.add_argument("-G", "--gamma", "--l2-batch", dest="gamma", type=ufloat, default=10, help="")
parser.add_argument("-L", "--lsgan", "--least-squares", action="store_true", help="")
parser.add_argument("-i", "--mixing", metavar="RATE", dest="mix", type=rate, default=0.5, help="")
parser.add_argument("-S", "--sgd", metavar="LR", type=positive, nargs=3, help="")
parser.add_argument("-A", "--alphas", metavar="ALPHA", type=positive, nargs=3, default=(0.00001, 0.001, 0.001), help="Adam's coefficients of learning rates of mapper, generator, and discriminator")
parser.add_argument("-B", "--betas", metavar="BETA", type=rate, nargs=2, default=(0.0, 0.99), help="Adam's exponential decay rates of the 1st and 2nd order moments")
parser.add_argument("-u", "--print-interval", metavar="ITER", dest="print", type=uint, nargs=2, default=(5, 500), help="")
parser.add_argument("-l", "--write-interval", metavar="ITER", dest="write", type=uint, nargs=4, default=(1000, 3000, 500, 500), help="")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained generator to load and retrain")
parser.add_argument("-d", "--discriminator", metavar="FILE", help="HDF5 file of serialized trained discriminator to load and retrain")
parser.add_argument("-o", "--optimizers", metavar="FILE", nargs=3, help="snapshot of optimizers of mapper, generator, and discriminator")
parser.add_argument("-p", "--preload", action="store_true", help="preload all dataset into RAM")
args = parser.add_output_args(default_dest="results").add_model_args().add_evaluation_args().parse_args()

print("Initializing models")
generator = Generator(args.size, args.depth, args.levels, *args.channels)
discriminator = Discriminator(args.levels, args.channels[1], args.channels[0])
if args.generator is not None: generator.load_model(args.generator)
if args.discriminator is not None: discriminator.load_model(args.discriminator)

mapper_optimizer = SGD(args.sgd[0]) if args.sgd else Adam(args.alphas[0], args.betas[0], args.betas[1], eps=1e-08)
generator_optimizer = SGD(args.sgd[1]) if args.sgd else Adam(args.alphas[1], args.betas[0], args.betas[1], eps=1e-08)
discriminator_optimizer = SGD(args.sgd[2]) if args.sgd else Adam(args.alphas[2], args.betas[0], args.betas[1], eps=1e-08)
optimizers = OptimizerSet(mapper_optimizer, generator_optimizer, discriminator_optimizer)
optimizers.setup(generator, discriminator)
if args.optimizer is not None: optimizers.load_state(args.optimizer)

generator.to_device(args.device)
discriminator.to_device(args.device)
#optimizers.to_device()

dataset = ImageDataset(args.dataset, generator.resolution, args.preload)
iterator = SerialIterator(dataset, args.batch, repeat=True, shuffle=True)
updater = CustomUpdater(generator, discriminator, iterator, optimizers, args.mix, args.gamma, args.lsgan)
trainer = CustomTrainer(updater, args.epoch, args.dest)

"""
# Init optimizers
print("Initializing optimizers")
if args.sgd is None:
	mapper_optimizer = optimizers.Adam(alpha=).setup(generator.mapper)
	print(f"Mapper: Adam(alpha: {args.adam_alphas[0]}, beta1: {args.adam_betas[0]}, beta2: {args.adam_betas[1]})")
	generator_optimizer = optimizers.Adam(alpha=args.adam_alphas[1], beta1=args.adam_betas[0], beta2=args.adam_betas[1], eps=1e-08).setup(generator.generator)
	print(f"Generator: Adam(alpha: {args.adam_alphas[1]}, beta1: {args.adam_betas[0]}, beta2: {args.adam_betas[1]})")
	discriminator_optimizer = optimizers.Adam(alpha=args.adam_alphas[2], beta1=args.adam_betas[0], beta2=args.adam_betas[1], eps=1e-08).setup(discriminator)
	print(f"Discriminator: Adam(alpha: {args.adam_alphas[2]}, beta1: {args.adam_betas[0]}, beta2: {args.adam_betas[1]})")
else:
	mapper_optimizer = optimizers.SGD(args.sgd[0]).setup(generator.mapper)
	print(f"Mapper: SGD(learning rate: {args.sgd[0]})")
	generator_optimizer = optimizers.SGD(args.sgd[1]).setup(generator.generator)
	print(f"Generator: SGD(learning rate: {args.sgd[1]})")
	discriminator_optimizer = optimizers.SGD(args.sgd[2]).setup(discriminator)
	print(f"Discriminator: SGD(learning rate: {args.sgd[2]})")
"""

mkdirs(args.dest)

"""
# Dump command-line options
path = filepath(args.result, "args_quit" if args.quit else "args", "json")
path = path if args.force else altfilepath(path)
with open(path, mode="w", encoding="utf-8") as fp:
	dump(vars(args), fp, indent=2, sort_keys=True)
"""
# Prepare updater
#updater = StyleGanUpdater(generator, discriminator, iterator, {"mapper": mapper_optimizer, "generator": generator_optimizer, "discriminator": discriminator_optimizer}, args.device, args.stage, args.mix, args.alpha, args.delta, args.gamma, args.lsgan)
# Prepare trainer
#trainer = GANTrainer(updater, epoch, )

"""
logpath = filepath(args.result, "report", "log")
logname = basename(logpath if args.force else altfilepath(logpath))
plotpath = filepath(args.result, "plot", "png")
plotname = basename(plotpath if args.force else altfilepath(plotpath))
trainer = Trainer(updater, (args.epoch, "epoch"), out=args.result)
if args.print[0] > 0: trainer.extend(extensions.ProgressBar(update_interval=args.print[0]))
if args.print[1] > 0: trainer.extend(extensions.PrintReport(["epoch", "iteration", "alpha", "loss (gen)", "loss (dis)", "loss (grad)"], extensions.LogReport(trigger=(args.print[1], "iteration"), log_name=None)))

if args.write[0] > 0: trainer.extend(save_middle_images(generator, args.stage, args.result, args.number, args.batch, args.mix, args.force), trigger=(args.write[0], "iteration"))
if args.write[1] > 0: trainer.extend(save_middle_models(generator, discriminator, args.stage, args.result, args.device, args.force), trigger=(args.write[1], "iteration"))
if args.write[1] > 0: trainer.extend(save_middle_optimizers(mapper_optimizer, generator_optimizer, discriminator_optimizer, args.stage, args.result, args.force), trigger=(args.write[1], "iteration"))

if args.write[2] > 0: trainer.extend(extensions.LogReport(trigger=(args.write[2], "iteration"), filename=logname))
if args.write[3] > 0: trainer.extend(extensions.PlotReport(["alpha", "loss (gen)", "loss (dis)", "loss (grad)"], "iteration", trigger=(args.write[3], "iteration"), filename=plotname))
"""

# Run ML
trainer.run()

generator.save_model("gen.hdf5")
discriminator.save_model("dis.hdf5")
optimizers.save_state("opt.hdf5")
