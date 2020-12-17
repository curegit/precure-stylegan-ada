from json import dump
from chainer import global_config
from chainer.iterators import SerialIterator
from chainer.optimizers import SGD, Adam
from stylegan.dataset import ImageDataset
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
if args.generator is not None: generator.load_state(args.generator)
if args.discriminator is not None: discriminator.load_state(args.discriminator)

mapper_optimizer = SGD(args.sgd[0]) if args.sgd else Adam(args.alphas[0], args.betas[0], args.betas[1], eps=1e-08)
generator_optimizer = SGD(args.sgd[1]) if args.sgd else Adam(args.alphas[1], args.betas[0], args.betas[1], eps=1e-08)
discriminator_optimizer = SGD(args.sgd[2]) if args.sgd else Adam(args.alphas[2], args.betas[0], args.betas[1], eps=1e-08)
optimizers = OptimizerSet(mapper_optimizer, generator_optimizer, discriminator_optimizer)
if args.optimizers is not None: optimizers.load_states(args.optimizers)
optimizers.setup(generator, discriminator)

generator.to_device(args.device)
discriminator.to_device(args.device)

mkdirs(args.dest)
dataset = ImageDataset(args.dataset, generator.resolution, args.preload)
iterator = SerialIterator(dataset, args.batch, repeat=True, shuffle=True)
updater = CustomUpdater(generator, discriminator, iterator, optimizers, args.mix, args.gamma, args.lsgan)
trainer = CustomTrainer(updater, args.epoch, args.dest)
trainer.hook_state_save(1000)
trainer.hook_image_generation(1000, 32)
trainer.enable_reports(500)
trainer.enable_progress_bar(1)
trainer.run()

generator.save_state(build_filepath(args.dest, "gen", "hdf5"))
discriminator.save_state(build_filepath(args.dest, "dis", "hdf5"))
optimizers.save_states(build_filepath(args.dest, "opt", "hdf5"))
