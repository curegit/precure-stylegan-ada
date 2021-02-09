#from json import dump
#from shutil import rmtree
#from argparse import ArgumentParser
import numpy as np
from chainer import global_config
from stylegan.networks import Generator
from interface.args import CustomArgumentParser
#from modules.argtypes import uint, natural, ufloat, positive, rate, filename, device
from utilities.image import save_image
from utilities.filesys import mkdirs, build_filepath
from utilities.iter import range_batch

# Parse command line arguments
parser = CustomArgumentParser("")
parser.add_output_args("images").add_model_args().add_evaluation_args().add_generation_args()
#parser.add_argument("-q", "--quit", action="store_true", help="")
#parser.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
#parser.add_argument("-w", "--wipe", action="store_true", help="")
#parser.add_argument("-j", "--dump-json", action="store_true", help="")
#parser.add_argument("-i", "--image-only", action="store_true", help="")
'''
parser.add_argument("-r", "--result", "-d", "--directory", metavar="DEST", dest="directory", default="images", help="destination directory for generated images")
parser.add_argument("-p", "--prefix", type=filename, default="", help="filename prefix for generated images")
parser.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained model to load")
parser.add_argument("-s", "--stage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="growth stage, defining image resolution")
parser.add_argument("-x", "--max-stage", dest="maxstage", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9], default=7, help="final stage")
parser.add_argument("-c", "--channels", metavar="CH", type=natural, nargs=2, default=(512, 16), help="numbers of channels at initial stage and final stage")
parser.add_argument("-z", "--z-size", dest="size", type=natural, default=512, help="latent vector (feature vector) size")
parser.add_argument("-m", "--mlp-depth", metavar="DEPTH", dest="depth", type=natural, default=8, help="MLP depth of mapping network")
parser.add_argument("-n", "--number", type=uint, default=1, help="the number of images to generate")
parser.add_argument("-b", "--batch", type=natural, default=1, help="batch size, affecting memory usage")
parser.add_argument("-a", "--alpha", type=rate, default=1.0, help="")
parser.add_argument("-l", "--latent", "--center", metavar="FILE", dest="center", help="")
parser.add_argument("-e", "--deviation", "--sd", metavar="SIGMA", dest="sd", type=positive, default=1.0, help="")
parser.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=ufloat, help="")
parser.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=-1, help="use specified GPU or CPU device")
'''
args = parser.parse_args()

# Config chainer
global_config.train = False
global_config.autotune = True
global_config.cudnn_deterministic = True

# Init model
print("Initializing model")
generator = Generator(args.size, args.depth, args.levels, *args.channels)

# Print information
#args.stage = args.maxstage if args.stage > args.maxstage else args.stage
#h, w = generator.resolution(args.stage)
#print(f"Total Generation: {args.number}, Batch: {args.batch}")
#print(f"MLP: {args.size}x{args.depth}, Stage: {args.stage}/{args.maxstage} ({w}x{h})")
#print(f"Channel: {args.channels[0]} (initial) -> {args.channels[1]} (final)")
#print(f"Alpha: {args.alpha}, Latent: {'Yes' if args.center is not None else 'No'}, Deviation: {args.sd}")
#print(f"Truncation Trick: {args.psi if args.psi is not None else 'No'}, Device: {'CPU' if args.device < 0 else f'GPU {args.device}'}")

# Load model
if args.generator is not None:
	print("Loading generator")
	generator.load_state(args.generator)

# GPU setting
generator.to_device(args.device)

# Load center latent
'''
if args.center is not None:
	print("Loading latent")
	center = generator.wrap_latent(load_array(args.center))
else:
	center = None
'''

# Init destination folder
#print("Initializing destination directory")
#if args.wipe:
#	rmtree(args.directory, ignore_errors=True)
mkdirs(args.dest)

'''
# Dump command-line options
if args.dump_json:
	path = filepath(args.directory, "args_quit" if args.quit else "args", "json")
	path = path if args.force else altfilepath(path)
	with open(path, mode="w", encoding="utf-8") as fp:
		dump(vars(args), fp, indent=2, sort_keys=True)

# Quit mode
if args.quit:
	print("Finished (Quit mode)")
	exit(0)
'''

for i, n in range_batch(args.number, args.batch):
	#mixing = mix > random()
	z = generator.generate_latents(n)
	#mix_z = generator.generate_latent(n) if mixing else None
	y = generator(z)
	z.to_cpu()
	y.to_cpu()
	for j in range(n):
		filename = f"{i + j + 1}"
		np.save(build_filepath(args.dest, filename, "npy", args.force), z.array[j])
		save_image(y.array[j], build_filepath(args.dest, filename, "png", args.force))
'''
# Generate images
c = 0
mean_w = None if args.psi is None else generator.calculate_mean_w()
while c < args.number:
	n = min(args.number - c, args.batch)
	z = generator.generate_latent(n, center=center, sd=args.sd)
	y = generator(z, args.stage, alpha=args.alpha, psi=args.psi, mean_w=mean_w)
	z.to_cpu()
	y.to_cpu()
	for i in range(n):
		path = filepath(args.directory, f"{args.prefix}{c + i + 1}", "png")
		path = path if args.force else altfilepath(path)
		save_image(y.array[i], path)
		print(f"{c + i + 1}/{args.number}: Saved as {path}")
		if not args.image_only:
			path = filepath(args.directory, f"{args.prefix}{c + i + 1}", "npy")
			path = path if args.force else altfilepath(path)
			save_array(z.array[i], path)
	c += n
'''
