import numpy as np
from chainer import global_config, Variable
from stylegan.networks import Generator
from interface.args import CustomArgumentParser
from interface.stdout import chainer_like_tqdm
from utilities.image import save_image
from utilities.stdio import eprint
from utilities.filesys import mkdirs, build_filepath
from utilities.iter import range_batch

def main(args):
	global_config.train = False
	global_config.autotune = True
	global_config.cudnn_deterministic = True
	print("Loading model...")
	generator = Generator.load(args.generator)
	generator.to_device(args.device)
	if args.center is not None:
		print("Loading latent...")
		center = Variable(np.load(args.center))
	else:
		center = None
	if args.label is not None:
		if not generator.conditional:
			eprint("Unconditional model doesn't have labels!")
			raise RuntimeError("Label error")
		category = generator.lookup_label(args.label)
		if category is None:
			eprint("No such label in the model!")
			raise RuntimeError("Label error")
	else:
		category = None
	if args.psi != 1.0:
		mean_w = generator.calculate_mean_w()
	else:
		mean_w = None
	mkdirs(args.dest)
	with chainer_like_tqdm(desc="generation", total=args.number) as bar:
		for i, n in range_batch(args.number, args.batch):
			z = generator.generate_latents(n, center=center, sd=args.sd)
			c = generator.generate_conditions(n, category=category) if generator.conditional else None
			ws, y = generator(z, c, psi=args.psi, mean_w=mean_w)
			z.to_cpu()
			y.to_cpu()
			for j in range(n):
				filename = f"{i + j + 1}"
				np.save(build_filepath(args.dest, filename, "npy", args.force), z.array[j])
				save_image(y.array[j], build_filepath(args.dest, filename, "png", args.force))
				bar.update()

def check_args(args):
	return args

def parse_args():
	parser = CustomArgumentParser("")
	parser.require_generator().add_output_args("images").add_model_args().add_evaluation_args().add_generation_args()
	return parser.parse_args()

if __name__ == "__main__":
	try:
		main(check_args(parse_args()))
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
