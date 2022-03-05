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
	print("Loading a model...")
	generator = Generator.load(args.generator)
	generator.to_device(args.device)
	if args.center is not None:
		print("Loading a latent vector...")
		center = Variable(np.load(args.center))
	else:
		center = None
	if args.classes is not None or args.labels is not None:
		if not generator.conditional:
			eprint("Unconditional model doesn't have image classes!")
			raise RuntimeError("Class error")
		categories = [] if args.classes is None else list(args.classes)
		categories += [] if args.labels is None else [generator.lookup_label(l) for l in args.labels]
		if any(c >= generator.categories for c in categories):
			eprint("Some class numbers are not in the valid range!")
			raise RuntimeError("Class error")
	else:
		categories = None
	if args.psi != 1.0:
		mean_w = generator.calculate_mean_w()
	else:
		mean_w = None
	mkdirs(args.dest)
	with chainer_like_tqdm(desc="generation", total=args.number) as bar:
		for i, n in range_batch(args.number, args.batch):
			z = generator.generate_latents(n, center=center, sd=args.sd)
			c = generator.generate_conditions(n, categories=categories) if generator.conditional else None
			ws, y = generator(z, c, psi=args.psi, mean_w=mean_w)
			z.to_cpu()
			y.to_cpu()
			for j in range(n):
				filename = f"{i + j + 1}"
				np.save(build_filepath(args.dest, filename + ".latent", "npy", args.force), z.array[j])
				for k, w in enumerate(ws, 1):
					w.to_cpu()
					np.save(build_filepath(args.dest, filename + f"-{k}.style", "npy", args.force), w.array[j])
				save_image(y.array[j], build_filepath(args.dest, filename, "png", args.force))
				bar.update()

def parse_args():
	parser = CustomArgumentParser("Generate images of a trained generator from random latent vectors")
	parser.require_generator().add_output_args("images").add_generation_args().add_evaluation_args()
	return parser.parse_args()

if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
