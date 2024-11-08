#!/usr/bin/env python3

import cure
cure.patch()

from sys import exit
from stylegan.networks import Generator
from interface.args import CustomArgumentParser
from interface.stdout import chainer_like_tqdm
from utilities.iter import range_batch
from utilities.image import save_image
from utilities.stdio import eprint
from utilities.filesys import mkdirs, open_filepath_write
from utilities.numpy import load, save
from utilities.chainer import to_variable, config_valid

def main(args):
	config_valid()
	print("Loading a model...")
	generator = Generator.load(args.generator)
	generator.to_device(args.device)
	if args.center is not None:
		print("Loading a latent vector...")
		center = to_variable(load(args.center), device=args.device)
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
		if generator.conditional and args.local_truncation:
			mean_w = generator.calculate_mean_ws_by_category()
		else:
			mean_w = generator.calculate_mean_w(categories=(None if args.global_truncation else categories))
	else:
		mean_w = None
	mkdirs(args.dest)
	with chainer_like_tqdm(desc="generation", total=args.number) as bar:
		for i, n in range_batch(args.number, args.batch):
			z = generator.generate_latents(n, center=center, sd=args.sd)
			k, c = generator.generate_conditions(n, categories=categories) if generator.conditional else (None, None)
			mw = mean_w[k] if mean_w is not None and mean_w.ndim == 2 else mean_w
			ws, y = generator(z, c, psi=args.psi, mean_w=mw, noise=args.noisy, fixed=args.fixed)
			z.to_cpu()
			y.to_cpu()
			ws[0].to_cpu()
			for j in range(n):
				filename = f"{i + j + 1}"
				with open_filepath_write(args.dest, filename + "-latent", "npy", args.force) as fp:
					save(fp, z.array[j])
				with open_filepath_write(args.dest, filename + "-style", "npy", args.force) as fp:
					save(fp, ws[0].array[j])
				with open_filepath_write(args.dest, filename, "png", args.force) as fp:
					save_image(y.array[j], fp)
				bar.update()

def parse_args():
	parser = CustomArgumentParser("Generate images from random latent vectors using a trained generator")
	parser.require_generator().add_output_args("images").add_generation_args().add_evaluation_args(default_batch=1)
	return parser.parse_args()


if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(130)
