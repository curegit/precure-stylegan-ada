#!/usr/bin/env python3

import cure
cure.patch()

from sys import exit
from numpy import array
from chainer.functions import stack
from stylegan.networks import Generator
from interface.args import CustomArgumentParser
from interface.argtypes import uint, real
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
	ws = array([load(s) for s in args.style])
	w = sum(w * k for w, k in zip(ws, args.coefs))
	mkdirs(args.dest)
	with open_filepath_write(args.dest, "new-style", "npy", args.force) as fp:
		save(fp, w)
	with chainer_like_tqdm(desc="generation", total=args.number) as bar:
		for i, n in range_batch(args.number, args.batch):
			y = generator.synthesizer([stack([to_variable(w, device=args.device)] * n)] * generator.levels, noise=args.noisy, fixed=args.fixed)
			y.to_cpu()
			for j in range(n):
				filename = f"{i + j + 1}"
				with open_filepath_write(args.dest, filename, "png", args.force) as fp:
					save_image(y.array[j], fp)
				bar.update()

def check_args(args):
	if len(args.coefs) != len(args.style):
		eprint("The number of coefficients doesn't match the number of styles!")
		raise RuntimeError("Input error")
	return args

def preprocess_args(args):
	if args.coefs is None:
		n = len(args.style)
		args.coefs = [1 / n for i in range(n)]
	return args

def parse_args():
	parser = CustomArgumentParser("Create a new style vector and its images by linear combination of style vectors")
	parser.require_generator().add_output_args("combined")
	parser.add_argument("style", metavar="STYLE_FILE", nargs="+", help="input style NPY file")
	parser.add_argument("-c", "--coefs", dest="coefs", metavar="K", nargs="+", type=real, help="linear combination coefficients to multiply style vectors respectively (average all styles when this option is not used)")
	parser.add_argument("-n", "--number", metavar="N", type=uint, default=3, help="the number of images to generate")
	return parser.add_evaluation_args(default_batch=1).parse_args()


if __name__ == "__main__":
	try:
		main(check_args(preprocess_args(parse_args())))
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(130)
