#!/usr/bin/env python3

import numpy as np
from chainer.functions import stack
from stylegan.networks import Generator
from interface.args import CustomArgumentParser
from interface.argtypes import uint
from interface.stdout import chainer_like_tqdm
from utilities.image import save_image
from utilities.stdio import eprint
from utilities.filesys import mkdirs, build_filepath
from utilities.iter import range_batch
from utilities.chainer import to_variable, config_valid

def main(args):
	config_valid()
	print("Loading a model...")
	generator = Generator.load(args.generator)
	generator.to_device(args.device)
	ws = []
	for s in args.style:
		if (s == "..."):
			if (ws):
				ws.append(ws[-1])
			else:
				eprint("You must supply a 1st level style!")
				raise RuntimeError("Input error")
		else:
			ws.append(to_variable(np.load(s), device=args.device))
	if (len(ws) > generator.levels):
		eprint("Too many styles!")
		raise RuntimeError("Input error")
	elif (len(ws) != generator.levels):
		ws += [ws[-1]] * (generator.levels - len(ws))
	mkdirs(args.dest)
	with chainer_like_tqdm(desc="generation", total=args.number) as bar:
		for i, n in range_batch(args.number, args.batch):
			y = generator.synthesizer([stack([w] * n) for w in ws], noise=args.noisy, freeze=args.freeze)
			y.to_cpu()
			for j in range(n):
				filename = f"{i + j + 1}"
				save_image(y.array[j], build_filepath(args.dest, filename, "png", args.force))
				bar.update()

def parse_args():
	parser = CustomArgumentParser("Mix style vectors to compose feature-mixed images")
	parser.require_generator().add_output_args("mixtures")
	parser.add_argument("style", metavar="STYLE_FILE", nargs="+", help="input style NPY file for each level, specify '...' to use the previous one (you can omit the tailing '...')")
	parser.add_argument("-n", "--number", metavar="N", type=uint, default=10, help="the number of images to generate")
	return parser.add_evaluation_args().parse_args()

if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
