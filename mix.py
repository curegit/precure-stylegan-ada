#!/usr/bin/env python3

import numpy as np
from chainer.functions import stack
from stylegan.networks import Generator
from interface.args import CustomArgumentParser
from interface.argtypes import uint
from interface.stdout import chainer_like_tqdm
from utilities.iter import first, range_batch
from utilities.math import lerp, ilerp
from utilities.image import save_image
from utilities.stdio import eprint
from utilities.filesys import mkdirs, build_filepath
from utilities.chainer import to_variable, config_valid

def lerp_ellipsis(xs):
	ys = []
	left, lw = None, None
	right, rw = first(xs, lambda x: x is not ...)
	for i, w in enumerate(xs):
		if w is ...:
			if lw is None:
				if rw is None:
					raise ValueError()
				ys.append(rw)
			elif rw is None:
				if lw is None:
					raise ValueError()
				ys.append(lw)
			else:
				ys.append(lerp(lw, rw, ilerp(left, right, i)))
		else:
			ys.append(w)
			left, lw = i, w
			right, (_, rw) = first(enumerate(xs), lambda ix: ix[0] > i and ix[1] is not ..., default=(-1, None))
	return ys

def justify(xs, length, align_end=False, fill=...):
	ys = []
	ixs = enumerate(xs)
	j, x = next(ixs)
	ys.append(x)
	for i in range(1, length):
		if align_end:
			k = i * (len(xs) - 1) // (length - 1)
		else:
			k = i * len(xs) // length
		if k >= j + 1:
			j, x = next(ixs)
			ys.append(x)
		else:
			ys.append(fill)
	return ys

def main(args):
	config_valid()
	print("Loading a model...")
	generator = Generator.load(args.generator)
	generator.to_device(args.device)
	ws = []
	for s in args.style:
		if (s == "..."):
			if args.lerp:
				ws.append(...)
			elif (ws):
				ws.append(ws[-1])
			else:
				eprint("You must supply a 1st level style!")
				raise RuntimeError("Input error")
		else:
			ws.append(to_variable(np.load(s), device=args.device))
	if all(w is ... for w in ws):
		eprint("You must supply at least one style file!")
		raise RuntimeError("Input error")
	if (len(ws) > generator.levels):
		eprint("Too many styles!")
		raise RuntimeError("Input error")
	if (len(ws) != generator.levels):
		ws += [ws[-1]] * (generator.levels - len(ws))
	if args.lerp:
		ws = lerp_ellipsis(ws)
	mkdirs(args.dest)
	with chainer_like_tqdm(desc="generation", total=args.number) as bar:
		for i, n in range_batch(args.number, args.batch):
			y = generator.synthesizer([stack([w] * n) for w in ws], noise=args.noisy, fixed=args.fixed)
			y.to_cpu()
			for j in range(n):
				filename = f"{i + j + 1}"
				save_image(y.array[j], build_filepath(args.dest, filename, "png", args.force))
				bar.update()

def parse_args():
	parser = CustomArgumentParser("Mix style vectors to compose feature-mixed images")
	parser.require_generator().add_output_args("mixtures")
	parser.add_argument("style", metavar="STYLE_FILE", nargs="+", help="input style NPY file for each level, specify '...' to use the previous level's one (you can omit the tailing '...')")
	parser.add_argument("-a", "--auto", action="store_true", help="")
	parser.add_argument("-l", "--lerp", action="store_true", help="")
	parser.add_argument("-n", "--number", metavar="N", type=uint, default=10, help="the number of images to generate")
	return parser.add_evaluation_args().parse_args()

if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
