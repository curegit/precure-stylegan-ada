#!/usr/bin/env python3

from sys import exit
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
from utilities.numpy import load
from utilities.chainer import to_variable, config_valid

def justify(xs, length, align_end=False, fill=...):
	ys = []
	ixs = enumerate(xs)
	i, x = next(ixs)
	ys.append(x)
	for j in range(1, length):
		if align_end:
			k = j * (len(xs) - 1) // (length - 1)
		else:
			k = j * len(xs) // length
		if k >= i + 1:
			i, x = next(ixs)
			ys.append(x)
		else:
			ys.append(fill)
	return ys

def slide_ellipsis(xs):
	ys = []
	_, lx = first(xs, lambda x: x is not ...)
	for x in xs:
		if x is ...:
			ys.append(lx)
		else:
			ys.append(x)
			lx = x
	return ys

def lerp_ellipsis(xs):
	ys = []
	left, lx = None, None
	right, rx = first(xs, lambda x: x is not ...)
	for i, x in enumerate(xs):
		if x is ...:
			if lx is None:
				if rx is None:
					raise ValueError()
				ys.append(rx)
			elif rx is None:
				if lx is None:
					raise ValueError()
				ys.append(lx)
			else:
				ys.append(lerp(lx, rx, ilerp(left, right, i)))
		else:
			ys.append(x)
			left, lx = i, x
			right, (_, rx) = first(enumerate(xs), lambda ix: ix[0] > i and ix[1] is not ..., default=(-1, None))
	return ys

def main(args):
	config_valid()
	print("Loading a model...")
	generator = Generator.load(args.generator)
	generator.to_device(args.device)
	ws = []
	for s in args.style:
		if (s == "..."):
			ws.append(...)
		else:
			ws.append(to_variable(load(s), device=args.device))
	if all(w is ... for w in ws):
		eprint("You must supply at least one style file!")
		raise RuntimeError("Input error")
	if (len(ws) > generator.levels):
		eprint("Too many styles!")
		raise RuntimeError("Input error")
	if (len(ws) != generator.levels):
		if args.justify:
			ws = justify(ws, generator.levels, align_end=args.lerp)
		else:
			ws += [...] * (generator.levels - len(ws))
	styles = []
	count = 0
	for w in ws:
		if w is ...:
			styles.append("...")
		else:
			count += 1
			styles.append(f"$style{count}")
	print("mix(" + ", ".join(styles) + ")")
	if args.lerp:
		ws = lerp_ellipsis(ws)
	else:
		ws = slide_ellipsis(ws)
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
	parser.add_argument("-j", "--justify", action="store_true", help="justify input style arguments evenly automatically")
	parser.add_argument("-l", "--lerp", action="store_true", help="use linear interpolation to resolve '...' instead of repeating")
	parser.add_argument("-n", "--number", metavar="N", type=uint, default=3, help="the number of images to generate")
	return parser.add_evaluation_args(default_batch=1).parse_args()


if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(130)
