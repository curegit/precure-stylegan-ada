#!/usr/bin/env python3

import numpy as np
from chainer import global_config, Variable
from chainer.functions import stack
from stylegan.networks import Generator
from interface.args import CustomArgumentParser
from interface.argtypes import uint, natural
from interface.stdout import chainer_like_tqdm
from utilities.image import to_pil_image, save_image
from utilities.stdio import eprint
from utilities.filesys import mkdirs, build_filepath
from utilities.iter import range_batch, iter_batch

def interpolate(ws, middles=15, loop=True, closed=True):
	n = len(ws)
	w_pairs = [(ws[i], ws[(i + 1) % n]) for i in range(n if loop else n - 1)]
	for w1, w2 in w_pairs:
		yield w1
		for i in range(1, middles + 1):
			yield w1 + (i / (middles + 1)) * (w2 - w1)
	if closed and not loop:
		yield ws[n - 1]

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
	if args.prepend is not None:
		print("Loading prepended styles...")
		prepend = [Variable(np.load(s)) for s in args.prepend]
		print(f"Prepended styles: {len(prepend)}")
	else:
		prepend = []
	if args.append is not None:
		print("Loading appended styles...")
		append = [Variable(np.load(s)) for s in args.append]
		print(f"Appended styles: {len(append)}")
	else:
		append = []
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
		mean_w = generator.calculate_mean_w(categories=(None if args.global_truncation else categories))
	else:
		mean_w = None
	mkdirs(args.dest)
	sampled_ws = []
	if args.number > 0:
		with chainer_like_tqdm(desc="generation", total=args.number) as bar:
			for i, n in range_batch(args.number, args.batch):
				z = generator.generate_latents(n, center=center, sd=args.sd)
				c = generator.generate_conditions(n, categories=categories) if generator.conditional else None
				w = generator.truncation_trick(generator.mapper(z, c if c is None else generator.embedder(c)), args.psi, mean_w)
				sampled_ws += [w[i] for i in range(n)]
				if not args.no_samples:
					y = generator.synthesizer([w] * generator.levels, noise=args.noisy, freeze=args.freeze)
					z.to_cpu()
					w.to_cpu()
					y.to_cpu()
					for j in range(n):
						filename = f"{i + j + 1}"
						np.save(build_filepath(args.dest, filename + "-latent", "npy", args.force), z.array[j])
						np.save(build_filepath(args.dest, filename + "-style", "npy", args.force), w.array[j])
						save_image(y.array[j], build_filepath(args.dest, filename, "png", args.force))
						bar.update()
	ws = prepend + sampled_ws + append
	frame_ws = list(interpolate(ws, args.interpolate, args.loop))
	count = 0
	images = []
	with chainer_like_tqdm(desc="frames", total=len(frame_ws)) as bar:
		for ws in iter_batch(frame_ws, args.batch):
			y = generator.synthesizer([stack(list(ws))] * generator.levels, noise=args.noisy, freeze=args.freeze)
			y.to_cpu()
			for i in range(y.shape[0]):
				image = to_pil_image(y.array[i])
				images.append(image)
				if args.frames:
					filename = f"frame-{count + 1}"
					image.save(build_filepath(args.dest, filename, "png", args.force))
				bar.update()
				count += 1
	for ext in ["png"] + (["webp"] if args.webp else []) + (["gif"] if args.gif else []):
		filepath = build_filepath(args.dest, "analogy", ext, args.force)
		if args.repeat:
			images[0].save(filepath, save_all=True, duration=args.duration, append_images=images[1:], loop=0)
		else:
			images[0].save(filepath, save_all=True, duration=args.duration, append_images=images[1:])

def check_args(args):
	if len(args.prepend or []) + args.number + len(args.append or []) < 2:
		eprint("More styles required!")
		raise RuntimeError("Input error")
	return args

def parse_args():
	parser = CustomArgumentParser("Create style-interpolating animation")
	parser.require_generator().add_output_args("animation").add_generation_args(allow_zero=True)
	group = parser.add_argument_group("animation arguments")
	group.add_argument("-J", "--no-samples", action="store_true", help="don't save key images")
	group.add_argument("-G", "--gif", action="store_true", help="output an additional GIF animation file")
	group.add_argument("-W", "--webp", action="store_true", help="output an additional WebP animation file")
	group.add_argument("-F", "--frames", action="store_true", help="output all frames as isolated images additionally")
	group.add_argument("-L", "--loop", action="store_true", help="interpolate between the last and first images")
	group.add_argument("-R", "--repeat", action="store_true", help="tell the image writer explicitly that an output image should loop")
	group.add_argument("-D", "--duration", metavar="MS", type=natural, default=100, help="the display duration of each frame in milliseconds")
	group.add_argument("-I", "--interpolate", metavar="N", type=uint, default=15, help="the number of frames between key images")
	group.add_argument("-P", "--prepend", metavar="STYLE_FILE", nargs="+", action="extend", help="add specified key images (by style NPY file) to the head")
	group.add_argument("-A", "--append", metavar="STYLE_FILE", nargs="+", action="extend", help="add specified key images (by style NPY file) to the tail")
	return parser.add_evaluation_args().parse_args()

if __name__ == "__main__":
	try:
		main(check_args(parse_args()))
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
