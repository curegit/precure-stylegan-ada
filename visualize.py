#!/usr/bin/env python3

import cure
cure.patch()

from sys import exit
from pydot import graph_from_dot_data
from chainer.computational_graph import build_computational_graph
from stylegan.networks import Generator, Discriminator
from stylegan.training import CustomUpdater
from interface.args import CustomArgumentParser
from interface.argtypes import natural
from interface.stdout import print_model_args, print_parameter_counts, print_cnn_architecture
from utilities.stdio import eprint
from utilities.filesys import mkdirs, build_filepath
from utilities.chainer import config_train

gen_varstyle = {"fillcolor": "#5edbf1", "shape": "record", "style": "filled"}
gen_funstyle = {"fillcolor": "#ffa9e0", "shape": "record", "style": "filled"}
dis_varstyle = {"fillcolor": "#7a9fe6", "shape": "record", "style": "filled"}
dis_funstyle = {"fillcolor": "#fea21d", "shape": "record", "style": "filled"}

def main(args):
	config_train()
	print("Initializing models...")
	if args.snapshot is not None:
		print("Reconstructing networks from a snapshot...")
		generator, discriminator = CustomUpdater.reconstruct_models(args.snapshot, load_weights=False)
		categories = generator.categories
	else:
		categories = args.categories or 1
		generator = Generator(args.size, args.depth, args.levels, *args.channels, categories)
		discriminator = Discriminator(args.levels, args.channels[1], args.channels[0], categories, args.depth)
	generator.to_device(args.device)
	discriminator.to_device(args.device)
	print_model_args(generator)
	print_parameter_counts(generator, discriminator)
	print_cnn_architecture(generator, discriminator)
	print("Exporting graphs...")
	z = generator.generate_latents(args.batch)
	c = generator.generate_conditions(args.batch)[1] if categories > 1 else None
	_, x = generator(z, c)
	y = discriminator(x, c)
	gen_graph = build_computational_graph([x], variable_style=gen_varstyle, function_style=gen_funstyle).dump()
	x.unchain_backward()
	dis_graph = build_computational_graph([y], variable_style=dis_varstyle, function_style=dis_funstyle).dump()
	mkdirs(args.dest)
	gen_path = build_filepath(args.dest, "generator", "pdf", args.force)
	graph_from_dot_data(gen_graph)[0].write_pdf(gen_path)
	print(f"Saved: {gen_path}")
	dis_path = build_filepath(args.dest, "discriminator", "pdf", args.force)
	graph_from_dot_data(dis_graph)[0].write_pdf(dis_path)
	print(f"Saved: {dis_path}")

def parse_args():
	parser = CustomArgumentParser("Draw computational graphs of generator and discriminator in PDFs")
	ex_group = parser.add_mutually_exclusive_group()
	ex_group.add_argument("-s", "--snapshot", metavar="HDF5_FILE", help="import network architecture from a training snapshot")
	ex_group.add_argument("-n", "--class", type=natural, metavar="N", dest="categories", help="specify the number of data classes")
	parser.add_output_args("graphs").add_model_args().add_evaluation_args(include_noise=False, default_batch=4)
	return parser.parse_args()


if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(130)
