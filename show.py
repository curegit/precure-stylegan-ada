#!/usr/bin/env python3

from stylegan.networks import Generator
from interface.args import CustomArgumentParser
from interface.stdout import print_model_args, print_parameter_counts, print_cnn_architecture, print_data_classes
from utilities.stdio import eprint

def main(args):
	print("Loading a model...")
	generator = Generator.load(args.generator)
	print_model_args(generator)
	print_parameter_counts(generator)
	print_cnn_architecture(generator)
	print_data_classes(generator)

def parse_args():
	parser = CustomArgumentParser("Show model arguments and data classes of a serialized generator")
	parser.require_generator()
	return parser.parse_args()

if __name__ == "__main__":
	try:
		main(parse_args())
	except KeyboardInterrupt:
		eprint("KeyboardInterrupt")
		exit(1)
