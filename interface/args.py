from json import dump
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from interface.argtypes import uint, natural, ufloat, positive, device

def dump_json(args, filepath):
	with open(filepath, mode="w", encoding="utf-8") as fp:
		dump(vars(args), fp, indent=2, sort_keys=True, default=str)


class CustomArgumentParser(ArgumentParser):

	def __init__(self, description):
		super().__init__(allow_abbrev=False, description=description, formatter_class=ArgumentDefaultsHelpFormatter)

	def require_generator(self):
		self.add_argument("generator", metavar="GEN_FILE", help="HDF5 file of a serialized trained generator")
		return self

	def add_output_args(self, default_dest):
		group = self.add_argument_group("output arguments")
		group.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
		group.add_argument("-o", "--output", "--dest", "--result", metavar="DIR", dest="dest", default=default_dest, help="write output into destination directory DIR")
		return self

	def add_model_args(self):
		group = self.add_argument_group("model arguments")
		group.add_argument("-d", "--depth", metavar="N", type=natural, default=8, help="the number of layers in a multilayer perceptron (mapper module)")
		group.add_argument("-z", "--size", metavar="D", type=natural, default=512, help="the number of nodes in a dense layer (dimension of latent vector)")
		group.add_argument("-x", "--levels", metavar="N", type=natural, default=7, help="the number of stacked CNN blocks, defining output resolution as 2^(N+1)")
		group.add_argument("-c", "--channels", metavar="C", type=natural, nargs=2, default=(512, 64), help="the number of initial and final channels in CNN")
		return self

	def add_evaluation_args(self, include_batch=True, include_noise=True, default_batch=16):
		group = self.add_argument_group("evaluation arguments")
		if include_batch:
			group.add_argument("-b", "--batch", metavar="N", type=natural, default=default_batch, help="batch size, affecting speed and memory usage")
		if include_noise:
			group.add_argument("-N", "--noise", metavar="K", dest="noisy", type=ufloat, default=1.0, help="strength multiplier of random noise injections")
			group.add_argument("-z", "--fixed", metavar="N", type=uint, nargs="?", const=0, help="make noise injections deterministic by given seed N")
		group.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=device("CPU"), help="use GPU device of the specified ID (pass 'GPU' as ID to select GPU device automatically)")
		return self

	def add_generation_args(self, allow_zero=False):
		group = self.add_argument_group("generation arguments")
		group.add_argument("-n", "--number", metavar="N", type=uint if allow_zero else natural, default=10, help="the number of images to generate")
		group.add_argument("-c", "--class", metavar="NUM", type=uint, nargs="+", dest="classes", action="extend", help="specify image classes to generate by number")
		group.add_argument("-l", "--label", metavar="CLASS", nargs="+", dest="labels", action="extend", help="specify image classes to generate by label")
		group.add_argument("-s", "--similar", "--latent", "--center", metavar="LATENT_FILE", dest="center", help="move the mean of random latent vectors to a specified one")
		group.add_argument("-d", "--deviation", "--sd", metavar="SIGMA", dest="sd", type=positive, default=1.0, help="the standard deviation of latent vectors")
		group.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=ufloat, default=1.0, help="apply the truncation trick")
		ex_group = group.add_mutually_exclusive_group()
		ex_group.add_argument("-C", "--local-truncation", "--local", dest="local_truncation", action="store_true", help="truncate to the each center of data classes when the truncation trick is used with conditional models (by default, the collective center of specified data classes)")
		ex_group.add_argument("-T", "--global-truncation", "--global", dest="global_truncation", action="store_true", help="truncate to the center of all data classes when the truncation trick is used with conditional generation (by default, the collective center of specified data classes)")
		return self
