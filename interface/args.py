from json import dump
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from interface.argtypes import uint, natural, ufloat, positive, rate, device

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
		group = self.add_argument_group("output")
		group.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
		group.add_argument("-d", "--result", "--dest", metavar="DIR", dest="dest", default=default_dest, help="")
		return self

	def add_model_args(self):
		group = self.add_argument_group("model")
		group.add_argument("-m", "--depth", metavar="", type=natural, default=8, help="")
		group.add_argument("-z", "--size", metavar="", type=natural, default=512, help="")
		group.add_argument("-x", "--levels", metavar="", type=natural, default=7, help="")
		group.add_argument("-c", "--channels", metavar="", type=natural, nargs=2, default=(512, 16), help="")
		group.add_argument("-K", "--labels", metavar="CLASS", nargs="+", help="")
		return self

	def add_evaluation_args(self, include_batch=True):
		group = self.add_argument_group("evaluation")
		if include_batch: group.add_argument("-b", "--batch", type=natural, default=16, help="")
		group.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=device("CPU"), help="")
		return self

	def add_generation_args(self):
		group = self.add_argument_group("generation")
		group.add_argument("-n", "--number", type=uint, default=10, help="the number of middle images to generate")
		group.add_argument("-E", "--label", metavar="L", dest="label", help="")
		group.add_argument("-l", "--latent", "--center", metavar="FILE", dest="center", help="")
		group.add_argument("-e", "--deviation", "--sd", metavar="SIGMA", dest="sd", type=positive, default=1.0, help="")
		group.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=ufloat, default=1.0, help="")
		return self
