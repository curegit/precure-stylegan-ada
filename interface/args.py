from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from interface.argtypes import uint, natural, ufloat, positive, rate, device

class CustomArgumentParser(ArgumentParser):

	def __init__(self, description):
		super().__init__(allow_abbrev=False, description=description, formatter_class=ArgumentDefaultsHelpFormatter)

	def add_output_args(self, default_dest):
		group = self.add_argument_group("output", "")
		group.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
		group.add_argument("-d", "--result", "--dest", metavar="DIR", dest="dest", default=default_dest, help="")
		return self

	def add_model_args(self):
		group = self.add_argument_group("model", "")
		group.add_argument("-m", "--depth", metavar="", type=natural, default=8, help="")
		group.add_argument("-z", "--size", metavar="", type=natural, default=512, help="")
		group.add_argument("-x", "--levels", metavar="", type=natural, default=7, help="")
		group.add_argument("-c", "--channels", metavar="", type=natural, nargs=2, default=(512, 16), help="")
		group.add_argument("-N", "--narrow", action="store_true", help="")
		return self

	def add_evaluation_args(self):
		group = self.add_argument_group("evaluation", "")
		group.add_argument("-b", "--batch", type=natural, default=32, help="")
		group.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=device("CPU"), help="")
		return self

	def add_generation_args(self):
		self.add_argument("-n", "--number", type=uint, default=10, help="the number of middle images to generate")
		self.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained model to load")
		self.add_argument("-l", "--latent", "--center", metavar="FILE", dest="center", help="")
		self.add_argument("-e", "--deviation", "--sd", metavar="SIGMA", dest="sd", type=positive, default=1.0, help="")
		self.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=ufloat, help="")
		return self
