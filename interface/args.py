from argparse import ArgumentParser
from interface.argtypes import uint, natural, ufloat, positive, rate, device

class CustomArgumentParser(ArgumentParser):

	def __init__(self, description):
		super().__init__(allow_abbrev=False, description=description)

	def add_output_args(self, default_dest="."):
		group = self.add_argument_group("output", "")
		group.add_argument("-f", "--force", action="store_true", help="allow overwrite existing files")
		group.add_argument("-r", "--result", "--dest", metavar="DIR", dest="dest", default=default_dest, help="")
		return self

	def add_model_args(self):
		group = self.add_argument_group("model", "")
		group.add_argument("-m", "--depth", metavar="", type=natural, default=8, help="")
		group.add_argument("-z", "--size", metavar="", type=natural, default=512, help="")
		group.add_argument("-x", "--levels", metavar="", type=natural, default=7, help="")
		group.add_argument("-c", "--channels", metavar="", type=natural, default=(512, 16), help="")
		group.add_argument("-w", "--double", action="store_true", help="")
		return self

	def add_evaluation_args(self):
		group = self.add_argument_group("evaluation", "")
		group.add_argument("-b", "--batch", type=natural, default=16, help="")
		group.add_argument("-v", "--device", "--gpu", metavar="ID", dest="device", type=device, default=-1, help="")
		return self

	def add_train_args(self):
		self.add_argument("-e", "--epoch", type=natural, default=1, help="")
		self.add_argument("-G", "--gamma", "--l2-batch", dest="gamma", type=ufloat, default=10, help="")
		self.add_argument("-L", "--lsgan", "--least-squares", action="store_true", help="")
		self.add_argument("-i", "--style-mixing", metavar="RATE", dest="mix", type=rate, default=0.5, help="")
		self.add_argument("-S", "--sgd", metavar="LR", type=positive, nargs=3, help="")
		self.add_argument("-A", "--adam-alphas", metavar="ALPHA", type=positive, nargs=3, default=(0.00001, 0.001, 0.001), help="Adam's coefficients of learning rates of mapper, generator, and discriminator")
		self.add_argument("-B", "--adam-betas", metavar="BETA", type=rate, nargs=2, default=(0.0, 0.99), help="Adam's exponential decay rates of the 1st and 2nd order moments")
		self.add_argument("-u", "--print-interval", metavar="ITER", dest="print", type=uint, nargs=2, default=(5, 500), help="")
		self.add_argument("-l", "--write-interval", metavar="ITER", dest="write", type=uint, nargs=4, default=(1000, 3000, 500, 500), help="")
		self.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained generator to load and retrain")
		self.add_argument("-d", "--discriminator", metavar="FILE", help="HDF5 file of serialized trained discriminator to load and retrain")
		self.add_argument("-o", "--optimizers", metavar="FILE", nargs=3, help="snapshot of optimizers of mapper, generator, and discriminator")
		self.add_argument("-p", "--preload", action="store_true", help="preload all dataset into RAM")
		return self

	def add_generation_args(self):
		self.add_argument("-n", "--number", type=uint, default=10, help="the number of middle images to generate")
		self.add_argument("-g", "--generator", metavar="FILE", help="HDF5 file of serialized trained model to load")
		self.add_argument("-l", "--latent", "--center", metavar="FILE", dest="center", help="")
		self.add_argument("-e", "--deviation", "--sd", metavar="SIGMA", dest="sd", type=positive, default=1.0, help="")
		self.add_argument("-t", "--truncation-trick", "--psi", metavar="PSI", dest="psi", type=ufloat, help="")
		return self
