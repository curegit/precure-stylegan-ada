import sys

patched = False

def patch():
	global patched
	if not patched:
		patch_distutils()
		patch_sctypes()
		patch_type_names()
	patched = True

def patch_distutils():
	import numpy
	try:
		import numpy.distutils
	except Exception:
		import cure.numpy.distutils
		numpy.distutils = cure.numpy.distutils
		sys.modules["numpy.distutils"] = numpy.distutils
		import numpy.distutils

def patch_sctypes():
	import numpy as np
	try:
		np.sctypes
	except Exception:
		np.sctypes = {
			"int": [np.int8, np.int16, np.int32, np.int64],
			"uint": [np.uint8, np.uint16, np.uint32, np.uint64],
			"float": [np.float16, np.float32, np.float64],
			"complex": [np.complex64, np.complex128],
			"others": [bool, object, bytes, str, np.void],
		}

def patch_type_names():
	import numpy as np
	try:
		np.bool
	except Exception:
		np.bool = bool
		np.int = int
		np.float = float
		np.complex = complex
		np.object = object
		np.str = str
		np.long = int
		np.unicode = str
