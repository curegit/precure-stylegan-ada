import sys
import warnings

patched = False

def patch():
	global patched
	if not patched:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
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
	try:
		np.int
	except Exception:
		np.int = int
	try:
		np.float
	except Exception:
		np.float = float
	try:
		np.complex
	except Exception:
		np.complex = complex
	try:
		np.object
	except Exception:
		np.object = object
	try:
		np.str
	except Exception:
		np.str = str
	try:
		np.long
	except Exception:
		np.long = int
	try:
		np.unicode
	except Exception:
		np.unicode = str
