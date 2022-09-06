from sys import float_info
from chainer.backend import CpuDevice, GpuDevice

def uint(string):
	value = int(string)
	if value >= 0:
		return value
	raise ValueError()

def natural(string):
	value = int(string)
	if value > 0:
		return value
	raise ValueError()

def ufloat(string):
	value = float(string)
	if value >= 0:
		return value
	raise ValueError()

def positive(string):
	value = float(string)
	if value >= float_info.epsilon:
		return value
	raise ValueError()

def rate(string):
	value = float(string)
	if 0 <= value <= 1:
		return value
	raise ValueError()

def device(string):
	value = string.upper()
	if value == "CPU" or value == "@NUMPY":
		return CpuDevice()
	if value == "GPU" or value == "@CUPY":
		return GpuDevice.from_device_id(0)
	value = int(string)
	if value >= 0:
		return GpuDevice.from_device_id(value)
	if value == -1:
		return CpuDevice()
	raise ValueError()
