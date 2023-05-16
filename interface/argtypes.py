from sys import float_info
from math import isfinite
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

def real(string):
	value = float(string)
	if isfinite(value):
		return value
	raise ValueError()

def ufloat(string):
	value = real(string)
	if value >= 0:
		return value
	raise ValueError()

def positive(string):
	value = real(string)
	if value >= float_info.epsilon:
		return value
	raise ValueError()

def rate(string):
	value = real(string)
	if 0 <= value <= 1:
		return value
	raise ValueError()

def device(string):
	value = string.upper()
	if value == "CPU" or value == "@NUMPY":
		return CpuDevice()
	if value[:6] == "@CUPY:":
		return GpuDevice.from_device_id(int(value[6:]))
	if value == "GPU" or value == "@CUPY":
		return GpuDevice.from_device_id(0)
	try:
		value = int(string)
	except:
		raise ValueError() from None
	if value >= 0:
		return GpuDevice.from_device_id(value)
	if value == -1:
		return CpuDevice()
	raise ValueError()
