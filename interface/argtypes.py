from sys import float_info

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
	if value == "CPU":
		return -1
	if value == "GPU":
		return 0
	value = int(string)
	if value >= -1:
		return value
	raise ValueError()
