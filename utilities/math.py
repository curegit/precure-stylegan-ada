import math

def identity(x):
	return x

def sgn(x):
	if x == 0:
		return 0.0
	else:
		return math.copysign(1.0, x)

def clamp(a, x, b):
	return min(max(x, a), b)

def lerp(a, b, t):
	return a + t * (b - a)
