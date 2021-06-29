from math import log
from numpy import ones, sqrt, where, float32
from numpy.fft import fftfreq
from numpy.random import uniform, lognormal
from chainer.functions import fft, ifft

class Filtering():

	def __init__(self, sigma=1.0, probability_multiplier=1.0):
		self.sigma = sigma * log(2)
		self.probability_multiplier = probability_multiplier
		self.band1_probability = 1.0
		self.band2_probability = 1.0
		self.band3_probability = 1.0
		self.band4_probability = 1.0

	def __call__(self, x, p):
		probability = self.probability_multiplier * p
		if probability <= 0:
			return x
		batch, channels, height, width = x.shape
		fw = abs(fftfreq(width, 1 / width))
		fh = abs(fftfreq(height, 1 / height))
		ow = ones((batch, width), dtype=float32)
		oh = ones((batch, height), dtype=float32)
		aw = ones((batch, width), dtype=float32)
		ah = ones((batch, height), dtype=float32)
		if self.band1_probability > 0:
			condition = uniform(size=batch) < probability * self.band1_probability
			amplification = sqrt(lognormal(sigma=self.sigma, size=batch))
			a = where(condition, amplification, 1.0).reshape(batch, 1)
			bw = (0 < fw) & (fw <= width / 2 / 8)
			bh = (0 < fh) & (fh <= height / 2 / 8)
			aw *= where(bw, ow * a, ow) / ((10 * a + 1 + 1 + 1) / 13)
			ah *= where(bh, oh * a, oh) / ((10 * a + 1 + 1 + 1) / 13)
		if self.band2_probability > 0:
			condition = uniform(size=batch) < probability * self.band2_probability
			amplification = sqrt(lognormal(sigma=self.sigma, size=batch))
			a = where(condition, amplification, 1.0).reshape(batch, 1)
			bw = (width / 2 / 8 < fw) & (fw <= width / 2 / 4)
			bh = (height / 2 / 8 < fh) & (fh <= height / 2 / 4)
			aw *= where(bw, ow * a, ow) / ((10 + a + 1 + 1) / 13)
			ah *= where(bh, oh * a, oh) / ((10 + a + 1 + 1) / 13)
		if self.band3_probability > 0:
			condition = uniform(size=batch) < probability * self.band3_probability
			amplification = sqrt(lognormal(sigma=self.sigma, size=batch))
			a = where(condition, amplification, 1.0).reshape(batch, 1)
			bw = (width / 2 / 4 < fw) & (fw <= width / 2 / 2)
			bh = (height / 2 / 4 < fh) & (fh <= height / 2 / 2)
			aw *= where(bw, ow * a, ow) / ((10 + 1 + a + 1) / 13)
			ah *= where(bh, oh * a, oh) / ((10 + 1 + a + 1) / 13)
		if self.band4_probability > 0:
			condition = uniform(size=batch) < probability * self.band4_probability
			amplification = sqrt(lognormal(sigma=self.sigma, size=batch))
			a = where(condition, amplification, 1.0).reshape(batch, 1)
			bw = (width / 2 / 2 < fw) & (fw <= width / 2)
			bh = (height / 2 / 2 < fh) & (fh <= height / 2)
			aw *= where(bw, ow * a, ow) / ((10 + 1 + 1 + a) / 13)
			ah *= where(bh, oh * a, oh) / ((10 + 1 + 1 + a) / 13)
		aw[:, 0] = 1.0
		ah[:, 0] = 1.0
		sw = x.xp.asarray(aw).reshape(batch, 1, 1, width)
		sh = x.xp.asarray(ah).reshape(batch, 1, 1, height)
		zeros = x.xp.zeros((batch, channels, height, width), dtype=x.xp.float32)
		r1, i1 = fft((x, zeros))
		r2, i2 = r1 * sw, i1 * sw
		r3, i3 = r2.transpose(0, 1, 3, 2), i2.transpose(0, 1, 3, 2)
		r4, i4 = fft((r3, i3))
		r5, i5 = r4 * sh, i4 * sh
		r6, i6 = ifft((r5, i5))
		r7, i7 = r6.transpose(0, 1, 3, 2), i6.transpose(0, 1, 3, 2)
		return ifft((r7, i7))[0]
