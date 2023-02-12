import numpy as np

def save(filepath, array):
	np.save(filepath, array, allow_pickle=False)

def load(filepath):
	return np.load(filepath, allow_pickle=False)
