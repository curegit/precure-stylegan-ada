import itertools

def range_batch(stop, batch=1):
	for i in range(0, stop, batch):
		yield i, min(batch, stop - i)

def iter_batch(iterable, batch=1):
	iterator = iter(iterable)
	for i in iterator:
		yield itertools.chain([i], itertools.islice(iterator, batch - 1))
