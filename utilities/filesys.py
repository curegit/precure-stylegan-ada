import os
import os.path
import inspect

def mkdirs(dirpath):
	os.makedirs(dirpath, exist_ok=True)

def filepath(dirpath, filename, fileext):
	return os.path.normpath(os.path.join(dirpath, filename) + os.extsep + fileext)

def altfilepath(filepath, suffix="+"):
	while os.path.lexists(filepath):
		root, ext = os.path.splitext(filepath)
		head, tail = os.path.split(root)
		path = os.path.join(head, tail + suffix) + ext
	return path
