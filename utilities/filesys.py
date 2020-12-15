import os
import os.path
import glob

def mkdirs(dirpath):
	os.makedirs(dirpath, exist_ok=True)

def alt_filepath(filepath, suffix="+"):
	while os.path.lexists(filepath):
		root, ext = os.path.splitext(filepath)
		head, tail = os.path.split(root)
		filepath = os.path.join(head, tail + suffix) + ext
	return filepath

def build_filepath(dirpath, filename, fileext, exist_ok=True, suffix="+"):
	filepath = os.path.normpath(os.path.join(dirpath, filename) + os.extsep + fileext)
	return filepath if exist_ok else alt_filepath(filepath, suffix)

def glob_recursively(dirpath, fileext):
	return [f for f in glob.glob(build_filepath(dirpath, f"**{os.sep}*", fileext), recursive=True) if os.path.isfile(f)]
