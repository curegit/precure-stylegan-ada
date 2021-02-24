import os
import os.path
import glob

def mkdirs(dirpath):
	os.makedirs(os.path.normpath(dirpath), exist_ok=True)

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
	pattern = build_filepath(glob.escape(dirpath), os.path.join("**", "*"), glob.escape(fileext))
	return [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]
