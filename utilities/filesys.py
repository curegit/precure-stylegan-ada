import os
import os.path
import glob
from utilities.iter import dict_groupby

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

def glob_recursively(dirpath, fileext, robust_letter_case=False):
	pattern = build_filepath(glob.escape(dirpath), os.path.join("**", "*"), glob.escape(fileext))
	ls = [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]
	if robust_letter_case:
		ls_dict = dict_groupby(ls, lambda f: os.path.basename(f).lower())
		exts = {fileext.lower(), fileext.capitalize(), fileext.upper()} - {fileext}
		for e in exts:
			pattern = build_filepath(glob.escape(dirpath), os.path.join("**", "*"), glob.escape(e))
			new = []
			for f in glob.glob(pattern, recursive=True):
				if os.path.isfile(f):
					name = os.path.basename(f).lower()
					if name in ls_dict:
						for l in ls_dict[name]:
							if os.path.samefile(f, l):
								break
						else:
							new.append(f)
					else:
						new.append(f)
			ls += new
	return ls
