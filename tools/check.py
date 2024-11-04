#!/usr/bin/env python3

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.resolve()))
import cure
cure.patch()

from chainer import print_runtime_info

def main():
	print_runtime_info()


if __name__ == "__main__":
	main()
