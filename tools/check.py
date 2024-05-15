#!/usr/bin/env python3

import cure
cure.patch()

from chainer import print_runtime_info

def main():
	print_runtime_info()


if __name__ == "__main__":
	main()
