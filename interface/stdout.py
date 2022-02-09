from tqdm import tqdm

bar_format = "{desc} [{bar}] {percentage:5.1f}%"

def chainer_like_tqdm(desc, total):
	return tqdm(desc=desc, total=total, bar_format=bar_format, miniters=1, ascii=".#", ncols=70)

def print_model_args(args, generator):
	h, w = generator.resolution
	print(f"Multilayer perceptron: {args.size}x{args.depth}")
	print(f"CNN layers: {args.levels} levels (output = {w}x{h})")
	print(f"CNN channels: {args.channels[0]} (initial) -> {args.channels[1]} (final)")

def print_training_args(args):
	if args.accum is None:
		print(f"Batch size: {args.batch} (Group size: {args.batch if args.group == 0 else args.group})")
	else:
		print(f"Batch size: {args.accum}/{args.batch} (Group size: {args.accum if args.group == 0 else args.group})")
	print(f"Style-mixing rate: {args.mix * 100}%")
	if args.gamma > 0 and args.r1 > 1:
		print(f"R1 regularization: coefficient = {args.gamma} (every {args.r1} iterations)")
	elif args.gamma > 0:
		print(f"R1 regularization: coefficient = {args.gamma} (every iteration)")
	else:
		print("R1 regularization: disabled")
	if args.weight > 0 and args.pl > 1:
		print(f"Path length regularization: coefficient = {args.weight}, decay = {args.decay} (every {args.pl} iterations)")
	elif args.weight > 0:
		print(f"Path length regularization: coefficient = {args.weight}, decay = {args.decay} (every iteration)")
	else:
		print(f"Path length regularization: disabled")
	print(f"Objective: {'least squares loss' if args.lsgan else 'logistic loss'}")
	print(f"Adam: alpha = {args.alpha}, beta1 = {args.betas[0]}, beta2 = {args.betas[1]}")
