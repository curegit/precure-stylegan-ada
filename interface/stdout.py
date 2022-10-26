from tqdm import tqdm

bar_format = "{desc} [{bar}] {percentage:5.1f}%"

def chainer_like_tqdm(desc, total):
	return tqdm(desc=desc, total=total, bar_format=bar_format, miniters=1, ascii=".#", ncols=70)

def print_model_args(generator):
	print(f"Multilayer perceptron: {generator.size}x{generator.depth}")
	print(f"CNN layers: {generator.levels} levels (output = {generator.width}x{generator.height})")
	print(f"CNN channels: {generator.first_channels} (initial) -> {generator.last_channels} (final)")

def print_data_classes(generator):
	print(f"Data classes: {generator.categories if generator.conditional else '1 (unconditional)'}")
	if generator.conditional:
		for i, l in enumerate(generator.labels):
			print(f"- class {i}: {l}")

def print_parameter_counts(generator, discriminator=None):
	if discriminator is None:
		print(f"Parameter counts: {generator.count_params()}")
	else:
		print("Parameter counts:")
		print(f"- G: {generator.count_params()}")
		print(f"- D: {discriminator.count_params()}")

def print_cnn_architecture(generator, discriminator=None, transfer=None, freeze=None):
	if discriminator is None:
		print("CNN channels:")
	else:
		print("Generator CNN channels:")
	pad = max(max(len(str(s)) for s in s.channels) for i, s in generator.synthesizer.blocks)
	for i, s in generator.synthesizer.blocks:
		if transfer is not None and i >= transfer[0]:
			if freeze is not None and i >= freeze[0]:
				info = "(transferred and frozen)"
			else:
				info = "(transferred)"
		elif freeze is not None and i >= freeze[0]:
			info = "(frozen)"
		else:
			info = ""
		print(f"- Level {i}: " + " -> conv -> ".join(str(c).rjust(pad) for c in s.channels) + f" {info}")
	if discriminator is not None:
		print("Discriminator CNN channels:")
		pad = max(max(len(str(b)) for b in b.channels) for i, b in discriminator.blocks if i > 0)
		for i, b in discriminator.blocks:
			if i > 0:
				if transfer is not None and i <= transfer[1]:
					if freeze is not None and i <= freeze[1]:
						info = "(transferred and frozen)"
					else:
						info = "(transferred)"
				elif freeze is not None and i <= freeze[1]:
					info = "(frozen)"
				else:
					info = ""
				print(f"- Level {i}: " + " -> conv -> ".join(str(b).rjust(pad) for b in b.channels) + f" {info}")

def print_training_args(args):
	if args.accum is None:
		print(f"Batch size: {args.batch} (Group size: {'entire batch' if args.group == 0 else args.group})")
	else:
		print(f"Accum/batch size: {args.accum}/{args.batch} (Group size: {'entire accum/batch' if args.group == 0 else args.group})")
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
