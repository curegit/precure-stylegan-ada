from chainer import global_config, Variable

def to_variable(*args, device=None, **kwargs):
	v = Variable(*args, **kwargs)
	v.to_device(device)
	return v

def config_train():
	global_config.train = True
	global_config.autotune = True
	global_config.cudnn_deterministic = False

def config_valid(faster=False):
	global_config.train = False
	global_config.autotune = True
	global_config.cudnn_deterministic = not faster
	global_config.enable_backprop = False
	global_config.type_check = not faster
