from pydot import graph_from_dot_data
from chainer import global_config
from chainer.computational_graph import build_computational_graph
from stylegan.networks import Generator, Discriminator
from interface.args import CustomArgumentParser
from utilities.filesys import mkdirs, build_filepath

global_config.train = False
global_config.autotune = True
global_config.cudnn_deterministic = True

parser = CustomArgumentParser("")
parser.add_output_args("graphs").add_model_args().add_evaluation_args()
args = parser.parse_args()

generator = Generator(args.size, args.depth, args.levels, *args.channels)
discriminator = Discriminator(args.levels, args.channels[1], args.channels[0])

z = generator.generate_latents(args.batch)
#mix = gen.generate_latent(args.batch)
ws, i = generator(z)

y = discriminator(i)

gen_varstyle = {"fillcolor": "#5edbf1", "shape": "record", "style": "filled"}
gen_funstyle = {"fillcolor": "#ffa9e0", "shape": "record", "style": "filled"}
dis_varstyle = {"fillcolor": "#7a9fe6", "shape": "record", "style": "filled"}
dis_funstyle = {"fillcolor": "#fea21d", "shape": "record", "style": "filled"}

gen_graph = build_computational_graph([i], variable_style=gen_varstyle, function_style=gen_funstyle).dump()

i.unchain_backward()

dis_graph = build_computational_graph([y], variable_style=dis_varstyle, function_style=dis_funstyle).dump()

mkdirs(args.dest)
gen_path = build_filepath(args.dest, "generator", "pdf", args.force)
graph_from_dot_data(gen_graph)[0].write_pdf(gen_path)
print(f"Saved: {gen_path}")
dis_path = build_filepath(args.dest, "discriminator", "pdf", args.force)
graph_from_dot_data(dis_graph)[0].write_pdf(dis_path)
print(f"Saved: {dis_path}")
