import torch
import numpy as np
import sys
from utils.start_iteration import start_train
import argparse
import pprint as pp
from envs import env_dict


# ===================================
#           Initializations
# ===================================

torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
np.random.seed(np_seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

# ======================================
#           Set Hyperparameters
# ======================================
parser = argparse.ArgumentParser(description='Argument parser')

# The environment. Current options: BottleneckCylinderEnv, TwoDiagCylinderEnv, QuadcopterEnv
parser.add_argument('--env_name',                   default='BottleneckCylinderEnv', help='The environment.')

parser.add_argument('--torch_seed',            default=torch_seed)
parser.add_argument('--np_seed',               default=np_seed)
parser.add_argument('--max_epochs',            default=int(5e5))
parser.add_argument('--device',                default=device)
parser.add_argument('--print_rate',            default=1000, help='How often to print to console and log')

parser.add_argument('--batch_size',            default=50)
parser.add_argument('--ns',                    default=100, help='Network size')
parser.add_argument('--disc_lr',               default=4e-4)
parser.add_argument('--gen_lr',                default=1e-4)
parser.add_argument('--betas',                 default=(0.5, 0.9), help='Adam only')
parser.add_argument('--weight_decay',          default=1e-4)
parser.add_argument('--act_func_disc',         default='tanh', help='Activation function for discriminator')
parser.add_argument('--act_func_gen',          default='relu', help='Activation function for generator')
parser.add_argument('--gen_every_disc',        default=1, help='How many discriminator updates before one generator update')
parser.add_argument('--hh',                    default=0.5, help='ResNet step-size')
parser.add_argument('--lam_hjb_error',         default=1, help='L2 hjb error strength')
parser.add_argument('--grad_norm_clip_value',  default=np.inf, help='Gradient clipping for the discriminator and generator')

bool_logging = True
parser.add_argument('--do_logging',            default=bool_logging)
parser.add_argument('--show_plots',            default=False, help='Whether to show plots')

# Parse args, get the environment, and set some arguments automatically
args = vars(parser.parse_args())
env = env_dict[args['env_name']](device=device)
args['env'] = env
args['experiment_name'] = '_' + str(env.name)

# ==================================
#           Start training
# ==================================
pp.pprint(env.info_dict)
pp.pprint(args)
the_logger = start_train(args)
