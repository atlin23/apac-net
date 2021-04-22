import torch
from os.path import join
import pickle
from neural_net_defs import DiscNet, GenNet
from utils.utils import act_funcs, Plotter, Logger
import numpy as np
import matplotlib.pyplot as plt
import time
from envs import env_dict
import imageio
import argparse


# =============================================================
#           Load experiment and environment arguments
# =============================================================
def load_args(experiment_path):
    experiment_args_pickle = join(experiment_path, 'args', 'experiment_args.pkl')
    args = pickle.load(open(experiment_args_pickle, 'rb'))

    environment_args_pickle = join(experiment_path, 'args', 'env_args.pkl')
    env_args = pickle.load(open(environment_args_pickle, 'rb'))

    return args, env_args

# ======================================
#           Plotter for testing
# ======================================
class TestPlotter(Plotter):
    """
    This is like Plotter, but adds animations and a way to generate new rho0_saved
    """
    def __init__(self, args, the_logger,
                 num_plot_samples=100, linspace_size=201, timesteps=10):
        super(TestPlotter, self).__init__(args, the_logger,
                                          num_plot_samples, linspace_size, timesteps)

        self.anim_timesteps = 60

    def make_new_rho0(self):
        self.rho0_saved = self.env.sample_rho0(self.num_plot_samples).to(args['device'])

    def _draw_one_timestep(self, generator, single_timestep):
        with torch.no_grad():
            # Make plot fig and ax
            if self.plot_dim == 2:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
            elif self.plot_dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

            ax.contour(self.X, self.Y, self.Z, levels=[0], colors='gray', zorder=1)

            the_timepoint = torch.tensor([single_timestep],
                                         dtype=torch.float).to(self.device)
            plot_sample = generator(the_timepoint[0].expand(self.num_plot_samples, 1),
                                    self.rho0_saved).cpu().detach().numpy()
            alpha = 0.75
            color = 'C0'
            if self.plot_dim == 2:
                ax.scatter(plot_sample[:, 0], plot_sample[:, 1], color=color, alpha=alpha, zorder=2)
            elif self.plot_dim == 3:
                ax.scatter(plot_sample[:, 0], plot_sample[:, 1], plot_sample[:, 2], color=color, alpha=alpha, zorder=2)

            ax.set_xlim([-self.plot_window_size, self.plot_window_size])
            ax.set_ylim([-self.plot_window_size, self.plot_window_size])
            if self.plot_dim == 3:
                ax.set_zlim([-self.plot_window_size, self.plot_window_size])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)

            return image

    def make_animation(self, generator, timesteps=None):
        with torch.no_grad():
            if timesteps == None:
                timesteps = self.timesteps

            kwargs_write = {'fps': 30}
            imageio.mimsave(join(self.the_logger.experiment_dir, './movie.gif'),
                            [self._draw_one_timestep(generator, (self.TT / self.anim_timesteps) * time_idx)
                             for time_idx in range(self.anim_timesteps)],
                            **kwargs_write)



# ==============================
#           Start test
# ==============================
def start_test(experiment_path, epoch, num_plot_samples, timesteps, plot_window_size, device):

    assert(type(experiment_path) == str)

    # Load the training and environment arguments, and environment
    args, env_args = load_args(experiment_path)
    env = env_dict[env_args['env_name']](device=device)

    # Set the parameters from args
    act_func_disc = act_funcs[args['act_func_disc']]
    act_func_gen = act_funcs[args['act_func_gen']]

    # Parameters for the test
    args['device'] = device
    args['do_logging'] = True
    args['experiment_name'] = 'TEST_' + str(env.name)
    args['show_plots'] = True
    args['original_experiment_path'] = experiment_path  # save the experiment path which is being tested

    # Load the logger and the plotter
    the_logger = Logger(args)
    the_plotter = TestPlotter(args, the_logger,
                              num_plot_samples=num_plot_samples, linspace_size=201, timesteps=timesteps)

    # Set plotter settings
    if plot_window_size is 'args':
        the_plotter.plot_window_size = env.plot_window_size
    else:
        args['plot_window'] = plot_window_size

    # Get the neural network models and load them
    discriminator = DiscNet(dim=env.dim, ns=args['ns'], act_func=act_func_disc, hh=args['hh'], device=device,
                            psi_func=env.psi_func, TT=env.TT)
    generator = GenNet(dim=env.dim, ns=args['ns'], act_func=act_func_gen, hh=args['hh'], device=device, mu=None,
                       std=None, TT=env.TT)
    disc_load_path = join(experiment_path, 'models', 'discriminator-epoch-' + str(epoch) + '.pth.tar')
    gen_load_path = join(experiment_path, 'models', 'generator-epoch-' + str(epoch) + '.pth.tar')
    disc_load = torch.load(disc_load_path, map_location=lambda storage, loc: storage)  # Puts model on CPU
    gen_load = torch.load(gen_load_path, map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(disc_load['model_state_dict'])
    generator.load_state_dict(gen_load['model_state_dict'])
    generator.mu, generator.std = gen_load['gen_mu'], gen_load['gen_std']
    discriminator.eval()
    generator.eval()

    # The iteration to generate plots
    for idx in range(0, 100):
        the_plotter.make_plots(idx, generator, the_logger)
        the_plotter.make_animation(generator, 60)
        time.sleep(3)
        the_plotter.make_new_rho0()


# ====================
#         Main
# ====================
if __name__ == '__main__':
    device = torch.device('cpu')
    experiment_path = './Experiments/Run_2021-04-17__12-10-40__BottleneckCylinderEnv'

    parser = argparse.ArgumentParser(description='Argument parser')

    parser.add_argument('--experiment_path',  default=experiment_path, help='Path to the experiment.')
    parser.add_argument('--epoch',            default=331000, help='Choose the model to run, based on epoch.')
    parser.add_argument('--num_plot_samples', default=100, help='The number of samples to plot for the 2D static plot.')
    parser.add_argument('--timesteps',        default=10, help='The number of timesteps to plot for the 2D static plot.')
    parser.add_argument('--plot_window_size', default='args', help='The plot window size. args will use the environment default.')
    parser.add_argument('--device',           default=device, help='Which device to use, cpu or gpu.')

    args = vars(parser.parse_args())

    start_test(experiment_path=experiment_path, epoch=args['epoch'], num_plot_samples=args['num_plot_samples'],
               timesteps=args['timesteps'], plot_window_size=args['plot_window_size'], device=args['device'])