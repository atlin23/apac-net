import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import datetime
import os
from os.path import join
import pprint as pp
import pickle
import csv
import imageio


# =======================================================
#           Utility functions and miscellaneous
# =======================================================
DISC_STRING = 'DISC'
GEN_STRING = 'GEN'

# Some activation functions
act_funcs = {'tanh': lambda x: torch.tanh(x),
             'relu': lambda x: torch.relu(x),
             'leaky_relu': lambda x: torch.nn.functional.leaky_relu(x),
             'softplus': lambda x: torch.nn.functional.softplus(x)}


def sqeuc(xx):
    return torch.sum(xx * xx, dim=1, keepdim=True)


def uniform_time_sampler(batch_size):
    return torch.rand(size=(batch_size, 1))


# ==============================================
#           Plotter class to make plots
# ==============================================
class Plotter(object):

    def __init__(self, args, the_logger,
                 num_plot_samples=100, linspace_size=201, timesteps=10):
        self.env = args['env']
        self.TT = self.env.TT
        self.device = args['device']
        self.show_plots = args['show_plots']
        self.the_logger = the_logger
        self.FF_obstacle_func = self.env.FF_obstacle_func
        self.plot_window_size = self.env.plot_window_size
        self.plot_dim = self.env.plot_dim
        self.linspace_size = linspace_size
        self.num_plot_samples = num_plot_samples
        self.timesteps = timesteps
        self.X, self.Y, self.Z = self._make_obstacle_contours()

        # Saved points to plot
        self.rho0_saved = self.env.sample_rho0(self.num_plot_samples).to(args['device'])

    def _make_obstacle_contours(self):
        """
        Precomputes the obstacle contour.
        """
        self.plot_window_size = 3  # plot window size
        xlin = np.linspace(-self.plot_window_size, self.plot_window_size, self.linspace_size)
        ylin = np.linspace(-self.plot_window_size, self.plot_window_size, self.linspace_size)
        X, Y = np.meshgrid(xlin, ylin)
        vec_func = np.vectorize(self.FF_obstacle_func)
        Z = self.env.lam_obstacle * vec_func(X, Y)

        return X, Y, Z

    def _get_plot_fig(self, generator, timesteps=None):
        """
        Helper function for make_plots.
        """
        with torch.no_grad():
            if timesteps == None:
                timesteps = self.timesteps

            # Make the plot fig
            if self.plot_dim == 2:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5))
            elif self.plot_dim == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

            color = iter(cm.rainbow(np.linspace(0, 1, timesteps + 2)))
            next(color)  # do not use purple
            for time_idx in range(0, timesteps + 1):
                c = next(color)
                ax.contour(self.X, self.Y, self.Z, levels=[0], colors='gray', zorder=1)

                the_timepoint = torch.tensor([(self.TT / timesteps) * (time_idx)],
                                             dtype=torch.float).to(self.device)
                plot_sample = generator(the_timepoint[0].expand(self.num_plot_samples, 1),
                                        self.rho0_saved).cpu().detach().numpy()
                alpha = 1 if (time_idx == 0 or time_idx == timesteps) else 0.3
                if self.plot_dim == 2:
                    ax.scatter(plot_sample[:, 0], plot_sample[:, 1], color=c, alpha=alpha, zorder=2)
                elif self.plot_dim == 3:
                    ax.scatter(plot_sample[:, 0], plot_sample[:, 1], plot_sample[:, 2], color=c, alpha=alpha, zorder=2)

                ax.set_xlim([-self.plot_window_size, self.plot_window_size])
                ax.set_ylim([-self.plot_window_size, self.plot_window_size])
                if self.plot_dim == 3:
                    ax.set_zlim([-self.plot_window_size, self.plot_window_size])
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')

        return fig

    def make_plots(self, epoch, generator, the_logger):
        """
        Prints trajectories for 8 timesteps. Blue points are the start (time=0),
        red points are the end (time=T).
        """
        with torch.no_grad():
            generator.eval()

            # Make the plot
            fig = self._get_plot_fig(generator)
            fig.suptitle(f'epoch: {epoch}')
            if self.show_plots:
                plt.show()
            the_logger.save_plot(epoch, fig)
            plt.close(fig)

            generator.train()


# ===========================================================
#           Helper function to get time as a string
# ===========================================================
def _get_time_string():
    """
    Get the current time in a string
    """
    out = str(datetime.datetime.now()).replace(':', '-').replace(' ', '-')[:-7]
    out = out[:10] + '__' + out[11:]  # separate year-month-day from hour-minute-seconds

    return out

# ================================
#           Logger class
# ================================
class Logger(object):
    """
    A logger to log training.
    """
    def __init__(self, args):
        self.do_logging = args['do_logging']
        if self.do_logging:
            # Make the experiment directory
            run_dir = 'Run_' + _get_time_string() + '_' + args['experiment_name']
            self.experiment_dir = join('Experiments', run_dir)

            # Make plots folder
            self.plots_dir = join(self.experiment_dir, 'plots')
            os.makedirs(self.plots_dir)

            # Make CSV file for logging
            self.disc_csv_filepath = join(self.experiment_dir, 'disc_train_log.csv')
            self.gen_csv_filepath = join(self.experiment_dir, 'gen_train_log.csv')
            self.dict_of_csv_filepaths = {DISC_STRING: self.disc_csv_filepath,
                                          GEN_STRING: self.gen_csv_filepath}

            # Create dictionary of lists for discriminator and generator
            # Intended to become a dictionary of two dictionaries, where
            # each dictionary contains values that are lists. This is
            # allows us to automatically record values as long as it's
            # in the dictionary.
            self.train_log_dict = {DISC_STRING: {}, GEN_STRING: {}}

            # Print rate
            self.print_rate = args['print_rate']

            # Print/save training hyperparameters
            self.args_dir = join(self.experiment_dir, 'args')
            os.makedirs(self.args_dir)
            with open(join(self.args_dir, 'experiment_args.txt'), 'w') as ff:
                pp.pprint(args, ff)
            with open(join(self.args_dir, 'experiment_args.pkl'), 'wb') as ff:
                pickle.dump(args, ff)
            # Save environment parameters
            self.env = args['env']
            with open(join(self.args_dir, 'env_args.txt'), 'w') as ff:
                pp.pprint(self.env.info_dict, ff)
            with open(join(self.args_dir, 'env_args.pkl'), 'wb') as ff:
                pickle.dump(self.env.info_dict, ff)

    def _initialize_dict_of_lists(self, source_dict, disc_or_gen):
        """
        Create a new dictionary with the same keys as the source dictionary,
        and turn the values into lists.
        """
        self.train_log_dict[disc_or_gen] = source_dict.copy()
        for key, val in self.train_log_dict[disc_or_gen].items():
            self.train_log_dict[disc_or_gen][key] = [val]

    def log_training(self, training_info_dict, disc_or_gen):
        """
        Append stuff to the list for logging
        the_dict: Information about the training.
        """
        if self.do_logging:
            # If the dictionary of lists is empty, then initialize it
            if len(self.train_log_dict[disc_or_gen]) == 0:
                self._initialize_dict_of_lists(training_info_dict, disc_or_gen)
            else:
                for key, val in training_info_dict.items():
                    self.train_log_dict[disc_or_gen][key].append(val)

    def _initialize_training_csv(self, csv_path, disc_or_gen):
        """
        Essentially creates the csv, and writes the header.
        """
        key_list = ['epoch'] + list(self.train_log_dict[disc_or_gen].keys())
        with open(csv_path, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(key_list)

    def write_training_csv(self, epoch):
        """
        Write to the csv file.
        """
        if self.do_logging:
            for STRING in [DISC_STRING, GEN_STRING]:
                csvpath = self.dict_of_csv_filepaths[STRING]
                # Initialize the csv file
                if not os.path.exists(csvpath):
                    self._initialize_training_csv(csvpath, STRING)
                # Create a dictionary to print into a csv file
                dict_of_avg = {'epoch': epoch}
                for key, val in self.train_log_dict[STRING].items():
                    dict_of_avg[key] = np.mean(val[-self.print_rate:]) if val[0] != '--' else '--'
                # Write to the discriminator or generator training log file
                with open(csvpath, 'a') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(list(dict_of_avg.values()))

    def save_nets(self, the_dict):
        """
        Save the discriminator and generator models,
        and their optimizers.
        """
        if self.do_logging:
            self.model_path = join(self.experiment_dir, 'models')
            os.makedirs(self.model_path, exist_ok=True)

            epoch = the_dict['epoch']
            discriminator = the_dict['discriminator']
            discriminator_optimizer = the_dict['discriminator_optimizer']
            generator = the_dict['generator']
            generator_optimizer = the_dict['generator_optimizer']

            # Save discriminator
            torch.save({'epoch': epoch,
                        'model_state_dict': discriminator.state_dict(),
                '       optimizer': discriminator_optimizer.state_dict()},
                join(self.model_path, f'discriminator-epoch-{epoch}.pth.tar'))

            # Save generator
            torch.save({'epoch': epoch,
                        'model_state_dict': generator.state_dict(),
                        'optimizer': generator_optimizer.state_dict(),
                        'gen_mu': generator.mu,
                        'gen_std': generator.std},
                join(self.model_path, f'generator-epoch-{epoch}.pth.tar'))

    def save_plot(self, epoch, fig):
        """
        Save plots.
        """
        if self.do_logging:
            # Save the 2D (static) plot
            fig.savefig(join(self.plots_dir, f'plot-{epoch}.png'), bbox_inches='tight')
            fig.savefig(join(self.experiment_dir, f'latest_plot.png'), bbox_inches='tight')

    def print_to_console(self, td, disc_or_gen):
        """
        Stuff to print to the console.
        """
        with torch.no_grad():
            # Setup variables
            error_msg = f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}'
            if disc_or_gen == DISC_STRING:
                disc_00_loss = td['disc_00_loss']

            # Start printing to the console
            if disc_or_gen == DISC_STRING:
                print('DISCRIMINATOR losses:')
            elif disc_or_gen == GEN_STRING:
                print('GENERATOR losses:')
            else:
                raise ValueError(error_msg)
            print('hjb_loss:', td['hjb_loss'])
            print('hjb_error:', td['hjb_error'])
            if disc_or_gen == DISC_STRING:
                print('disc_00_loss:', disc_00_loss)
            print('FF_obstacle_loss:', td['FF_obstacle_loss'])
            print('FF_congestion_loss:', td['FF_congestion_loss'])
            print('total_loss:', td['total_loss'])

            print()
