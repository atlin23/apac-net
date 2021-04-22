import math
import numpy as np
import torch
from utils.utils import DISC_STRING, GEN_STRING
from torch import sin, cos


# ==========================================
#           Quadcopter Environment
# ==========================================
class QuadcopterEnv(object):
    """
    Quadcopter environment.
    """
    def __init__(self, device):
        self.dim = 12
        self.latent_dim = self.dim
        self.nu = 0.01
        self.TT = 4
        self.mm = 0.5  # Mass of quadcopter
        self.gg = 9.81  # Gravity acceleration constant
        self.ham_scale = 1
        self.psi_scale = 5
        self.rho0_var = 1/4
        self.lam_obstacle = 0
        self.lam_congestion = 20
        self.lam_quadrun = 0
        self.device = device
        self.name = "QuadcopterEnv"  # Environment name

        # Options for plotting
        self.plot_window_size = 3
        self.plot_dim = 3

        self.info_dict = {'env_name': self.name, 'dim': self.dim, 'nu': self.nu,
                          'ham_scale': self.ham_scale, 'psi_scale': self.psi_scale,
                          'lam_obstacle': self.lam_obstacle, 'lam_congestion': self.lam_congestion}

    def _sqeuc(self, xx):
        return torch.sum(xx * xx, dim=1, keepdim=True)

    def sample_rho0(self, num_samples):
        """
        The initial distribution rho_0 of the agents.
        """
        mu = torch.tensor([[-2, -2, -2]], dtype=torch.float)
        out_pos = math.sqrt(self.rho0_var) * torch.randn(size=(num_samples, 3)) + mu
        out_rest = math.sqrt(0.001) * torch.randn(size=(num_samples, 9))  # small perturbation
        out = torch.cat((out_pos, out_rest), dim=1)
        return out

    def ham(self, tt, xx, pp):
        """
        The Hamiltonian.
        """
        sum1 = torch.sum(xx[:, 6:] * pp[:, :6], dim=1)
        sum2 = (0.25 / (self.mm ** 2)) * (
                    (sin(xx[:, 5])*sin(xx[:, 3]) + cos(xx[:, 5])*cos(xx[:, 3])*sin(xx[:, 4])) * pp[:, 6]
                    + (cos(xx[:, 5])*sin(xx[:, 4])*sin(xx[:, 3]) - cos(xx[:, 3])*sin(xx[:, 5])) * pp[:, 7]
                    + (cos(xx[:, 4])*cos(xx[:, 5])) * pp[:, 8]
                ) ** 2
        sum3 = -pp[:, 8] * self.gg + 0.25 * torch.sum(pp[:, 9:]**2, dim=1) - 2

        out = torch.unsqueeze(sum1 + sum2 + sum3, -1)
        out = self.ham_scale * out

        return out

    def get_trace(self, grad, xx, batch_size, dim, grad_outputs_vec):
        """
        Computation of the second-order term in the HJB equation.
        """
        hess_stripes = torch.autograd.grad(outputs=grad, inputs=xx,
                                   grad_outputs=grad_outputs_vec,
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        pre_laplacian = torch.stack([hess_stripes[i * batch_size: (i+1) * batch_size, i]
                                     for i in range(0, dim)], dim=1)
        laplacian = torch.sum(pre_laplacian, dim=1)
        laplacian_sum_repeat = laplacian.repeat((1, dim))

        return laplacian_sum_repeat.T

    def psi_func(self, xx_inp):
        """
        The final-time cost function.
        """
        xx_euc_pos = xx_inp[:,  0:3]
        xx_ang_pos = xx_inp[:, 3:6]
        xx_euc_vel = xx_inp[:, 6:9]
        xx_ang_vel = xx_inp[:, 9:]

        center_euc_pos = torch.tensor([[2, 2, 2]], dtype=torch.float).to(self.device)
        center_euc_vel = torch.tensor([[0, 0, 0]], dtype=torch.float).to(self.device)

        xx_euc = torch.cat([xx_euc_pos, xx_euc_vel], dim=1)
        center_euc = torch.cat([center_euc_pos, center_euc_vel], dim=1)

        out = torch.norm(xx_euc - center_euc, dim=1, keepdim=True)
        out = self.psi_scale * out

        return out

    def FF_func(self, td, tt_samples, rhott_samples, disc_or_gen):
        """
        Computes the forcing term (i.e. obstacle or interaction between agents) of the
        Hamilton-Jacobi equation.
        """
        sample_size = rhott_samples.size(0)
        FF_total_tensor = torch.zeros((sample_size, 1)).to(td['device'])
        info = {'FF_obstacle_loss': '--', 'FF_congestion_loss': '--'}

        # Obstacle
        if self.lam_obstacle > 0:
            FF_obstacle_tensor = self.lam_obstacle * self._FF_obstacle_loss(rhott_samples)
            FF_total_tensor += FF_obstacle_tensor
            info['FF_obstacle_loss'] = FF_obstacle_tensor.mean(dim=0).item()

        # Congestion
        if self.lam_congestion > 0:
            FF_congestion_tensor = self.lam_congestion * self._FF_congestion_loss(
                td['generator'], tt_samples, rhott_samples, disc_or_gen, first_d_dim=3)
            FF_total_tensor += FF_congestion_tensor
            info['FF_congestion_loss'] = FF_congestion_tensor.mean(dim=0).item()

        FF_total_tensor *= td['TT'][0].item()

        return FF_total_tensor, info

    def _FF_obstacle_loss(self, xx_inp, scale=1):
        """
        Calculate interaction term. Calculates F(x), where F is the forcing term in the HJB equation.
        """
        batch_size = xx_inp.size(0)
        out = torch.zeros((batch_size, 1))

        return out

    def _FF_congestion_loss(self, generator, tt_samples, rhott_samples, disc_or_gen, first_d_dim=3):
        """
        Interaction term, for congestion.
        """
        dim = 3
        rho00_2 = self.sample_rho0(rhott_samples.size(0)).to(self.device)

        if disc_or_gen == DISC_STRING:
            rhott_samples2 = generator(tt_samples, rho00_2).detach()
        elif disc_or_gen == GEN_STRING:
            rhott_samples2 = generator(tt_samples, rho00_2)
        else:
            raise ValueError(f'Invalid disc_or_gen. Should be \'disc\' or \'gen\' but got: {disc_or_gen}')
        rhott_samples_first_d = rhott_samples[:, :first_d_dim]
        rhott_samples2_first_d = rhott_samples2[:, :first_d_dim]

        # Exponential distance
        sig = 1
        squared_dist = self._sqeuc(rhott_samples_first_d - rhott_samples2_first_d)
        normalizing_const = (1 / sig) * (1 / (np.sqrt(2 * np.pi) ** dim))
        out = normalizing_const * torch.exp(-0.5 * squared_dist / (sig * sig))

        return out

    def FF_obstacle_func(self, x, y):
        """
        No obstacles
        """
        out = 0

        return out