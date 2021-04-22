import math
import numpy as np
import torch
from utils.utils import DISC_STRING, GEN_STRING, sqeuc


# =======================================
#           Bottleneck obstacle
# =======================================
class BottleneckCylinderEnv(object):
    """
    Cylindrical Bottleneck obstacle environment.
    """
    def __init__(self, device):
        self.dim = 2
        self.nu = 0.1
        self.TT = 1
        self.ham_scale = 5
        self.psi_scale = 1
        self.lam_obstacle = 5
        self.lam_congestion = 1
        self.device = device
        self.name = "BottleneckCylinderEnv"  # Environment name

        # Options for plotting
        self.plot_window_size = 3
        self.plot_dim = 2

        self.info_dict = {'env_name': self.name, 'dim': self.dim, 'nu': self.nu,
                          'ham_scale': self.ham_scale, 'psi_scale': self.psi_scale,
                          'lam_obstacle': self.lam_obstacle, 'lam_congestion': self.lam_congestion}

    def sample_rho0(self, num_samples, var_scale=1/10):
        """
        The initial distribution rho_0 of the agents.
        """
        mu = torch.tensor([[-2, 0] + [-2] * (self.dim - 2)], dtype=torch.float)
        out = math.sqrt(var_scale) * torch.randn(size=(num_samples, self.dim)) + mu

        return out

    def ham(self, tt, xx, pp):
        """
        The Hamiltonian.
        """
        out = self.ham_scale * torch.norm(pp, dim=1, keepdim=True)

        return out

    def get_trace(self, grad, xx, batch_size, dim, grad_outputs_vec):
        """
        Computation of the second-order term in the HJB equation.
        """
        hess_stripes = torch.autograd.grad(outputs=grad, inputs=xx,
                                           grad_outputs=grad_outputs_vec,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]
        pre_laplacian = torch.stack([hess_stripes[i * batch_size: (i + 1) * batch_size, i]
                                     for i in range(0, dim)], dim=1)
        laplacian = torch.sum(pre_laplacian, dim=1)
        laplacian_sum_repeat = laplacian.repeat((1, dim))

        return laplacian_sum_repeat.T

    def psi_func(self, xx_inp):
        """
        The final-time cost function.
        """
        xx = xx_inp[:, 0:2]
        center = torch.tensor([[2, 0]], dtype=torch.float).to(self.device)
        out = self.psi_scale * torch.norm(xx - center, dim=1, keepdim=True)

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
                td['generator'], tt_samples, rhott_samples, disc_or_gen, first_d_dim=2)
            FF_total_tensor += FF_congestion_tensor
            info['FF_congestion_loss'] = FF_congestion_tensor.mean(dim=0).item()

        FF_total_tensor *= td['TT'][0].item()

        return FF_total_tensor, info

    def _FF_obstacle_loss(self, xx_inp, scale=1):
        """
        Calculate interaction term. Calculates F(x), where F is the forcing term in the HJB equation.
        """
        batch_size = xx_inp.size(0)
        xx = xx_inp[:, 0:2]
        dim = xx.size(1)
        assert (dim == 2), f"Require dim=2 but, got dim={dim} (BAD)"

        center = torch.tensor([0, 0] + [0] * (dim - 2), dtype=torch.float).to(self.device)
        covar_mat = torch.eye(dim, dtype=torch.float)
        covar_mat[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, -1]]),
                                           dtype=torch.float)
        covar_mat = covar_mat.expand(batch_size, dim, dim).to(self.device)
        xxmu = (xx - center).unsqueeze(1).bmm(covar_mat)
        out = (-1) * torch.bmm(xxmu, (xx - center).unsqueeze(2)) - 0.1
        out = scale * out.view(-1, 1)
        out = torch.relu(out)

        return out

    def _FF_congestion_loss(self, generator, tt_samples, rhott_samples, disc_or_gen, first_d_dim=2):
        """
        Interaction term, for congestion.
        """
        rho00_2 = self.sample_rho0(rhott_samples.size(0)).to(self.device)

        if disc_or_gen == DISC_STRING:
            rhott_samples2 = generator(tt_samples, rho00_2).detach()
        elif disc_or_gen == GEN_STRING:
            rhott_samples2 = generator(tt_samples, rho00_2)
        else:
            raise ValueError(f'Invalid disc_or_gen. Should be \'disc\' or \'gen\' but got: {disc_or_gen}')
        rhott_samples_first_d = rhott_samples[:, :first_d_dim]
        rhott_samples2_first_d = rhott_samples2[:, :first_d_dim]

        distances = sqeuc(rhott_samples_first_d - rhott_samples2_first_d)
        out = 1 / (distances + 1)

        return out

    def FF_obstacle_func(self, x, y):
        """
        The bottleneck obstacle, located at the top and bottom, for plotting purposes.
        """
        center = np.array([0, 0], dtype=np.float)
        mat = np.array([[5, 0], [0, -1]], dtype=np.float)
        vec = np.array([x, y], dtype=np.float) - center
        quad = np.dot(vec, np.dot(mat, vec))
        out = (-1) * quad - 0.1

        return out