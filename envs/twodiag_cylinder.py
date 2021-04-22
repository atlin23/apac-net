import math
import numpy as np
import torch
from utils.utils import DISC_STRING, GEN_STRING, sqeuc


# =========================================
#           Two Diagonal Obstacle
# =========================================
class TwoDiagCylinderEnv(object):
    """
    Cylindrical, two-obstacle environment.
    """
    def __init__(self, device):
        """
        The environment class.
        """
        self.dim = 2
        self.nu = 0.4
        self.TT = 1
        self.ham_scale = 8
        self.psi_scale = 1
        self.lam_obstacle = 5
        self.lam_congestion = 0
        self.device = device
        self.name = "TwoDiagEnv"  # Environment name

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
        mu = torch.tensor([[-2, -2] + [0] * (self.dim - 2)], dtype=torch.float)
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
        center = torch.tensor([[2, 2]], dtype=torch.float).to(self.device)
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

        # Two diagonal obstacles
        # Rotation matrix
        theta = torch.tensor(np.pi / 5)
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]]).expand(batch_size, dim, dim).to(self.device)

        # Bottom/Left obstacle  # TODO: Clean it up
        center1 = torch.tensor([-2, 0.5], dtype=torch.float).to(self.device)
        xxcent1 = xx - center1
        xxcent1 = xxcent1.unsqueeze(1).bmm(rot_mat).squeeze(1)
        covar_mat1 = torch.eye(dim, dtype=torch.float)
        covar_mat1[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
        covar_mat1 = covar_mat1.expand(batch_size, dim, dim).to(self.device)
        bb_vec1 = torch.tensor([0, 2], dtype=torch.float).expand(xx.size()).to(self.device)
        xxcov1 = xxcent1.unsqueeze(1).bmm(covar_mat1)
        quad1 = torch.bmm(xxcov1, xxcent1.unsqueeze(2)).view(-1, 1)
        lin1 = torch.sum(xxcent1 * bb_vec1, dim=1, keepdim=True)
        out1 = (-1) * ((quad1 + lin1) + 1)
        out1 = scale * out1.view(-1, 1)
        out1 = torch.clamp_min(out1, min=0)

        # Top/Right obstacle
        center2 = torch.tensor([2, -0.5], dtype=torch.float).to(self.device)
        xxcent2 = xx - center2
        xxcent2 = xxcent2.unsqueeze(1).bmm(rot_mat).squeeze(1)
        covar_mat2 = torch.eye(dim, dtype=torch.float)
        covar_mat2[0:2, 0:2] = torch.tensor(np.array([[5, 0], [0, 0]]), dtype=torch.float)
        covar_mat2 = covar_mat2.expand(batch_size, dim, dim).to(self.device)
        bb_vec2 = torch.tensor([0, -2], dtype=torch.float).expand(xx.size()).to(self.device)
        xxcov2 = xxcent2.unsqueeze(1).bmm(covar_mat2)
        quad2 = torch.bmm(xxcov2, xxcent2.unsqueeze(2)).view(-1, 1)
        lin2 = torch.sum(xxcent2 * bb_vec2, dim=1, keepdim=True)
        out2 = (-1) * ((quad2 + lin2) + 1)
        out2 = scale * out2.view(-1, 1)
        out2 = torch.clamp_min(out2, min=0)

        out = out1 + out2

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
        The two-diagonal obstacle, agents going bottom-left to top-right
        """
        ## Two diagonal obstacles
        # Rotation matrix
        theta = np.pi / 5
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float)

        # Bottom/Left obstacle
        center1 = np.array([-2, 0.5], dtype=np.float)
        vec1 = np.array([x, y], dtype=np.float) - center1
        vec1 = np.dot(vec1, rot_mat)
        mat1 = np.array([[5, 0], [0, 0]], dtype=np.float)
        bb1 = np.array([0, 2], dtype=np.float)
        quad1 = np.dot(vec1, np.dot(mat1, vec1))
        lin1 = np.dot(vec1, bb1)
        out1 = np.clip((-1) * (quad1 + lin1 + 1), a_min=-0.1, a_max=None)

        # Top/Right obstacle
        center2 = np.array([2, -0.5], dtype=np.float)
        vec2 = np.array([x, y], dtype=np.float) - center2
        vec2 = np.dot(vec2, rot_mat)
        mat2 = np.array([[5, 0], [0, 0]], dtype=np.float)
        bb2 = np.array([0, -2], dtype=np.float)
        quad2 = np.dot(vec2, np.dot(mat2, vec2))
        lin2 = np.dot(vec2, bb2)
        out2 = np.clip((-1) * (quad2 + lin2 + 1), a_min=-0.1, a_max=None)

        out = out1 + out2

        return out
