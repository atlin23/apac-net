import torch
from utils.utils import uniform_time_sampler, DISC_STRING, GEN_STRING


# ================
# Helper functions
# ================
def get_generator_samples(td, disc_or_gen):
    """
    Get generator samples.
    """
    rho00 = td['env'].sample_rho0(td['batch_size']).to(td['device'])
    tt_samples = (td['TT'][0].item() * uniform_time_sampler(td['batch_size'])).to(td['device'])

    if disc_or_gen == DISC_STRING:
        rhott_samples = td['generator'](tt_samples, rho00).detach().requires_grad_(True)
    elif disc_or_gen == GEN_STRING:
        rhott_samples = td['generator'](tt_samples, rho00)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be \'disc\' or \'gen\' but got: {disc_or_gen}')

    return rho00, tt_samples, rhott_samples


def get_hjb_loss(td, tt_samples, rhott_samples, batch_size, ones_of_size_phi_out, grad_outputs_vec):
    """
    Compute the HJB error.
    """
    env = td['env']
    # Integral for the Hamilton-Jacobi part
    if env.nu > 0:  # Repeate to parallelize computing the Laplacian/trace for each sample of the batch.
        rhott_samples = rhott_samples.repeat(repeats=(env.dim, 1))
        tt_samples = tt_samples.repeat(repeats=(env.dim, 1))
    tt_samples.requires_grad_(True)  # WARNING: Keep this after generator evaluation, or else you chain rule generator's time variable
    phi_out = td['discriminator'](tt_samples, rhott_samples)
    phi_grad_tt = torch.autograd.grad(outputs=phi_out, inputs=tt_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    phi_grad_xx = torch.autograd.grad(outputs=phi_out, inputs=rhott_samples,
                                      grad_outputs=ones_of_size_phi_out,
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    if env.nu > 0:
        phi_trace_xx = env.get_trace(phi_grad_xx, rhott_samples, batch_size, env.dim, grad_outputs_vec)
    else:
        phi_trace_xx = torch.zeros(phi_grad_tt.size()).to(td['device'])
    ham = env.ham(tt_samples, rhott_samples, (-1) * phi_grad_xx)

    out = (phi_grad_tt + env.nu * phi_trace_xx - ham) * td['TT'][0].item()

    # Compute some info
    info = {'phi_trace_xx': phi_trace_xx.mean(dim=0).item() * td['TT'][0].item()}

    return out, info


def get_FF_loss(td, tt_samples, rhott_samples, disc_or_gen):
    """
    Compute the forcing terms (interaction terms and obstacles).
    """
    FF_total_tensor, FF_info = td['env'].FF_func(td, tt_samples, rhott_samples, disc_or_gen)

    return FF_total_tensor, FF_info


def get_disc_00_loss(td, discriminator):
    """
    Integral of phi_0 * rho_0.
    """
    rho0_samples = td['env'].sample_rho0(td['batch_size']).to(td['device'])
    disc_00_loss = discriminator(td['zero'], rho0_samples).mean(dim=0)

    return disc_00_loss


def set_requires_grad(discriminator, generator, disc_or_gen):
    """
    Turn on requires_grad for the one we're training, and turn off for the one we aren't. For speed.
    """
    if disc_or_gen == DISC_STRING:
        for param in discriminator.parameters():
            param.requires_grad_(True)
        for param in generator.parameters():
            param.requires_grad_(False)
    elif disc_or_gen == GEN_STRING:
        for param in discriminator.parameters():
            param.requires_grad_(False)
        for param in generator.parameters():
            param.requires_grad_(True)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


def set_zero_grad(disc_optimizer, gen_optimizer, disc_or_gen):
    """
    Zero the gradients for the one we're training.
    """
    if disc_or_gen == DISC_STRING:
        disc_optimizer.zero_grad()
    elif disc_or_gen == GEN_STRING:
        gen_optimizer.zero_grad()
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


def do_grad_clip(discriminator, generator, clip_value, disc_or_gen):
    """
    Clips the gradient of the discriminator and/or generator.
    """
    if disc_or_gen == DISC_STRING:
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=clip_value)
    elif disc_or_gen == GEN_STRING:
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=clip_value)
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')


def optimizer_step(disc_optimizer, gen_optimizer, disc_or_gen):
    """
    Take a step of the discriminator or generator optimizers.
    """
    if disc_or_gen == DISC_STRING:
        disc_optimizer.step()
    elif disc_or_gen == GEN_STRING:
        gen_optimizer.step()
    else:
        raise ValueError(f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}')