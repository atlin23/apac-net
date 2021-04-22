import torch
from utils.utils import DISC_STRING, GEN_STRING
from utils.helpers_train_once import set_requires_grad, set_zero_grad, get_generator_samples, get_hjb_loss, \
                                     get_FF_loss, get_disc_00_loss, optimizer_step


# ====================================
#           The main trainer
# ====================================
def train_once(td, disc_or_gen):
    """
    Trains the discriminator and generator.
    """
    error_msg = f'Invalid disc_or_gen. Should be {DISC_STRING} or {GEN_STRING} but got: {disc_or_gen}'
    assert (disc_or_gen == DISC_STRING or disc_or_gen == GEN_STRING), error_msg

    # Activate computing computational graph of discriminator/generator
    set_requires_grad(td['discriminator'], td['generator'], disc_or_gen)

    # Zero the gradients
    set_zero_grad(td['disc_optimizer'], td['gen_optimizer'], disc_or_gen)

    # Integral for the Hamilton-Jacobi part
    rho00, tt_samples, rhott_samples = get_generator_samples(td, disc_or_gen)
    hjb_loss_tensor, hjb_loss_info = get_hjb_loss(td, tt_samples, rhott_samples, td['batch_size'],
                                                  td['ones_of_size_phi_out'], td['grad_outputs_vec'])
    hjb_loss_tensor = hjb_loss_tensor[:td['batch_size']]
    hjb_loss = hjb_loss_tensor.mean(dim=0)

    # Interaction terms
    FF_total_tensor, forcing_info = get_FF_loss(td, tt_samples, rhott_samples, disc_or_gen)

    # Finish computing the total loss
    if disc_or_gen == DISC_STRING:
        # Integral of phi_0 * rho_0
        disc_00_loss = get_disc_00_loss(td, td['discriminator'])
        # L2 Hamiltonian residual
        disc_hjb_error = torch.norm(hjb_loss_tensor + FF_total_tensor, dim=1).mean(dim=0) / td['TT'][0].item()
        # Total loss
        total_loss = (-1) * (disc_00_loss + hjb_loss) + td['lam_hjb_error'] * disc_hjb_error
    else:  # disc_or_gen == GEN_STRING:
        # Total loss
        FF_total_loss = FF_total_tensor.mean(dim=0)
        total_loss = hjb_loss + FF_total_loss

    # Backprop and optimize
    total_loss.backward()
    optimizer_step(td['disc_optimizer'], td['gen_optimizer'], disc_or_gen)

    # Get info about the training
    with torch.no_grad():
        hjb_error = torch.norm(hjb_loss_tensor + FF_total_tensor, dim=1).mean(dim=0) / td['TT'][0].item()
        training_info = {'total_loss': total_loss.item(), 'hjb_loss': hjb_loss.item(), 'hjb_error': hjb_error.item()}
        training_info.update(hjb_loss_info)
        training_info.update(forcing_info)
        if disc_or_gen == DISC_STRING:
            training_info['disc_00_loss'] = disc_00_loss.item()

    return training_info
