from utils.utils import act_funcs, DISC_STRING, GEN_STRING
from neural_net_defs import *
from utils.train_once import train_once
from utils.utils import Plotter, Logger


def start_train(a):
    """
    a: A dictionary containing the training arguments
    """
    env = a['env']
    the_logger = Logger(a)
    the_plotter = Plotter(a, the_logger)

    # =============================================
    #           Precompute some variables
    # =============================================
    # Precompute ones tensor of size phi out for gradient computation
    ones_of_size_phi_out = torch.ones(a['batch_size'] * env.dim, 1).to(a['device']) if env.nu > 0 \
                           else torch.ones(a['batch_size'], 1).to(a['device'])

    # Precompute grad outputs vec for laplacian for Hessian computation
    list_1 = []
    for i in range(env.dim):
        vec = torch.zeros(size=(a['batch_size'], env.dim), dtype=torch.float).to(a['device'])
        vec[:, i] = torch.ones(size=(a['batch_size'],)).to(a['device'])
        list_1.append(vec)
    grad_outputs_vec = torch.cat(list_1, dim=0)

    # ======================================
    #           Setup the learning
    # ======================================
    # Compute the mean and variance of rho0 (assuming rho0 is a simple Gaussian)
    temp_sample = env.sample_rho0(int(1e4)).to(a['device'])
    mu = temp_sample.mean(axis=0)
    std = torch.sqrt(temp_sample.var(axis=0))
    if 0 in std:
        raise ValueError("std of sample_rho0 has a zero!")

    # Make the networks
    discriminator = DiscNet(dim=env.dim, ns=a['ns'], act_func=act_funcs[a['act_func_disc']], hh=a['hh'],
                            device=a['device'], psi_func=env.psi_func, TT=env.TT).to(a['device'])
    generator = GenNet(dim=env.dim, ns=a['ns'], act_func=act_funcs[a['act_func_gen']], hh=a['hh'], device=a['device'],
                       mu=mu, std=std, TT=env.TT).to(a['device'])

    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=a['disc_lr'], weight_decay=a['weight_decay'],
                                      betas=a['betas'])
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=a['gen_lr'], weight_decay=a['weight_decay'],
                                     betas=a['betas'])

    # ===================================
    #           Start iteration
    # ===================================
    # Define initial time and final time constants
    zero = torch.tensor([0], dtype=torch.float).expand((a['batch_size'], 1)).to(a['device'])
    TT = torch.tensor([env.TT], dtype=torch.float).expand((a['batch_size'], 1)).to(a['device'])

    # Start the iteration
    for epoch in range(a['max_epochs'] + 1):
        # =============================
        #           Info dump
        # =============================
        if epoch % a['print_rate'] == 0:
            print()
            print('-' * 10)
            print(f'epoch: {epoch}\n')

            if epoch != 0:
                # Saving neural network and saving to csv
                the_logger.save_nets({'epoch': epoch,
                                      'discriminator': discriminator,
                                      'discriminator_optimizer': disc_optimizer,
                                      'generator': generator,
                                      'generator_optimizer': gen_optimizer})
                the_logger.write_training_csv(epoch)

        # ===========================================
        #           Setup training dictionary
        # ===========================================
        train_dict = a.copy()
        train_dict.update({'discriminator': discriminator,
                           'generator': generator,
                           'disc_optimizer': disc_optimizer,
                           'gen_optimizer': gen_optimizer,
                           'ham_func': env.ham,
                           'epoch': epoch,
                           'zero': zero,
                           'TT': TT,
                           'ones_of_size_phi_out': ones_of_size_phi_out,
                           'grad_outputs_vec': grad_outputs_vec,
                           'the_logger': the_logger})

        # ===========================================
        #           Train phi/discriminator
        # ===========================================
        train_info = train_once(train_dict, DISC_STRING)

        the_logger.log_training(train_info, DISC_STRING)
        if epoch % a['print_rate'] == 0:
            the_logger.print_to_console(train_info, DISC_STRING)

        # ======================================
        #           Train rho/generator
        # ======================================
        if epoch % a['gen_every_disc'] == 0:  # How many times to update discriminator per one update of generator.
            train_info = train_once(train_dict, GEN_STRING)

        the_logger.log_training(train_info, GEN_STRING)
        if epoch % a['print_rate'] == 0:
            the_logger.print_to_console(train_info, GEN_STRING)

        # =======================================
        #           Plot images and etc.
        # =======================================
        if epoch % a['print_rate'] == 0:
            the_plotter.make_plots(epoch, generator, the_logger)

    return the_logger
