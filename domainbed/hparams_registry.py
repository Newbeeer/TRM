# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np

def _hparams(algorithm, dataset, random_state, args):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}
    hparams['opt'] = (args.opt, args.opt)
    hparams['data_augmentation'] = (True, True)
    hparams['resnet50'] = (args.resnet50, False)
    hparams['resnet_dropout'] = (0., random_state.choice([0., 0.1, 0.5]))
    hparams['class_balanced'] = (args.class_balanced, False)
    hparams['shift'] = (args.shift, args.shift)
    if dataset not in SMALL_IMAGES:
        hparams['lr'] = (1e-1, 10**random_state.uniform(-5, -3.5))
        hparams['sch_size'] = (600, 100)
        args.epochs = 50
        if dataset == 'DomainNet':
            hparams['batch_size'] = (32, int(2**random_state.uniform(3, 5)))
        else:
            hparams['batch_size'] = (64, int(2**random_state.uniform(3, 5.5)))
        if algorithm == "ARM":
            hparams['batch_size'] = (8, 8)
        if dataset == 'PACS' or dataset == 'VLCS' or dataset == 'OfficeHome':
            hparams['lr'] = (1e-4, 10 ** random_state.uniform(-5, -3.5))
            # no schedule
            hparams['sch_size'] = (60000, 100)
            hparams['batch_size'] = (32, int(2 ** random_state.uniform(3, 5)))
        if dataset == 'SceneCOCO':
            args.epochs = 100
            hparams['batch_size'] = (128, int(2 ** random_state.uniform(3, 9)))
            hparams['sch_size'] = (600, 100)
        if dataset == 'VLCS':
            hparams['batch_size'] = (16, int(2 ** random_state.uniform(3, 5)))
        if dataset == 'Celeba':
            # Small images
            hparams['lr'] = (1e-4, 10 ** random_state.uniform(-4.5, -2.5))
            hparams['batch_size'] = (32, int(2 ** random_state.uniform(3, 9)))
            hparams['sch_size'] = (600000, 100)
    else:
        # Small images
        hparams['lr'] = (1e-1, 10**random_state.uniform(-4.5, -2.5))
        hparams['batch_size'] = (128, int(2**random_state.uniform(3, 9)))
        hparams['sch_size'] = (600, 100)
        if dataset == 'ColoredMNIST':
            args.epochs = 10
        elif dataset == 'SceneCOCO':
            args.epochs = 100

    if dataset in SMALL_IMAGES:
        hparams['weight_decay'] = (1e-4, 0.)
    else:
        hparams['weight_decay'] = (1e-4, 10**random_state.uniform(-6, -2))
    if algorithm in ['DANN', 'CDANN']:
        if dataset not in SMALL_IMAGES:
            hparams['lr_g'] = (5e-5, 10**random_state.uniform(-5, -3.5))
            hparams['lr_d'] = (5e-5, 10**random_state.uniform(-5, -3.5))
        else:
            hparams['lr_g'] = (1e-3, 10**random_state.uniform(-4.5, -2.5))
            hparams['lr_d'] = (1e-3, 10**random_state.uniform(-4.5, -2.5))

        if dataset in SMALL_IMAGES:
            hparams['weight_decay_g'] = (0., 0.)
        else:
            hparams['weight_decay_g'] = (0., 10**random_state.uniform(-6, -2))

        hparams['lambda'] = (1.0, 10**random_state.uniform(-2, 2))
        hparams['weight_decay_d'] = (0., 10**random_state.uniform(-6, -2))
        hparams['d_steps_per_g_step'] = (1, int(2**random_state.uniform(0, 3)))
        hparams['grad_penalty'] = (0., 10**random_state.uniform(-2, 1))
        hparams['beta1'] = (0.5, random_state.choice([0., 0.5]))
        hparams['mlp_width'] = (256, int(2 ** random_state.uniform(6, 10)))
        hparams['mlp_depth'] = (3, int(random_state.choice([3, 4, 5])))
        hparams['mlp_dropout'] = (0., random_state.choice([0., 0.1, 0.5]))
    elif algorithm == "RSC":
        hparams['rsc_f_drop_factor'] = (1/3, random_state.uniform(0,0.5)) # Feature drop factor
        hparams['rsc_b_drop_factor'] = (1/3, random_state.uniform(0, 0.5)) # Batch drop factor
    elif algorithm == 'Fish':
        hparams['meta_lr'] = (args.fish_lam, lambda r: r.choice([0.05, 0.1, 0.5]))
        hparams['iters'] = (200, int(10 ** random_state.uniform(0, 4)))
    elif algorithm == "SagNet":
        hparams['sag_w_adv'] = (0.1, 10**random_state.uniform(-2, 1))
    elif algorithm == "IRM":
        hparams['irm_lambda'] = (args.irm_lam, 10**random_state.uniform(-1, 5))
        hparams['irm_penalty_anneal_iters'] = (100, int(10**random_state.uniform(0, 4)))
    elif algorithm == "Mixup":
        hparams['mixup_alpha'] = (0.2, 10**random_state.uniform(-1, -1))
    elif algorithm == "GroupDRO":
        hparams['groupdro_eta'] = (args.dro_eta, 10**random_state.uniform(-3, -1))
        hparams['weight_decay'] = (1e-4, 0.)
    elif algorithm == "MMD" or algorithm == "CORAL":
        hparams['mmd_gamma'] = (1., 10**random_state.uniform(-1, 1))
    elif algorithm == "MLDG":
        hparams['mldg_beta'] = (1, 10**random_state.uniform(-1, 1))
        hparams['iters'] = (200, int(10 ** random_state.uniform(0, 4)))
    elif algorithm == "MTL":
        hparams['mtl_ema'] = (.99, random_state.choice([0.5, 0.9, 0.99, 1.]))
    elif algorithm == "VREx":
        hparams['vrex_lambda'] = (args.rex_lam, 10**random_state.uniform(-1, 5))
        hparams['vrex_penalty_anneal_iters'] = (500, int(10**random_state.uniform(0, 4)))
    elif algorithm == "ERM":
        pass
    elif algorithm == "TRM" or algorithm == "TRM2":
        hparams['cos_lambda'] = (args.cos_lam, 10 ** random_state.uniform(-1, 5))
        hparams['n'] = (100, 10**random_state.uniform(-1, 5))
        hparams['iters'] = (100, int(10 ** random_state.uniform(0, 4)))
        hparams['groupdro_eta'] = (args.dro_eta, 10 ** random_state.uniform(-3, -1))
        if dataset == 'SceneCOCO':
            hparams['iters'] = (200, int(10**random_state.uniform(0, 4)))
        # elif dataset == 'PACS' or dataset == 'VLCS':
        #     hparams['iters'] = (1000, int(10**random_state.uniform(0, 4)))
        # elif dataset == 'OfficeHome':
        #     hparams['iters'] = (1000, int(10**random_state.uniform(0, 4)))
    elif algorithm == "TRM_DRO":
        hparams['cos_lambda'] = (args.cos_lam, 10 ** random_state.uniform(-1, 5))
        hparams['n'] = (10, 10**random_state.uniform(-1, 5))
        if dataset == 'SceneCOCO':
            hparams['iters'] = (200, int(10**random_state.uniform(0, 4)))
        elif dataset == 'PACS':
            hparams['iters'] = (100, int(10**random_state.uniform(0, 4)))
            hparams['n'] = (50, 10 ** random_state.uniform(-1, 5))
        elif dataset == 'Celeba':
            hparams['iters'] = (300, int(10 ** random_state.uniform(0, 4)))
            hparams['n'] = (50, 10 ** random_state.uniform(-1, 5))
        else:
            hparams['iters'] = (100, int(10 ** random_state.uniform(0, 4)))

        hparams['groupdro_eta'] = (args.dro_eta, 10**random_state.uniform(-3, -1))
    return hparams

def default_hparams(algorithm, dataset, args):
    dummy_random_state = np.random.RandomState(0)
    return {a: b for a,(b,c) in
        _hparams(algorithm, dataset, dummy_random_state, args).items()}

def random_hparams(algorithm, dataset, seed, args):
    random_state = np.random.RandomState(seed)
    return {a: c for a,(b,c) in _hparams(algorithm, dataset, random_state, args).items()}
