# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import random
import copy
import numpy as np

from domainbed import networks
from domainbed import grad_fun
from domainbed.lib.misc import random_pairs_of_minibatches, ParamDict
from domainbed.lib.misc import kl_loss_function
import torch.nn.utils.prune as prune

ALGORITHMS = [
    'ERM',
    'IRM',
    'GroupDRO',
    'Mixup',
    'Fish',
    'MLDG',
    'CORAL',
    'MMD',
    'DANN',
    'CDANN',
    'MTL',
    'SagNet',
    'ARM',
    'VREx',
    'RSC',
    'EC',
    'Swap',
    'TRM'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)

        if self.hparams['opt'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        if self.hparams['opt'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches):
        self.network.train()
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        feature = self.featurizer(all_x)
        loss = F.cross_entropy(self.classifier(feature), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.network.eval()
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        penalty = 0.
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            penalty += self._irm_penalty(logits, y)
        penalty /= len(minibatches)
        return {'loss': loss.item(), 'penalty': penalty.item()}

    def update_mi(self, w_list, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        feature = self.featurizer(all_x)
        start = 0
        mi_loss = 0.0
        feature_list = []
        for i in range(3):
            feature_list.append(feature[start: start + len(minibatches[i][0])])
            start += len(minibatches[i][0])

        for i in range(3):
            mi_loss += kl_loss_function(w_list[(i + 1) % 3](feature_list[i]), w_list[(i + 2) % 3](feature_list[i]), 1)

        loss = F.cross_entropy(self.classifier(feature), all_y) + mi_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def prune_unstruct(self):
        for name, module in self.featurizer.named_modules():
            # prune 20% of connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.2)
            # prune 40% of connections in all linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0.4)

        print(dict(self.featurizer.named_buffers()).keys())  # to verify that all masks exist

    def predict(self, x):
        return self.network(x)

    def predict_feature(self, x):
        return self.featurizer(x)

    def predict_classifier(self, feature):
        return self.classifier(feature)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()


class EC(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(EC, self).__init__(input_shape, num_classes, num_domains,
                                 hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.latent_dim = 64
        self.input_dim = self.featurizer.n_outputs
        self.num_domains = num_domains
        del self.featurizer

        self.classifiers = [torch.nn.Sequential(nn.Linear(self.input_dim, self.latent_dim), nn.ReLU(True),
                                                nn.Linear(self.latent_dim, num_classes)).cuda() for i in
                            range(num_domains)]
        self.optimizer = [torch.optim.Adam(
            self.classifiers[i].parameters(),
            lr=1e-2,
            weight_decay=self.hparams['weight_decay']
        ) for i in range(num_domains)]

    def update_ec(self, minibatches, feature):
        features = feature.detach()
        start = 0
        for i in range(self.num_domains):
            loss = F.cross_entropy(self.classifiers[i](features[start: start + minibatches[i][1].size(0)]),
                                   minibatches[i][1])
            self.optimizer[i].zero_grad()
            loss.backward()
            self.optimizer[i].step()
            start += minibatches[i][1].size(0)

    def predict_envs(self, env, x):
        return self.classifiers[env](x)

    def predict(self, x):
        pass


class ARM(ERM):
    """ Adaptive Risk Minimization (ARM) """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        original_input_shape = input_shape
        input_shape = (1 + original_input_shape[0],) + original_input_shape[1:]
        super(ARM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.context_net = networks.ContextNet(original_input_shape)
        self.support_size = hparams['batch_size']

    def predict(self, x):
        batch_size, c, h, w = x.shape
        if batch_size % self.support_size == 0:
            meta_batch_size = batch_size // self.support_size
            support_size = self.support_size
        else:
            meta_batch_size, support_size = 1, batch_size
        context = self.context_net(x)
        context = context.reshape((meta_batch_size, support_size, 1, h, w))
        context = context.mean(dim=1)
        context = torch.repeat_interleave(context, repeats=support_size, dim=0)
        x = torch.cat([x, context], dim=1)
        return self.network(x)


class AbstractDANN(Algorithm):
    """Domain-Adversarial Neural Networks (abstract class)"""

    def __init__(self, input_shape, num_classes, num_domains,
                 hparams, conditional, class_balance):

        super(AbstractDANN, self).__init__(input_shape, num_classes, num_domains,
                                           hparams)

        self.register_buffer('update_count', torch.tensor([0]))
        self.conditional = conditional
        self.class_balance = class_balance

        # Algorithms
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.discriminator = networks.MLP(self.featurizer.n_outputs,
                                          num_domains, self.hparams)
        self.class_embeddings = nn.Embedding(num_classes,
                                             self.featurizer.n_outputs)

        # Optimizers
        self.disc_opt = torch.optim.Adam(
            (list(self.discriminator.parameters()) +
             list(self.class_embeddings.parameters())),
            lr=self.hparams["lr_d"],
            weight_decay=self.hparams['weight_decay_d'],
            betas=(self.hparams['beta1'], 0.9))

        self.gen_opt = torch.optim.Adam(
            (list(self.featurizer.parameters()) +
             list(self.classifier.parameters())),
            lr=self.hparams["lr_g"],
            weight_decay=self.hparams['weight_decay_g'],
            betas=(self.hparams['beta1'], 0.9))

    def update(self, minibatches):
        self.update_count += 1
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        all_z = self.featurizer(all_x)
        if self.conditional:
            disc_input = all_z + self.class_embeddings(all_y)
        else:
            disc_input = all_z
        disc_out = self.discriminator(disc_input)
        disc_labels = torch.cat([
            torch.full((x.shape[0],), i, dtype=torch.int64, device='cuda')
            for i, (x, y) in enumerate(minibatches)
        ])

        if self.class_balance:
            y_counts = F.one_hot(all_y).sum(dim=0)
            weights = 1. / (y_counts[all_y] * y_counts.shape[0]).float()
            disc_loss = F.cross_entropy(disc_out, disc_labels, reduction='none')
            disc_loss = (weights * disc_loss).sum()
        else:
            disc_loss = F.cross_entropy(disc_out, disc_labels)

        disc_softmax = F.softmax(disc_out, dim=1)
        input_grad = autograd.grad(disc_softmax[:, disc_labels].sum(),
                                   [disc_input], create_graph=True)[0]
        grad_penalty = (input_grad ** 2).sum(dim=1).mean(dim=0)
        disc_loss += self.hparams['grad_penalty'] * grad_penalty

        d_steps_per_g = self.hparams['d_steps_per_g_step']
        if (self.update_count.item() % (1 + d_steps_per_g) < d_steps_per_g):

            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            return {'disc_loss': disc_loss.item()}
        else:
            all_preds = self.classifier(all_z)
            classifier_loss = F.cross_entropy(all_preds, all_y)
            gen_loss = (classifier_loss +
                        (self.hparams['lambda'] * -disc_loss))
            self.disc_opt.zero_grad()
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            return {'gen_loss': gen_loss.item()}

    def predict(self, x):
        return self.classifier(self.featurizer(x))


class DANN(AbstractDANN):
    """Unconditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DANN, self).__init__(input_shape, num_classes, num_domains,
                                   hparams, conditional=False, class_balance=False)


class CDANN(AbstractDANN):
    """Conditional DANN"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CDANN, self).__init__(input_shape, num_classes, num_domains,
                                    hparams, conditional=True, class_balance=True)


class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                          hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        # domain number
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma'] * penalty)).backward()
        self.optimizer.step()
        self.scheduler.step()
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)
class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches):
        penalty_weight = (
            self.hparams['irm_lambda'] if self.update_count >= self.hparams['irm_penalty_anneal_iters'] else 0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters'] and self.hparams['opt'] == 'Adam' and  self.hparams['irm_lambda'] != 1:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(VREx, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams['sch_size'], gamma=0.1)

    def update(self, minibatches):
        if self.update_count >= self.hparams["vrex_penalty_anneal_iters"]:
            penalty_weight = self.hparams["vrex_lambda"]
        else:
            penalty_weight = 0.

        nll = 0.
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        losses = torch.zeros(len(minibatches))
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll = F.cross_entropy(logits, y)
            losses[i] = nll

        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        loss = (mean + penalty_weight * penalty)

        if self.update_count == self.hparams['vrex_penalty_anneal_iters'] and self.hparams['opt'] == 'Adam':
            # Reset Adam (like IRM), because it doesn't like the sharp jump in
            # gradient magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}


class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(GroupDRO, self).__init__(input_shape, num_classes, num_domains,
                                       hparams)
        self.register_buffer("q", torch.Tensor())

    def update(self, minibatches):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"

        if not len(self.q):
            self.q = torch.ones(len(minibatches)).to(device)

        losses = torch.zeros(len(minibatches)).to(device)

        for m in range(len(minibatches)):
            x, y = minibatches[m]
            losses[m] = F.cross_entropy(self.predict(x), y)
            self.q[m] *= (self.hparams["groupdro_eta"] * losses[m].data).exp()

        self.q /= self.q.sum()
        loss = torch.dot(losses, self.q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {'loss': loss.item()}

class MLDG(ERM):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MLDG, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)

    def update(self, minibatches, unlabeled=None):
        """
        Terms being computed:
            * Li = Loss(xi, yi, params)
            * Gi = Grad(Li, params)
            * Lj = Loss(xj, yj, Optimizer(params, grad(Li, params)))
            * Gj = Grad(Lj, params)
            * params = Optimizer(params, Grad(Li + beta * Lj, params))
            *        = Optimizer(params, Gi + beta * Gj)
        That is, when calling .step(), we want grads to be Gi + beta * Gj
        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        self.optimizer.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            # fine tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)
            if self.hparams["opt"] == 'Adam':
                inner_opt = torch.optim.Adam(
                    inner_net.parameters(),
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams['weight_decay']
                )
            else:
                inner_opt = torch.optim.SGD(
                    inner_net.parameters(),
                    lr=self.hparams["lr"],
                    momentum=0.9,
                    weight_decay=self.hparams['weight_decay']
                )

            inner_obj = F.cross_entropy(self.network(xi), yi)

            inner_opt.zero_grad()
            inner_obj.backward()
            inner_opt.step()

            # The network has now accumulated gradients Gi
            # The clone-network has now parameters P - lr * Gi
            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_(p_src.grad.data / num_mb)

            # `objective` is populated for reporting purposes
            objective += inner_obj.item()

            # this computes Gj on the clone-network
            loss_inner_j = F.cross_entropy(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                allow_unused=True)
            # `objective` is populated for reporting purposes
            objective += (self.hparams['mldg_beta'] * loss_inner_j).item()
            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_(
                        self.hparams['mldg_beta'] * g_j.data / num_mb)

            # The network has now accumulated gradients Gi + beta * Gj
            # Repeat for all train-test splits, do .step()

        objective /= len(minibatches)
        self.optimizer.step()
        self.scheduler.step()
        return {'loss': objective}


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = networks.Featurizer(input_shape, hparams)
        classifier = nn.Linear(featurizer.n_outputs, num_classes)
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)

class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = WholeFish(input_shape, num_classes, hparams)
        if self.hparams['opt'] == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        if self.hparams['opt'] == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        if self.hparams['opt'] == 'SGD':
            self.optimizer_inner = torch.optim.SGD(
                self.network_inner.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        if self.hparams['opt'] == 'Adam':
            self.optimizer_inner = torch.optim.Adam(
                self.network_inner.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class Swap(Algorithm):
    """
    Swap
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Swap, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.register_buffer("q", torch.Tensor())
        self.featurizer_new = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer_new.n_outputs, num_classes).cuda()
        # TODO: more comlicated networks
        self.clist = [nn.Linear(self.featurizer_new.n_outputs, num_classes).cuda() for i in range(4)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
            # I think we shouldn't use momentum in this case
            # momentum=0.9,
            weight_decay=self.hparams['weight_decay']
        ) for i in range(4)]
        self.slist = [torch.optim.lr_scheduler.StepLR(self.olist[i], step_size=self.hparams['sch_size'], gamma=1) for
                      i in range(4)]
        if self.hparams['opt'] == 'SGD':
            self.optimizer_f = torch.optim.SGD(
                self.featurizer_new.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
        elif self.hparams['opt'] == 'Adam':
            self.optimizer_f = torch.optim.Adam(
                self.featurizer_new.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )
        self.scheduler_f = torch.optim.lr_scheduler.StepLR(self.optimizer_f, step_size=self.hparams['sch_size'],
                                                           gamma=0.1)
        self.scheduler_c = torch.optim.lr_scheduler.StepLR(self.optimizer_c, step_size=self.hparams['sch_size'],
                                                           gamma=0.1)
        self.marker_1 = 0.
        self.marker_2 = 0.
        self.H_collect = torch.zeros(1280)
        self.normal_sum = 0.
        self.cnt = 1.
        self.loss_collect = []

    def update(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss_swap = 0.0
        loss_Q_sum = 0.0
        loss_P_0_sum = 0.0
        loss_cos_sum = 0.0
        normalize_sum = 0.0
        cos = nn.CosineSimilarity(dim=0, eps=1e-8)
        # updating featurizer
        if self.update_count >= self.hparams['iters']:
            self.featurizer_new.eval()
            all_feature = self.featurizer_new(all_x).detach()
            for i in range(self.hparams['n']):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            self.featurizer_new.train()
            all_feature = self.featurizer_new(all_x)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)
                P_0 = random.sample(sample_list, 1)[0]
                # calculate the swapping loss and product of gradient
                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                loss_Q_sum += loss_Q.item()
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)
                # def haha(a):
                #     a = a.view(10, 128)
                #     x = F.linear(feature_split[Q], a, self.clist[Q].bias.view(-1))
                #     loss = F.cross_entropy(x, y_split[Q])
                #     return loss
                # H = grad_fun.hessian(haha, self.clist[Q].weight.view(-1)).detach()
                # I = torch.eye(H.size(0)).cuda()
                # H = H * I
                # H_inv = torch.inverse(H)
                # #     H_inv_eig = torch.eig(H_inv, eigenvectors=False)[0][:,0]
                # #     H_inv_eig = torch.sort(H_inv_eig,descending=True)[0]
                # #     self.H_collect += H_inv_eig.detach().cpu().numpy()
                # #     self.cnt += 1
                # # #print("eig values:", H_inv_eig)
                # # #print("evaluating ================= >")
                # # #a = torch.sort(H_inv.reshape(H_inv.size(0)**2),descending=True)[0][:H.size(0)]
                # # self.H_collect += H_inv_eig.detach().cpu().numpy()
                # #np.set_printoptions(precision=2,threshold=np.inf)
                # #print(self.H_collect.cpu().numpy()/self.update_count)
                # print(f"{H_inv.size(0)}: Trace mean of Hessian inverse:{torch.trace(H_inv)/H_inv.size(0)}")

                loss_P_0 = F.cross_entropy(self.clist[Q](feature_split[P_0]), y_split[P_0])
                loss_P_0_sum += loss_P_0.item()
                grad_P_0 = autograd.grad(loss_P_0, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)
                vec_grad_P_0 = nn.utils.parameters_to_vector(grad_P_0)

                #H trace avg
                # logit = F.softmax(self.clist[Q](feature_split[Q]), 1)
                # weights = (1 - torch.sum(logit ** 2, 1)).unsqueeze(1)
                # normalize = (feature_split[Q] @ feature_split[Q].transpose(1, 0)) * weights / (logit.size(1))
                # I = torch.eye(normalize.size(0)).cuda()
                # normalize_ = (normalize * I).sum() / (feature_split[Q].size(1) * feature_split[Q].size(0))
                # normalize_ = 1/normalize_

                ### H_inv trace avg
                logit = F.softmax(self.clist[Q](feature_split[Q]), 1)
                weights = logit - logit ** 2
                weights = weights.view(len(logit),1,logit.size(1))
                s = (feature_split[Q] ** 2).unsqueeze(2)
                s = 1/((s * weights).sum(0))
                # print(s, torch.mean(s))
                normalize_ = torch.mean(s) * len(logit)
                #normalize_ = torch.clamp(normalize_, 0, 1000)
                #print(f"normalize:{normalize}")
                #print(f"step:{self.update_count}, normalize1:{1/normalize_1}, normalize2:{normalize_}")
                # if self.update_count % 100 == 0 and self.update_count > 0:
                    # H = self.H_collect.cpu().numpy()
                    # print(f"normalized constant:{ self.normal_sum/self.cnt, H /self.cnt, (H /self.cnt).mean()}")
                    # my_dict = {
                    #     'normalize' : self.normal_sum/self.cnt,
                    #     'H_eig': H /self.cnt
                    # }
                    # np.save('hessian',my_dict)
                # loss_swap += loss_P_0 - (
                #             vec_grad_P_0.detach() @ H_inv.detach() @ vec_grad_Q)
                loss_swap += loss_P_0 - self.hparams['cos_lambda'] * (vec_grad_P_0.detach() @ vec_grad_Q) * normalize_.detach()
            loss_swap /= len(minibatches)
            loss_Q_sum /= len(minibatches)
            loss_P_0_sum /= len(minibatches)
            loss_cos_sum = loss_P_0_sum - loss_swap.item()
            normalize_sum = normalize_.item()
            # self.loss_collect.append(loss_swap.item())
            # if self.update_count % 100 == 0 and self.update_count > 0:
            #     my_collect = np.array(list(self.loss_collect))
            #     np.save("loss_collect_hessian", my_collect)

        self.featurizer_new.train()
        all_feature = self.featurizer_new(all_x)
        # updating original network

        loss = F.cross_entropy(self.classifier(all_feature), all_y)
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap * self.hparams['swap_lambda'])
        else:
            loss_swap = loss
        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()
        # updating scheduler
        self.scheduler_f.step()
        self.scheduler_c.step()
        loss_swap = loss_swap.item() - loss.item()
        self.update_count += 1

        return {'loss': loss.item(), 'loss_swap': loss_swap, 'Q_loss': loss_Q_sum, 'P_loss': loss_P_0_sum, 'cos_avg': loss_cos_sum,'normalize':normalize_sum}

    def predict(self, x):
        return self.classifier(self.featurizer_new(x))

    def train(self):
        self.featurizer_new.train()

    def eval(self):
        self.featurizer_new.eval()



class TRM(Algorithm):
    """
    Swap
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(TRM, self).__init__(input_shape, num_classes, num_domains,
                                          hparams)
        self.register_buffer('update_count', torch.tensor([0]))
        self.featurizer_new = networks.Featurizer(input_shape, self.hparams)
        self.classifier = nn.Linear(self.featurizer_new.n_outputs, num_classes).cuda()
        self.clist = [nn.Linear(self.featurizer_new.n_outputs, num_classes).cuda() for i in range(3)]
        self.olist = [torch.optim.SGD(
            self.clist[i].parameters(),
            lr=1e-1,
        ) for i in range(3)]

        if self.hparams['opt'] == 'SGD':
            self.optimizer_f = torch.optim.SGD(
                self.featurizer_new.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )
            self.optimizer_c = torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                weight_decay=self.hparams['weight_decay']
            )

        self.scheduler_f = torch.optim.lr_scheduler.StepLR(self.optimizer_f, step_size=self.hparams['sch_size'],gamma=0.1)
        self.scheduler_c = torch.optim.lr_scheduler.StepLR(self.optimizer_c, step_size=self.hparams['sch_size'], gamma=0.1)
        # initial weights
        self.alpha = torch.ones((num_domains, num_domains)).cuda() - torch.eye(num_domains).cuda()

    def update(self, minibatches):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss_swap = 0.0
        loss_Q_sum = 0.0
        loss_P_sum_collect = 0.0
        loss_cos_sum = 0.0
        normalize_sum = 0.0
        cos = nn.CosineSimilarity(dim=0, eps=1e-8)
        # updating featurizer
        if self.update_count >= self.hparams['iters']:
            self.alpha /= self.alpha.sum(1, keepdim=True)
            # if self.update_count % 300 == 0 and self.update_count>0:
            #     print("alpha:", self.alpha.data)
            self.featurizer_new.eval()
            all_feature = self.featurizer_new(all_x).detach()
            for i in range(self.hparams['n']):
                all_logits_idx = 0
                loss_erm = 0.
                for j, (x, y) in enumerate(minibatches):
                    # j-th domain
                    feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                    all_logits_idx += x.shape[0]
                    loss_erm += F.cross_entropy(self.clist[j](feature), y)
                for opt in self.olist:
                    opt.zero_grad()
                loss_erm.backward()
                for opt in self.olist:
                    opt.step()

            self.featurizer_new.train()
            all_feature = self.featurizer_new(all_x)
            feature_split = list()
            y_split = list()
            all_logits_idx = 0
            for i, (x, y) in enumerate(minibatches):
                feature = all_feature[all_logits_idx:all_logits_idx + x.shape[0]]
                all_logits_idx += x.shape[0]
                feature_split.append(feature)
                y_split.append(y)

            for Q, (x, y) in enumerate(minibatches):
                sample_list = list(range(len(minibatches)))
                sample_list.remove(Q)
                # calculate the swapping loss and product of gradient
                loss_Q = F.cross_entropy(self.clist[Q](feature_split[Q]), y_split[Q])
                loss_Q_sum += loss_Q.item()
                grad_Q = autograd.grad(loss_Q, self.clist[Q].weight, create_graph=True)

                loss_P = [F.cross_entropy(self.clist[Q](feature_split[i]), y_split[i])*(self.alpha[Q, i].data.detach()) if i in sample_list else 0. for i in range(len(minibatches))]
                loss_P_sum = sum(loss_P)
                loss_P_sum_collect += loss_P_sum.item()
                grad_P = autograd.grad(loss_P_sum, self.clist[Q].weight, create_graph=True)
                vec_grad_Q = nn.utils.parameters_to_vector(grad_Q)
                vec_grad_P = nn.utils.parameters_to_vector(grad_P)

                logit = F.softmax(self.clist[Q](feature_split[Q]), 1)
                weights = logit - logit ** 2
                weights = weights.view(len(logit),1,logit.size(1))
                s = (feature_split[Q] ** 2).unsqueeze(2)
                s = 1/((s * weights).sum(0))
                normalize_ = torch.mean(s) * len(logit)
                normalize_ = torch.clamp(normalize_, 0, 1000)

                loss_swap += loss_P_sum - self.hparams['cos_lambda'] * (vec_grad_P.detach() @ vec_grad_Q) * (normalize_).detach()

                for i in sample_list:
                    self.alpha[Q, i] *= (self.hparams["groupdro_eta"] * loss_P[i].data).exp()
                self.alpha /= self.alpha.sum(1, keepdim=True)

            loss_swap /= len(minibatches)
            loss_Q_sum /= len(minibatches)
            loss_P_sum_collect /= len(minibatches)
            loss_cos_sum = loss_P_sum_collect - loss_swap.item()
            normalize_sum = normalize_.item()

        self.featurizer_new.train()
        all_feature = self.featurizer_new(all_x)
        # updating original network
        loss = F.cross_entropy(self.classifier(all_feature), all_y)
        self.optimizer_c.zero_grad()
        self.optimizer_f.zero_grad()
        if self.update_count >= self.hparams['iters']:
            loss_swap = (loss + loss_swap * self.hparams['trm_lambda']) / self.hparams['trm_lambda']
        else:
            loss_swap = loss

        loss_swap.backward()
        self.optimizer_f.step()
        self.optimizer_c.step()

        # updating scheduler
        self.scheduler_f.step()
        self.scheduler_c.step()
        loss_swap = loss_swap.item() - loss.item()
        self.update_count += 1

        return {'loss': loss.item(), 'loss_swap': loss_swap, 'Q_loss': loss_Q_sum, 'P_loss': loss_P_sum_collect,
                'cos_avg': loss_cos_sum, 'normalize': normalize_sum}

    def predict(self, x):
        return self.classifier(self.featurizer_new(x))

    def train(self):
        self.featurizer_new.train()

    def eval(self):
        self.featurizer_new.eval()