import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd

import syn_model
from syn_data import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--dim', type=int, default=10)
parser.add_argument('--n_train_envs', type=int, default=3)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--fractions', type=float, default=1.0)
parser.add_argument('--filters', action='store_true')
parser.add_argument('--select', action='store_true')
args = parser.parse_args()
print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
epoch = args.epoch
dim = args.dim
n_train_envs = args.n_train_envs
scale = args.scale
c_mean = torch.ones(dim)
e_mean = torch.randn((n_train_envs + 1, dim)) * scale
model = syn_model.mlp(dim=2 * dim).cuda()
domain_mlp = [syn_model.mlp(dim=2 * dim).cuda() for i in range(n_train_envs)]
domain_opt = [torch.optim.SGD(domain_mlp[i].parameters(), lr=0.1) for i in range(n_train_envs)]
opt = torch.optim.SGD(model.parameters(), lr=0.1)

train_datasets = [Gaussain_data(dim, dim, c_mean=c_mean, e_mean=e_mean[i], number_samples=100000, filters=args.filters,
                                fractions=args.fractions) for i in range(n_train_envs)]
train_loaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=256, shuffle=True) for i in
                 range(n_train_envs)]
eval_loaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=256, shuffle=False) for i in
                range(n_train_envs)]
test_datasets = Gaussain_data(dim, dim, c_mean=c_mean, e_mean=e_mean[n_train_envs], number_samples=100000)
test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=256, shuffle=False)


def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = F.cross_entropy(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def train(epoch):
    model.train()
    for i in range(n_train_envs):
        domain_mlp[i].train()
    correct = 0.0
    total = 0.1
    train_loaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=256, shuffle=True) for i in
                     range(n_train_envs)]
    train_minibatches_iterator = zip(*train_loaders)
    for data in tqdm(train_minibatches_iterator):
        all_x = torch.cat([x for x, y in data]).cuda()
        all_y = torch.cat([y for x, y in data]).cuda()

        for i, (x, y) in enumerate(data):
            x = x.cuda()
            y = y.cuda()
            domain_opt[i].zero_grad()
            domain_loss = F.cross_entropy(domain_mlp[i](x), y)
            domain_loss.backward()
            domain_opt[i].step()

        if epoch >= 1:
            opt.zero_grad()
            y_hat = model(all_x)
            loss = F.cross_entropy(y_hat, all_y)
            loss.backward()
            opt.step()

            correct += (y_hat.argmax(1).eq(all_y).float()).sum().item()
            total += len(all_y)
    print("Training accuracy:{}".format(correct / total))


def eval_loss():
    model.train()
    correct = 0.0
    total = 0.0
    train_minibatches_iterator = zip(*eval_loaders)
    loss_collect = []
    for idx, data in enumerate(train_minibatches_iterator):
        all_x = torch.cat([x for x, y in data]).cuda()
        all_y = torch.cat([y for x, y in data]).cuda()

        opt.zero_grad()
        y_hat = model(all_x)
        loss = F.cross_entropy(y_hat, all_y)
        loss.backward()
        opt.step()
        loss_collect.append(loss.item())

        correct += (y_hat.argmax(1).eq(all_y).float()).sum().item()
        total += len(all_y)
    loss_collect = np.array(loss_collect)
    print("First half:{}, second half:{}".format(loss_collect[:loss_collect.shape[0] // 2].sum(),
                                                 loss_collect[loss_collect.shape[0] // 2:].sum()))


def js_div(a, b):
    p1 = F.softmax(a, dim=1)
    p2 = F.softmax(b, dim=1)
    p3 = (p1 + p2) / 2
    return torch.sum(p1 * torch.log(p1 / p3), dim=1) + torch.sum(p2 * torch.log(p2 / p3), dim=1)


def l1(a,b):
    p1 = F.softmax(a, dim=1)
    p2 = F.softmax(b, dim=1)
    return torch.abs(p1-p2).sum(1)


def disagree(env1, env2, env3):
    print("Before selection: Avg margin of env:{}, :{}".format(env3, train_datasets[env3].cal_avg_margin()))
    for i in range(n_train_envs):
        domain_mlp[i].eval()
    loss_collect = []
    correct_1 = 0.
    correct_2 = 0.
    cnt_1 = 0.
    cnt_2 = 0.
    loader = torch.utils.data.DataLoader(dataset=train_datasets[env3], batch_size=256, shuffle=False)
    for idx, (data, label) in enumerate(loader):
        data = data.cuda()
        y1 = domain_mlp[env1](data)
        y2 = domain_mlp[env2](data)
        if idx <= len(loader) / 2:
            correct_1 += (y1.argmax(1).eq(y2.argmax(1)).float()).sum().item()
            cnt_1 += len(data)
        else:
            correct_2 += (y1.argmax(1).eq(y2.argmax(1)).float()).sum().item()
            cnt_2 += len(data)
        # loss = js_div(y1, y2)
        loss = F.l1_loss(y1, y2)
        loss_collect += list(loss)
    loss_collect = torch.tensor(loss_collect)
    sort_list = torch.argsort(loss_collect)
    olen = len(train_datasets[env3])
    train_datasets[env3].data = train_datasets[env3].data[sort_list]
    train_datasets[env3].label = train_datasets[env3].label[sort_list]
    train_datasets[env3].data = train_datasets[env3].data[: int(olen * 0.5)]
    train_datasets[env3].label = train_datasets[env3].label[: int(olen * 0.5)]
    print("First halt acc:{}, second half acc:{}".format(correct_1 / cnt_1, correct_2 / cnt_2))
    print("First half:{}, second half:{}".format(loss_collect[:loss_collect.shape[0] // 2].sum(),
                                                 loss_collect[loss_collect.shape[0] // 2:].sum()))
    print("After selection: Avg margin of env:{}, :{}".format(env3, train_datasets[env3].cal_avg_margin()))


def test(epoch):
    model.eval()
    correct = 0.0
    total = 0.0
    for data, label in tqdm(test_loader):
        data = data.cuda()
        label = label.cuda()

        y_hat = model(data)
        loss = F.cross_entropy(y_hat, label)

        correct += (y_hat.argmax(1).eq(label).float()).sum().item()
        total += len(label)
    print("Epoch:{}, Test accuracy:{}".format(epoch, correct / total))


if __name__ == '__main__':

    for iters in range(epoch):
        train(iters)
         #test(iters)
        # if args.select:
        #     disagree(0, 1, 2)
        #     disagree(1, 2, 0)
        #     disagree(0, 2, 1)

#         if iters == 0:
#             disagree(0,1,2)
#             disagree(1,2,0)
#             disagree(0,2,1)
#             exit()