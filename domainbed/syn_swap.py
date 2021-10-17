import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
import random
import syn_model
from syn_data import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--a', type=str, default='swap')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--n_train_envs', type=int, default=5)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--c_scale', type=float, default=1.0)
parser.add_argument('--e_scale', type=float, default=1.0)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--e_bias', type=float, default=1.0)
parser.add_argument('--c_bias', type=float, default=1.0)
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
c_scale = args.c_scale
e_scale = args.e_scale
c_mean = torch.tensor([args.c_bias])
e_mean = torch.randn((n_train_envs, dim)) * e_scale + args.e_bias
e_mean = torch.sqrt(1/ torch.var(e_mean)) * e_mean
e_mean = 1 - torch.mean(e_mean) + e_mean
e_mean_total = torch.mean(e_mean)
print("E: Mean:{}, Variance:{}".format(torch.mean(e_mean), torch.var(e_mean)))
model = syn_model.mlp_linear(dim=2 * dim).cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

train_datasets = [Gaussain_data(dim, dim, c_mean=c_mean, e_mean=e_mean[i], number_samples=10000, filters=args.filters,
                                fractions=args.fractions) for i in range(n_train_envs)]
train_loaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=256, shuffle=True) for i in
                 range(n_train_envs)]
print("init model weight:", model.fc.weight.data)


def logistic_loss(w, y):
    prod = w * (2 * y - 1)
    return - torch.log(F.sigmoid(prod)).mean()


def irm_penalty(w, z_0, z_1, data, env0, env1):

    loss_1 = logistic_loss(w * z_0, data[env0][1]) + logistic_loss(w * z_1, data[env1][1])
    grad_1 = autograd.grad(loss_1, w, create_graph=True)[0]
    grad_1 = nn.utils.parameters_to_vector(grad_1)
    result = torch.sum(grad_1 * grad_1)
    return result
best_loss = 1000.0
best_ratio = 0.
l2_collect = []
def train(epoch):
    global c_mean
    global e_mean
    global e_mean_total
    global best_loss
    global best_ratio
    global l2_collect
    model.train()
    correct = 0.0
    loss_sum = 0.0
    total = 0.1
    cnt = 0.
    k = 0.0
    l2_dis = 0.
    l2_cnt = 0
    cnt_k = 1.
    c_mean = c_mean.cuda()
    e_mean = e_mean.cuda()
    e_mean_total = e_mean_total.cuda()
    train_minibatches_iterator = zip(*train_loaders)
    for data in tqdm(train_minibatches_iterator):
        sample_list = list(range(len(data)))
        env0 = random.sample(sample_list, 1)[0]
        sample_list.remove(env0)
        env1 = random.sample(sample_list, 1)[0]
        opt.zero_grad()
        for i in [env0,env1]:
            for j in range(2):
                data[i][j] = data[i][j].cuda()

        z_0 = model(data[env0][0]).squeeze(1)
        z_1 = model(data[env1][0]).squeeze(1)
        w0 = 2 * (model.fc.weight[0][0] * c_mean + model.fc.weight[0][1] * e_mean[env0]) / (
                model.fc.weight[0][0] ** 2 + model.fc.weight[0][1] ** 2)
        w1 = 2 * (model.fc.weight[0][0] * c_mean + model.fc.weight[0][1] * e_mean[env1]) / (
                model.fc.weight[0][0] ** 2 + model.fc.weight[0][1] ** 2)
        for i in range(len(data)):
            for j in range(i+1, len(data)):
                wi = 2 * (model.fc.weight[0][0] * c_mean + model.fc.weight[0][1] * e_mean[i]) / (
                        model.fc.weight[0][0] ** 2 + model.fc.weight[0][1] ** 2)
                wj = 2 * (model.fc.weight[0][0] * c_mean + model.fc.weight[0][1] * e_mean[j]) / (
                        model.fc.weight[0][0] ** 2 + model.fc.weight[0][1] ** 2)
                l2_dis += float(torch.sum((wi-wj) ** 2).item())
                l2_cnt += 1

        l2_collect.append(l2_dis / l2_cnt)
        w = 2 * (model.fc.weight[0][0] * c_mean + model.fc.weight[0][1] * e_mean_total) / (
                model.fc.weight[0][0] ** 2 + model.fc.weight[0][1] ** 2)
        if args.a == 'swap':
            loss = logistic_loss(w1 * z_0, data[env0][1]) + logistic_loss(w0 * z_1, data[env1][1])
            correct += (((w0 * z_0) > 0).float().eq(data[env0][1]).float()).sum().item()
        elif args.a == 'erm':
            loss = logistic_loss(w * z_0, data[env0][1]) + logistic_loss(w * z_1, data[env1][1])
            correct += (((w * z_0) > 0).float().eq(data[env0][1]).float()).sum().item()
        elif args.a == 'irm':
            loss = logistic_loss(w * z_0, data[env0][1]) + logistic_loss(w * z_1, data[env1][1]) + irm_penalty(w,z_0,z_1,data,env0,env1)
            k += torch.sum((F.sigmoid(w*z_0) - data[env0][1])**2)
            cnt_k += len(data[env0][1])
            correct += (((w * z_0) > 0).float().eq(data[env0][1]).float()).sum().item()
        elif args.a == 'align':
            loss = logistic_loss(w * z_0, data[env0][1]) + logistic_loss(w * z_1, data[env1][1]) + args.lam * torch.mean((torch.mean(z_0) - torch.mean(z_1)) ** 2)
            correct += (((w * z_0) > 0).float().eq(data[env0][1]).float()).sum().item()
        total += len(data[env0][0])
        loss_sum += loss.item()
        loss.backward()
        opt.step()
        cnt += 1

        normalize = torch.norm((model.fc.weight.data),p=2)
        model.fc.weight.data[0][0] /= normalize
        model.fc.weight.data[0][1] /= normalize
    if loss_sum < best_loss:
        best_loss = loss_sum
        best_ratio = abs(model.fc.weight[0][1] / model.fc.weight[0][0])
    print("model weights:", model.fc.weight[0][0], model.fc.weight[0][1])
    print("weights ratio:", abs(model.fc.weight[0][1] / model.fc.weight[0][0]))
    print("loss:", loss_sum/cnt)
    if args.a == 'irm':
        print("avg k:", k/cnt_k)
    print("l2 dis:", l2_dis/l2_cnt)
    print("Training accuracy:{}".format(correct / total))


if __name__ == '__main__':
    for iters in range(epoch):
        train(iters)
    np.save("l2_collect5", np.array(l2_collect))
    print(f"Best acc:{best_loss}, best ratio:{best_ratio}")
