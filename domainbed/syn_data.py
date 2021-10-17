import torch
import torch.utils


class Gaussain_data(torch.utils.data.Dataset):
    def __init__(self, c_dimension, e_dimension, c_mean, e_mean, number_samples, filters=False, fractions=1.):
        self.cd = c_dimension
        self.ed = e_dimension
        self.c_mean = c_mean
        self.e_mean = e_mean
        self.num_samples = number_samples
        self.filters = filters
        self.fractions = fractions
        self.w = torch.empty(self.cd + self.ed, self.cd + self.ed)
        torch.nn.init.orthogonal_(self.w)
        print(self.w @ self.w.transpose(1, 0))
        self.construct_samples()

        # TODO: add orthogonal projections maybe?

    def construct_samples(self):
        # constructing dataset
        self.label = torch.zeros(self.num_samples)
        self.label[self.num_samples // 2:] = self.label[self.num_samples // 2:] + 1
        self.c = torch.randn(size=(self.num_samples, self.cd)) + self.c_mean.repeat(self.num_samples,1) * ( 2 * self.label.unsqueeze(1) - 1)
        self.e = torch.randn(size=(self.num_samples, self.ed)) + self.e_mean.unsqueeze(0).repeat(self.num_samples,1) * (2 * self.label.unsqueeze(1) - 1)

        self.data = torch.cat([self.c, self.e], dim=1)
        self.label = self.label.long()
        if self.filters:
            distance = ((self.c @ self.c_mean) ** 2)
            sort_list = torch.argsort(distance,descending=True)
            self.data = self.data[sort_list]
            self.label = self.label[sort_list]
            self.data = self.data[: int(self.fractions * self.num_samples)]
            self.label = self.label[: int(self.fractions * self.num_samples)]

    def cal_avg_margin(self):
        return torch.abs(self.data[:,:self.cd] @ self.c_mean).mean()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]