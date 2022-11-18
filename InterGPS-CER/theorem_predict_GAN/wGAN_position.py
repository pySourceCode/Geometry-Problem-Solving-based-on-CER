import torch
import torch.nn as nn
import json
import numpy as np
import math
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Data(torch.utils.data.Dataset):
    def __init__(self, DATA_PATH, LABELS_PATH):
        super(Data).__init__()
        prob2sol = json.load(open(DATA_PATH, 'r'))
        labels_onehot = json.load(open(LABELS_PATH, 'r'))
        self.solutions = []
        self.problems = []
        self.ind = []

        count_problem = 0
        count_solution = 0
        for pid in prob2sol.keys():
            probelm_seq = labels_onehot[pid]["one_hot"]
            probelm_seq = torch.tensor(probelm_seq).float()
            self.problems.append(probelm_seq)

            for solution in prob2sol[pid]['seqs']:
                solution = self.padding(solution, 30)
                self.solutions.append(solution)
                self.ind.append((count_problem, count_solution))
                count_solution += 1
            count_problem += 1

    @staticmethod
    def padding(lst, length):
        zeros = [0] * (length - len(lst))
        return torch.tensor(lst + zeros).float()

    def __getitem__(self, ind):
        problem_ind, solution_ind = self.ind[ind]
        pidproblem = self.problems[problem_ind]
        pidsolution = self.solutions[solution_ind]
        return problem_ind, solution_ind, pidproblem, pidsolution

    def __len__(self):
        return len(self.ind)


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv1d(1, 16, 5, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(16, 64, 5, 4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(64, 256, 5, 2, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 1, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            # nn.Sigmoid()
        )
        self.pe = PositionalEncoding(d_model=40)

    def forward(self, x):
        x = self.pe(x)
        x = self.disc(x)
        x = x.view(-1, 256 * 1)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):  # 修改参数

    def __init__(self, d_model, max_len=1):  # d_model = 40
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gene = nn.Sequential(
            nn.Conv1d(1, 8, 5, 2, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(8, 64, 5, 2, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(64, 256, 5, 2, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(256, 1024, 5, 5, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 30),
        )
        self.pe = PositionalEncoding(d_model=40)

    def forward(self, x):
        x = self.pe(x)
        x = self.gene(x)
        x = x.view(-1, 1024*1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    batch_size = 8
    # gepoch = 10
    print_freq = 20
    z_dimension = 30

    DATA_PATH = '../pred_seqs_train_merged_correct.json'
    LABELS_PATH = '../text_one_hot.json'
    trainset = Data(DATA_PATH, LABELS_PATH)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    D_net = discriminator()
    G_net = generator()
    D_net = D_net.to(device)
    G_net = G_net.to(device)
    G_optimizer = torch.optim.RMSprop(G_net.parameters(), lr=0.01)
    D_optimizer = torch.optim.RMSprop(D_net.parameters(), lr=0.001)
    criterion = nn.BCELoss().to(device)

    d_loss_real_total, d_loss_fake_total, g_loss_total = .0, .0, .0
    index = 0
    for epoch in range(10):
        for batch_num, (pb_id, _, problem, sol) in enumerate(trainloader):
            index += 1
            sol = sol.to(device)
            problem_d = problem.to(device)

            D_optimizer.zero_grad()
            labels_and_seqs = torch.cat((sol, problem_d), dim=1)
            real_out = D_net(labels_and_seqs[:, None, :])
            d_loss_real = (real_out * (-1)).mean()

            z = Variable(torch.Tensor(np.random.normal(0, 1, (problem_d.size(0), z_dimension)))).to(device)
            z = torch.cat((z, problem_d), dim=1)
            fake_seq = G_net(z[:, None, :]).detach()
            fake_seq = fake_seq[:, None, :]
            problem_d = problem_d[:, None, :]
            fake_seq = torch.cat((fake_seq, problem_d), dim=2)
            fake_out = D_net(fake_seq)
            d_loss_fake = fake_out.mean()

            d_loss = d_loss_real + d_loss_fake
            d_loss_real_total += d_loss_real.item()
            d_loss_fake_total += d_loss_fake.item()
            d_loss.backward()

            for p in D_net.parameters():
                p.data.clamp_(-0.01, 0.01)
            D_optimizer.step()

            if index % 5 == 0:
                G_optimizer.zero_grad()
                problem_g = problem.to(device)
                gz = torch.randn(problem_g.size(0), z_dimension).to(device)
                gz = torch.cat((gz, problem_g), dim=1)
                fake_seq = G_net(gz[:, None, :])
                fake_seq = fake_seq[:, None, :]
                problem_g = problem_g[:, None, :]
                fake_seq = torch.cat((fake_seq, problem_g), dim=2)
                output = D_net(fake_seq)
                g_loss = -torch.mean(output)
                g_loss.backward()
                G_optimizer.step()
                g_loss_total += g_loss.item()

            if batch_num % print_freq == print_freq - 1:
                print("{: 4d} d_loss_real:{:.10f} d_loss_fake:{:.10f} g_loss:{:.10f}".format(
                    batch_num, d_loss_real.item() / print_freq, d_loss_fake.item() / print_freq,
                               g_loss_total / (print_freq/5)))
                d_loss_real_total, d_loss_fake_total, g_loss_total = .0, .0, .0