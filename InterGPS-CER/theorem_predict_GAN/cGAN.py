import torch
import torch.nn as nn
import json

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
            nn.Conv1d(1, 10, 5, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(10, 20, 5, 4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(20, 30, 5, 2, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(30 * 1, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.disc(x)
        x = x.view(-1, 30 * 1)
        x = self.fc(x)
        return x

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gene = nn.Sequential(
            nn.Conv1d(1, 10, 5, 2, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(10, 20, 5, 2, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(20, 30, 5, 1, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(30 * 10, 150),
            nn.LeakyReLU(),
            nn.Linear(150, 30),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.gene(x)
        x = x.view(-1, 30 * 10)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    batch_size = 8
    gepoch = 10
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
    G_optimizer = torch.optim.Adam(G_net.parameters(), lr=0.005)
    D_optimizer = torch.optim.Adam(D_net.parameters(), lr=0.001)
    criterion = nn.BCELoss().to(device)

    d_loss_real_total, d_loss_fake_total, g_loss_total = .0, .0, .0
    for epoch in range(3):
        for batch_num, (pb_id, _, problem, sol) in enumerate(trainloader):
            D_real_label = torch.ones(problem.size(0), 1).to(device)
            D_real_label[0, 0] = 0.9
            D_fake_label = torch.zeros(problem.size(0), 1).to(device)
            D_fake_label[0, 0] = 0.1
            sol = sol.to(device)
            problem_d = problem.to(device)

            labels_and_seqs = torch.cat((sol, problem_d), dim=1)
            real_out = D_net(labels_and_seqs[:, None, :])
            d_loss_real = criterion(real_out, D_real_label)
            z = torch.randn(problem_d.size(0), z_dimension).to(device)
            z = torch.cat((z, problem_d), dim=1)
            fake_seq = G_net(z[:, None, :])
            fake_seq = fake_seq[:, None, :]

            problem_d = problem_d[:, None, :]
            fake_seq = torch.cat((fake_seq, problem_d), dim=2)
            fake_out = D_net(fake_seq)
            d_loss_fake = criterion(fake_out, D_fake_label)

            d_loss = d_loss_real + d_loss_fake
            d_loss_real_total += d_loss_real.item()
            d_loss_fake_total += d_loss_fake.item()
            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()

            for j in range(gepoch):
                G_real_label = torch.ones(problem.size(0), 1).to(device)
                G_fake_label = torch.zeros(problem.size(0), 1).to(device)
                problem_g = problem.to(device)
                gz = torch.randn(problem_g.size(0), z_dimension).to(device)
                gz = torch.cat((gz, problem_g), dim=1)
                fake_seq = G_net(gz[:, None, :])
                fake_seq = fake_seq[:, None, :]
                problem_g = problem_g[:, None, :]
                fake_seq = torch.cat((fake_seq, problem_g), dim=2)
                output = D_net(fake_seq)

                g_loss = criterion(output, G_real_label)

                G_optimizer.zero_grad()
                g_loss.backward()
                G_optimizer.step()
                g_loss_total += g_loss.item()
            if batch_num % print_freq == print_freq - 1:
                print("{: 4d} d_loss_real:{:.10f} d_loss_fake:{:.10f} g_loss:{:.10f}".format(
                    batch_num, d_loss_real.item() / print_freq, d_loss_fake.item() / print_freq,
                               g_loss_total / max(gepoch * print_freq, 1)))
                d_loss_real_total, d_loss_fake_total, g_loss_total = .0, .0, .0

    gene_seq = {}
    with open(LABELS_PATH, 'r') as f:
        labels_onehot = json.load(f)
    test_one_hot = {}
    for pid in range(2402, 3002):
        test_one_hot[pid] = {'id': pid, 'one_hot': []}
        test_one_hot[pid]["one_hot"] = labels_onehot[str(pid)]['one_hot']
        gene_seq[pid] = {'id': pid, 'seq': []}
        tz = torch.randn(z_dimension).to(device)
        tensortest = torch.tensor(test_one_hot[int(pid)]["one_hot"]).to(device)
        tz = torch.cat((tz, tensortest), dim=0)
        tz = torch.unsqueeze(tz, dim=0)
        tz = torch.unsqueeze(tz, dim=0)
        fake_seq = G_net(tz.clone().detach())
        fake_seq = fake_seq.detach().cpu().numpy()
        fake_seq = fake_seq.tolist()
        gene_seq[pid]['seq'] = fake_seq

    GENE_PATH = '../gene_seq.json'
    with open(GENE_PATH, 'w') as f:
        json.dump(gene_seq, f, indent=2, separators=(',', ': '))


