import torch
import torch.nn as nn
import torch.nn.functional as F


class SSRNet(nn.Module):
    def __init__(self):
        super(SSRNet, self).__init__()
        self.stream1_k1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        self.stream2_k1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.stream1_k2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        self.stream2_k2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )
        self.stream1_k3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.stream2_k3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2),
        )
        self.s1_pre1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(31)
        )
        self.s2_pre1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(31)
        )

        self.s1_pre2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(14)
        )
        self.s2_pre2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(14)
        )
        self.s1_pre3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.s2_pre3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        self.delta_k1 = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.delta_k2 = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.delta_k3 = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()
        )
        self.vec_p1 = nn.Sequential(
            nn.Linear(16, 3),
            nn.ReLU()
        )
        self.vec_p2 = nn.Sequential(
            nn.Linear(16, 3),
            nn.ReLU()

        )
        self.vec_p3 = nn.Sequential(
            nn.Linear(16, 3),
            nn.ReLU()
        )
        self.eta1 = nn.Sequential(
            nn.Linear(16, 3),
            nn.Tanh()
        )
        self.eta2 = nn.Sequential(
            nn.Linear(16, 3),
            nn.Tanh()
        )
        self.eta3 = nn.Sequential(
            nn.Linear(16, 3),
            nn.Tanh()
        )
        self.s1_dr1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.s2_dr1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.s1_dr2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.s2_dr2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.s1_dr3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.s2_dr3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.idx1 = torch.tensor(range(3), dtype=torch.float)
        self.idx2 = torch.tensor(range(3), dtype=torch.float)
        self.idx3 = torch.tensor(range(3), dtype=torch.float)

    def forward(self, x):
        t1 = self.stream1_k1(x)
        # torch.Size([2, 32, 31, 31])
        t2 = self.stream1_k2(t1)
        t3 = self.stream1_k3(t2)

        t4 = self.stream2_k1(x)
        t5 = self.stream2_k2(t4)
        t6 = self.stream2_k3(t5)

        # print(self.s1_pre1(t1).shape)
        # print(self.s2_pre1(t4).shape)
        # torch.Size([2, 32, 15, 15])
        # torch.Size([2, 32, 15, 15])
        # print((self.s1_pre1(t1) * self.s2_pre1(t4)).shape)
        # torch.Size([2, 32, 15, 15])

        delta_k_1 = self.delta_k1((self.s1_pre1(t1) * self.s2_pre1(t4)).view(-1, 32))
        delta_k_2 = self.delta_k2((self.s1_pre2(t2) * self.s2_pre2(t5)).view(-1, 32))
        delta_k_3 = self.delta_k3((self.s1_pre3(t3) * self.s2_pre3(t6)).view(-1, 32))
        print('delta:', delta_k_1.shape, delta_k_2.shape, delta_k_3.shape)
        # print(self.s1_pre1(t1).shape)
        # print(self.s1_dr1(self.s1_pre1(t1).view(-1,32)).shape)
        vec_p_1 = self.vec_p1(self.s1_dr1(self.s1_pre1(t1).view(-1, 32)) * self.s2_dr1(self.s2_pre1(t4).view(-1, 32)))
        # print(vec_p_1.shape)
        vec_p_2 = self.vec_p2(self.s1_dr2(self.s1_pre2(t2).view(-1, 32)) * self.s2_dr2(self.s2_pre2(t5).view(-1, 32)))
        vec_p_3 = self.vec_p3(self.s1_dr3(self.s1_pre3(t3).view(-1, 32)) * self.s2_dr3(self.s2_pre3(t6).view(-1, 32)))
        print('vec:', vec_p_1.shape, vec_p_2.shape, vec_p_3.shape)

        eta1 = self.eta1(self.s1_dr1(self.s1_pre1(t1).view(-1, 32)) * self.s2_dr1(self.s2_pre1(t4).view(-1, 32)))
        eta2 = self.eta2(self.s1_dr2(self.s1_pre2(t2).view(-1, 32)) * self.s2_dr2(self.s2_pre2(t5).view(-1, 32)))
        eta3 = self.eta3(self.s1_dr3(self.s1_pre3(t3).view(-1, 32)) * self.s2_dr3(self.s2_pre3(t6).view(-1, 32)))
        print('eta:', eta1.shape, eta2.shape, eta3.shape)

        # 可以理解为 30 + 5 + 0.5 = 35.5 这么去从粗到细预测
        '''
        delta: [N, 1]
        vec: [N, 3]
        eta: [N, 3]
        
        sum{ ((1, 2, 3) + eta) * vec_p  /  ( (101/3) * (1+delta_k) }
        
        '''


        output1 = torch.sum((self.idx1.view(1, 3) + eta1) * vec_p_1 / 3 / (1 + delta_k_1), dim=1)

        output2 = torch.sum((self.idx2.view(1, 3) + eta2) * vec_p_2 / 3 / (1 + delta_k_1) / 3 / (1 + delta_k_2), dim=1)
        output3 = torch.sum(
            (self.idx3.view(1, 3) + eta3) * vec_p_3 / 3 / (1 + delta_k_1) / 3 / (1 + delta_k_2) / 3 / (1 + delta_k_3),
            dim=1)
        print(output1, output2, output2)
        return (output1 + output2 + output3) * 101


if __name__ == "__main__":
    from torchsummary import summary

    model = SSRNet()
    # summary(model, (3, 64, 64), device="cpu")
    x = torch.randn(2, 3, 64, 64)
    print(model(x))
