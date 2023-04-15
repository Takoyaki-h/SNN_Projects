import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron



class soft_unit(nn.Module):
    def __init__(self):
        super(soft_unit, self).__init__()
        self.globalAvg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64, 4, bias=False),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4, 64, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        globalAvg = self.globalAvg(x) # [bs,64,1,1]
        x2 = self.fc1(globalAvg.squeeze(-1).squeeze(-1)) # [bs,64]
        x2 = self.fc2(x2)
        # x=torch.sigmoid(x)
        x2 = x2.unsqueeze(-1).unsqueeze(-1)# [bs,64,1,1]
        x = torch.multiply(x2, x)
        return x


class SRBU(nn.Module):
    def __init__(self):
        super(SRBU, self).__init__()
        self.bn1 = nn.Sequential(
            nn.BatchNorm2d(64),
            neuron.IFNode()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        )
        self.bn2 = nn.Sequential(
            nn.BatchNorm2d(128),
            neuron.IFNode()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        )
        self.soft_fun = soft_unit()

    def forward(self, x):
        identity = x
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)

        vth = self.soft_fun(x)
        #         soft thresholds
        sub = torch.sub(torch.abs(x), vth)
        zeros = torch.sub(sub, sub)
        n_sub = torch.max(sub, zeros)
        x = torch.multiply(torch.sign(x), n_sub)

        x += identity
        return x


class DSRSN(nn.Module):
    def __init__(self):
        super(DSRSN, self).__init__()
        # self.T = T
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            neuron.IFNode()
        )
        self.srbu1 = SRBU()
        self.srbu2 = SRBU()

        self.srbu3 = SRBU()
        self.srbu4 = SRBU()

        self.bn = nn.Sequential(
            nn.BatchNorm2d(64),
            # neuron.IFNode()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 10, bias=False),
            neuron.IFNode()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.srbu1(x)
        x = self.srbu2(x)
        x = self.srbu3(x)
        x = self.srbu4(x)

        x = self.bn(x)

        # x=self.fc(x)
        x = self.fc(x)
        # for t in range(1, self.T):
        #     out_spikes_counter += self.fc(x)
        return x

