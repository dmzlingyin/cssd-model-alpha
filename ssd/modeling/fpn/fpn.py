import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        # latlayer
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer6 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # smoothlayer
        self.smoothlayer1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.smoothlayer2 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1)
        self.smoothlayer3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.smoothlayer4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smoothlayer5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smoothlayer6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # normlayer
        self.norm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
        added = F.interpolate(x, size=(H, W), mode='trilinear', align_corners=True) + y
        return self.relu(added)

    def forward(self, features):
        c1 = self.latlayer1(features[0])
        c1 = self.norm(c1)
        c2 = self.latlayer2(features[1])
        c2 = self.norm(c2)
        c3 = self.latlayer3(features[2])
        c3 = self.norm(c3)
        c4 = self.latlayer4(features[3])
        c4 = self.norm(c4)
        c5 = self.latlayer5(features[4])
        c5 = self.norm(c5)
        c6 = self.latlayer6(features[5])
        c6 = self.norm(c6)
        c7 = features[6]
        # c7 = self.norm(c7)

        p6 = self.upsample_add(c7, c6)
        p5 = self.upsample_add(p6, c5)
        p4 = self.upsample_add(p5, c4)
        p3 = self.upsample_add(p4, c3)
        p2 = self.upsample_add(p3, c2)
        p1 = self.upsample_add(p2, c1)

        p1 = self.smoothlayer1(p1)
        p2 = self.smoothlayer2(p2)
        p3 = self.smoothlayer3(p3)
        p4 = self.smoothlayer4(p4)
        p5 = self.smoothlayer5(p5)
        p6 = self.smoothlayer6(p6)

        return [p1, p2, p3, p4, p5, p6, c7]
