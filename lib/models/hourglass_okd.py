'''
Hourglass network inserted in the pre-activated Resnet 
Use lr=0.01 for current version
(c) YANG, Wei 
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

BN_MOMENTUM = 0.1


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # inplanes -> 2*planes
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        # 1x1 conv with stride = 1, inplanes -> planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 3x3 conv with stride = ?, planes -> planes
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 1x1 conv with stride = 1, planes -> 2*planes
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for _ in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for _ in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, block, cfg, num_kernel=3):
        super(HourglassNet, self).__init__()
        # Parameters: num_feats=256, num_stacks=8, num_blocks=1, num_classes=16
        extra = cfg.MODEL.EXTRA
        num_feats = 256
        num_stacks = extra.NUM_STACKS
        num_blocks = 1
        num_classes = 16

        self.num_branches = 3

        # Parameters: self.inplanes=265/4=64
        self.inplanes = int(num_feats / 4)
        # Parameters: self.num_feats=256/2=128
        self.num_feats = int(num_feats / 2)
        # Parameters: self.num_stacks=8
        self.num_stacks = num_stacks
        # TODO: can be repalced by 3 3x3 convs
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        # Parameters: planes=64, blocks=1, stride=1, output channels = 128
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # Build hourglass modules
        # ch = self.num_feats * block.expansion
        # hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        # for i in range(num_stacks):
        #     # Parameters: num_blocks=4, self.num_feats=128
        #     hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
        #     res.append(self._make_residual(block, self.num_feats, num_blocks))
        #     fc.append(self._make_fc(ch, ch))
        #     score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
        #     if i < num_stacks-1:
        #         fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
        #         score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))

        # self.hg = nn.ModuleList(hg)
        # self.res = nn.ModuleList(res)
        # self.fc = nn.ModuleList(fc)
        # self.score = nn.ModuleList(score)
        # self.fc_ = nn.ModuleList(fc_)
        # self.score_ = nn.ModuleList(score_)

        channel = 256
        self.convs = nn.ModuleList([])
        for i in range(num_kernel):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3 + i * 2, stride=1, padding=1 + i),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=False)
            ))

        mid_channel = 64
        self.fc = nn.Linear(channel, mid_channel)
        self.fcs = nn.ModuleList([])
        for i in range(num_kernel):
            self.fcs.append(
                nn.Linear(mid_channel, num_classes)
            )
        self.softmax = nn.Softmax(dim=1)

        ch = self.num_feats * block.expansion
        for branch_idx in range(self.num_branches):
            for stack_idx in range(self.num_stacks):
                # setattr(self, 'hg_b_' + str(branch_idx) + '_s_' + str(stack_idx),
                #         Hourglass(block, num_blocks, self.num_feats, 4))
                # setattr(self, 'res_b_' + str(branch_idx) + '_s_' + str(stack_idx),
                #         self._make_residual(block, self.num_feats, num_blocks))
                # setattr(self, 'fc_b_' + str(branch_idx) + '_s_' + str(stack_idx),
                #         self._make_fc(ch, ch))

                # e.g b=3, stacks=2
                # ind_b_0_s_0, ind_b_1_s_0, ind_b_2_s_0,
                # ind_b_0_s_1, ind_b_1_s_1, ind_b_2_s_1
                setattr(self, 'ind_b_' + str(branch_idx) + '_s_' + str(stack_idx),
                        self._make_individual(block, num_blocks, self.num_feats, ch))
                setattr(self, 'score_b_' + str(branch_idx) + '_s_' + str(stack_idx),
                        nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))

                if stack_idx < num_stacks - 1:
                    setattr(self, '_fc_b_' + str(branch_idx) + '_s_' + str(stack_idx),
                            nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                    setattr(self, '_score_b_' + str(branch_idx) + '_s_' + str(stack_idx),
                            nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))

    # make individual learner
    def _make_individual(self, block, num_blocks, num_feats, ch):
        layers = []
        layers.append(Hourglass(block, num_blocks, num_feats, 4))
        layers.append(self._make_residual(block, num_feats, num_blocks))
        layers.append(self._make_fc(ch, ch))
        # layers.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
        return nn.Sequential(*layers)

    def _make_residual(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
            conv,
            bn,
            self.relu,
        )

    def forward(self, x):
        out = []
        x = self.conv1(x)  # [B,64,128,128]
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # [B,128,128,128]
        x = self.maxpool(x)  # [B,128,64,64]
        x = self.layer2(x)  # [B,256,64,64]
        x = self.layer3(x)  # [B,256,64,64] (256/4=64)

        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)  # [B,1,256,64,64]
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)  # [B,3,256,64,64]

        fea_U = torch.sum(feas, dim=1)  # [B,256,64,64]
        fea_s = fea_U.mean(-1).mean(-1)  # [B,256]
        fea_z = self.fc(fea_s)  # [B, 64]

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)  # [B,1,16]
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)  # [B,3,16]

        attention_vectors = self.softmax(attention_vectors)  # [B,3,16]
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)  # [B,3,16,1,1]

        for branch_idx in range(self.num_branches):
            vars()['x' + str(branch_idx)] = x  # x0, x1, x2, ...

            for stack_idx in range(self.num_stacks):
                y = getattr(self, 'ind_b_' + str(branch_idx) + '_s_' + str(stack_idx))(vars()['x' + str(branch_idx)])
                score = getattr(self, 'score_b_' + str(branch_idx) + '_s_' + str(stack_idx))(y)
                # i+=1
                out.append(score)
                # score branch_1, 2, 3,...
                if stack_idx < self.num_stacks - 1:
                    fc_ = getattr(self, '_fc_b_' + str(branch_idx) + '_s_' + str(stack_idx))(y)
                    score_ = getattr(self, '_score_b_' + str(branch_idx) + '_s_' + str(stack_idx))(score)
                    vars()['x' + str(branch_idx)] = vars()['x' + str(branch_idx)] + fc_ + score_
            #         I need to calulate one 3x17 matrix

            # original: [B, 16, 64, 64]
            if branch_idx == 0:
                heatmap_concat = out[self.num_stacks - 1].unsqueeze(1) # [B,1,16,64,64]
            else:
                heatmap_concat = torch.cat(
                    [heatmap_concat, out[self.num_stacks - 1 + branch_idx * (self.num_branches-1)].unsqueeze(1)],
                    dim=1)
                # heatnmap_concat: [B,3,16,64,64]

        fea_v = (heatmap_concat * attention_vectors).sum(dim=1)

        # print(f'i is {i}')

        return out, fea_v


def get_pose_net(cfg, is_train, **kwargs):
    model = HourglassNet(Bottleneck, cfg, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    input = torch.rand([8, 3, 256, 256])
    model = HourglassNet(Bottleneck, num_branches=3, num_kernel=3)
    output = model(input)
    print('test')
