# resgcn-n39-r8

import torch
from torch import nn
import torch.nn.functional as F

from augmentation import *
from graph import A, connect_joint

multi_input = MultiInput(connect_joint=connect_joint, enabled=True)

# Lấy các hàm từ file blocks.py
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class ResGCN_Module(nn.Module):
    def __init__(self, in_channels, out_channels, block, A, initial=False, stride=1, kernel_size=[9,2], **kwargs):
        super(ResGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            raise ValueError()
        
        temporal_window_size, max_graph_distance = kernel_size

        if initial:
            module_res, block_res = False, False
        elif block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )

        spatial_block = import_class('blocks.Spatial_{}_Block'.format(block))
        temporal_block = import_class('blocks.Temporal_{}_Block'.format(block))
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res, **kwargs)
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res, **kwargs)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, A):
        return self.tcn(self.scn(x, A*self.edge), self.residual(x))


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, block, num_channel, A, **kwargs):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)

        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True, **kwargs)]
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 64, block, A, **kwargs) for _ in range(structure[1] - 1)]
        module_list += [ResGCN_Module(64, 32, block, A, **kwargs)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        x = self.bn(x)
        for layer in self.layers:
            x = layer(x, self.A)

        return x


class ResGCN(nn.Module):
    def __init__(self, module = ResGCN_Module, structure = [1, 2, 2, 2], block = 'Bottleneck', num_input=1, num_channel=3, num_class=128, A = A, **kwargs):
        super(ResGCN, self).__init__()

        self.register_buffer('A', A)

        # input branches
        self.input_branches = nn.ModuleList([
            ResGCN_Input_Branch(structure, block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [module(32*num_input, 128, block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        # output
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):

        # N, I, C, T, V = x.size()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # input branches
        if len(self.input_branches) == 1:
            if x.dim() == 3:
              # Single sample: [60, 17, 3] → [1, 3, 60, 17]
              x = x.permute(2, 0, 1).unsqueeze(0)
            elif x.dim() == 4:
              # Batch input: [B, 60, 17, 3] → [B, 3, 60, 17]
              x = x.permute(0, 3, 1, 2)
            else:
              raise ValueError(f"Num input 1 expected 3D/4D tensor, got {x.dim()}D")
            
            x = self.input_branches[0](x)
            
        elif len(self.input_branches) == 3:
            x = multi_input(x)  
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
            x_cat = []
            for i, branch in enumerate(self.input_branches):
                x_cat.append(branch(x[:,i,:,:,:]))
            x = torch.cat(x_cat, dim=1)

        else:
            raise ValueError(f"Num input 1/3")

        # main stream
        for layer in self.main_stream:
            x = layer(x, self.A)

        # output
        x = self.global_pooling(x)
        # x = self.fcn(x.squeeze())
        x = self.fcn(x.squeeze(-1).squeeze(-1))

        # L2 normalization
        x = F.normalize(x , dim=1, p=2)

        return x


def init_param(modules):
    torch.manual_seed(1)
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    torch.manual_seed(1)
    for m in modules:
        if isinstance(m, ResGCN_Module):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)