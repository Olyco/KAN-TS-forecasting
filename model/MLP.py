import torch.nn as nn
import torch.nn.functional as F


class MLP_(nn.Module):
    def __init__(self, width, activation=nn.SiLU(), output_activation=None):
        
        super(MLP_, self).__init__()
        
        linears = []
        self.width = width
        self.depth = depth = len(width) - 1
        for i in range(depth):
            linears.append(nn.Linear(width[i], width[i+1]))
        self.linears = nn.ModuleList(linears)

        self.act_fun = activation
        
    def forward(self, x):
        # print(x.shape)
        reshape = len(x.shape) == 3
        if reshape:
          B, L, N = x.shape
          x = x.transpose(1, 2)
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1]) # x = .reshape(B * N, L)
        # print(x.shape)
        
        for i in range(self.depth):            
            x = self.linears[i](x)
            if i < self.depth - 1:
                x = self.act_fun(x)

        # print(x.shape)
        if reshape:
          x = x.reshape(B, N, -1).permute(0, 2, 1) 
        else:
          x = x.reshape(*original_shape[:-1], x.shape[-1])
        # print(x.shape)
            
        return x