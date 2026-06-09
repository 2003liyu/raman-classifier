import thop

import torch.nn as nn
import torch


class BaseNet(nn.Module):

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self.base_name:str = kwargs["base_name"]
        self.__n_parameters:int = 0
        self.__flops:int = 0
    
    @property
    def title(self)->str:
        pass

    @property
    def n_parameters(self)->int:
        if self.__n_parameters == 0:
            self.__n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return self.__n_parameters
    
    @property
    def device(self)->torch.device:
        return next(self.parameters()).device

    def flops(self, x:torch.Tensor)->int:
        if self.__flops != 0:
            return self.__flops
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)
        with torch.no_grad():
            self.eval()
            flops, params = thop.profile(self, inputs=(x,))

        self.__flops = flops
        
        return flops

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if m.weight is not None and 0 not in m.weight.shape:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None and 0 not in m.weight.shape:
                    nn.init.constant_(m.weight, 1)

                if m.bias is not None and 0 not in m.bias.shape:
                    nn.init.constant_(m.bias, 0)
