import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class PixelNorm(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class _PixelInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_PixelInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Pixel norm
        if self.affine:
            pn_w = self.weight * self.gate
        else:
            pn_w = self.gate
        out_pn = input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)
        out_pn.mul_(pn_w[None, :, None, None])

        # Instance norm
        b, c  = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])
        
        return out_pn + out_in
    

# class PixelInstanceNorm1d(_PixelInstanceNorm):
#     def _check_input_dim(self, input):
#         if input.dim() != 2 and input.dim() != 3:
#             raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class PixelInstanceNorm2d(_PixelInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


# class PixelInstanceNorm3d(_PixelInstanceNorm):
#     def _check_input_dim(self, input):
#         if input.dim() != 5:
#             raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))