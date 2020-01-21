import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class MaskedConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', mask: torch.FloatTensor = None):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                           dilation, groups, bias, padding_mode)
        if mask is None:
            self._mask = torch.ones_like(self.weight, dtype=self.weight.dtype)
        else:
            self._mask = mask
            assert self._mask.size() == self.weight.size()

    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight * self._mask)

    def prune_by_threshold(self, thr):
        self._mask *= (torch.abs(self.weight) >= thr).float()

    def prune_by_rank(self, rank):
        weights_val = self.weight[self._mask == 1]
        sorted_abs_weights = torch.sort(torch.abs(weights_val))[0]
        thr = sorted_abs_weights[rank]
        self.prune_by_threshold(thr)

    def prune_by_pct(self, pct):
        prune_idx = int(self.n_weights * pct)
        self.prune_by_rank(prune_idx)

    def random_prune_by_pct(self, pct):
        prune_idx = int(self.n_weights * pct)
        rand = torch.rand_like(self._mask, device=self._mask.device)
        rand_val = rand[self._mask == 1]
        sorted_abs_rand = torch.sort(rand_val)[0]
        thr = sorted_abs_rand[prune_idx]
        self._mask *= (rand >= thr).float()

    # def reinitialize(self):
    #     self._weights = Parameter(self._initial_weights)
    #     self._init_biases()

    def to_device(self, device: torch.device):
        # self._initial_weights = self._initial_weights.to(device)
        self._mask = self._mask.to(device)

    def to_conv2d(self):
        conv2d = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                                 self.dilation, self.groups, self.bias is not None, self.padding_mode)
        conv2d.load_state_dict(self.state_dict())
        with torch.no_grad():
            conv2d.weight.mul_(self._mask)
        return conv2d

    @property
    def mask(self):
        return self._mask

    @property
    def n_weights(self):
        return torch.nonzero(self._mask).size(0)
