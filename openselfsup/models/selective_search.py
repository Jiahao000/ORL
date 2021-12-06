import torch.nn as nn

from .registry import MODELS


@MODELS.register_module
class SelectiveSearch(nn.Module):
    """Selective-search proposal generation.
    """

    def __init__(self, **kwargs):
        super(SelectiveSearch, self).__init__()

    def forward_test(self, bbox, **kwargs):
        assert bbox.dim() == 3, \
            "Input bbox must have 3 dims, got: {}".format(bbox.dim())
        # bbox: 1xBx4
        return dict(bbox=bbox.cpu())

    def forward(self, mode='test', **kwargs):
        assert mode == 'test', \
            "Support test inference mode only, got: {}".format(mode)
        return self.forward_test(**kwargs)
