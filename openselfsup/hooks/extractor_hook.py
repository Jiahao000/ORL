import numpy as np

from mmcv.runner import Hook

import torch
import torch.nn as nn

from openselfsup.utils import print_log
from .registry import HOOKS
from .extractor import Extractor


@HOOKS.register_module
class ExtractorHook(Hook):
    """Feature extractor hook"""

    def __init__(self,
                 extractor,
                 normalize=True,
                 dist_mode=True):
        assert dist_mode, "non-dist mode is not implemented"
        self.extractor = Extractor(dist_mode=dist_mode, **extractor)
        self.normalize = normalize
        self.dist_mode = dist_mode

    def before_run(self, runner):
        self._extract_func(runner)

    def _extract_func(self, runner):
        # step 1: get features
        runner.model.eval()
        features = self.extractor(runner)
        if self.normalize:
            features = nn.functional.normalize(torch.from_numpy(features), dim=1)
        # step 2: save features
        if not self.dist_mode or (self.dist_mode and runner.rank == 0):
            np.save(
                "{}/feature_epoch_{}.npy".format(runner.work_dir,
                                                 runner.epoch),
                features.numpy())
            print_log(
                "Feature extraction done!!! total features: {}\tfeature dimension: {}".format(
                    features.size(0), features.size(1)),
                logger='root')
            