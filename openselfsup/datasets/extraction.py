import torch
from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class ExtractDataset(BaseDataset):
    """Dataset for feature extraction.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(ExtractDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
