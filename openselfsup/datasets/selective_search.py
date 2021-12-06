import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from openselfsup.utils import build_from_cfg

from torchvision.transforms import Compose
import torchvision.transforms.functional as TF

from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy


def selective_search(image, method="fast"):
    # initialize OpenCV's selective search implementation
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set the input image
    ss.setBaseImage(image)
    # check to see if we are using the *fast* but *less accurate* version
    # of selective search
    if method == "fast":
        # print("[INFO] using *fast* selective search")
        ss.switchToSelectiveSearchFast()
    # otherwise we are using the *slower* but *more accurate* version
    else:
        # print("[INFO] using *quality* selective search")
        ss.switchToSelectiveSearchQuality()
    # run selective search on the input image
    boxes = ss.process()
    return boxes

def box_filter(boxes, min_size=None, max_ratio=None, topN=None):
    proposal = []

    for box in boxes:
        # Calculate width and height of the box
        w, h = box[2], box[3]

        # Filter for size
        if min_size:
            if w < min_size or h < min_size:
                continue

        # Filter for box ratio
        if max_ratio:
            if w / h > max_ratio or h / w > max_ratio:
                continue

        proposal.append(box)

    if topN:
        if topN <= len(proposal):
            return proposal[:topN]
        else:
            return proposal
    else:
        return proposal


@DATASETS.register_module
class SelectiveSearchDataset(Dataset):
    """Dataset for generating selective search proposals.
    """

    def __init__(self,
                 data_source,
                 method='fast',
                 min_size=None,
                 max_ratio=None,
                 topN=None):
        self.data_source = build_datasource(data_source)
        self.method = method
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.topN = topN

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img = self.data_source.get_sample(idx)
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = selective_search(img_cv2, self.method)
        if self.topN is not None:
            boxes = box_filter(boxes, self.min_size, self.max_ratio, self.topN)
        boxes = torch.from_numpy(np.array(boxes))
        # bbox: Bx4
        # B is the total number of original/topN selective search bboxes
        return dict(bbox=boxes)

    def evaluate(self, bbox, **kwargs):
        if not isinstance(bbox, list):
            bbox = bbox.tolist()
        # dict
        data_ss = {}
        data_ss['bbox'] = bbox
        return data_ss
