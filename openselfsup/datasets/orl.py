import json
import math
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose

from openselfsup.utils import build_from_cfg

from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box (x, y, w, h)
    gt_box :   the coordinate for ground truth bounding box (x, y, w, h)
    return :   the iou score
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])

    iw = max(ixmax - ixmin, 0.)
    ih = max(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (pred_box[2] * pred_box[3] + gt_box[2] * gt_box[3] - inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / float(uni)

    return iou

def aug_bbox(img, box, shift, scale, ratio, iou_thr, attempt_num=200):
    img_w, img_h = img.size
    x, y, w, h = box[0], box[1], box[2], box[3]
    cx, cy  = (x + 0.5 * w), (y + 0.5 * h)
    area = w * h
    for attempt in range(attempt_num):
        aug_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aug_ratio = math.exp(random.uniform(*log_ratio))
        aug_w = int(round(math.sqrt(aug_area * aug_ratio)))
        aug_h = int(round(math.sqrt(aug_area / aug_ratio)))
        aug_cx = cx + random.uniform(*shift) * w
        aug_cy = cy + random.uniform(*shift) * h
        aug_x, aug_y = int(round(aug_cx - 0.5 * aug_w)), int(round(aug_cy - 0.5 * aug_h))
        if aug_x >= 0 and aug_y >= 0 and (aug_x + aug_w) <= img_w and (aug_y + aug_h) <= img_h:
            aug_box = [aug_x, aug_y, aug_w, aug_h]
            if iou_thr is not None:
                iou = get_iou(aug_box, box)
                if iou > iou_thr:
                    return aug_box
            else:
                return aug_box
    return box


@DATASETS.register_module
class ORLDataset(Dataset):
    """Dataset for ORL.
    """

    def __init__(self,
                 data_source,
                 image_pipeline1,
                 image_pipeline2,
                 patch_pipeline1,
                 patch_pipeline2,
                 patch_size=224,
                 interpolation=2,
                 shift=(-0.5, 0.5),
                 scale=(0.5, 2.),
                 ratio=(0.5, 2.),
                 iou_thr=0.5,
                 attempt_num=200,
                 prefetch=False):
        self.data_source = build_datasource(data_source)
        image_pipeline1 = [build_from_cfg(p, PIPELINES) for p in image_pipeline1]
        self.image_pipeline1 = Compose(image_pipeline1)
        image_pipeline2 = [build_from_cfg(p, PIPELINES) for p in image_pipeline2]
        self.image_pipeline2 = Compose(image_pipeline2)
        patch_pipeline1 = [build_from_cfg(p, PIPELINES) for p in patch_pipeline1]
        self.patch_pipeline1 = Compose(patch_pipeline1)
        patch_pipeline2 = [build_from_cfg(p, PIPELINES) for p in patch_pipeline2]
        self.patch_pipeline2 = Compose(patch_pipeline2)
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.shift = shift
        self.scale = scale
        self.ratio = ratio
        self.iou_thr = iou_thr
        self.attempt_num = attempt_num
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, knn_img, intra_box, knn_box = self.data_source.get_sample(idx)
        ibox1 = random.choice(intra_box)
        ibox2 = aug_bbox(
            img, ibox1, self.shift, self.scale, self.ratio, self.iou_thr, self.attempt_num)
        kbox_pair = random.choice(knn_box)
        kbox1, kbox2 = kbox_pair[:4], kbox_pair[4:]
        ipatch1 = TF.resized_crop(img, ibox1[1], ibox1[0], ibox1[3], ibox1[2],
            (self.patch_size, self.patch_size), interpolation=self.interpolation)
        ipatch2 = TF.resized_crop(img, ibox2[1], ibox2[0], ibox2[3], ibox2[2],
            (self.patch_size, self.patch_size), interpolation=self.interpolation)
        kpatch1 = TF.resized_crop(img, kbox1[1], kbox1[0], kbox1[3], kbox1[2],
            (self.patch_size, self.patch_size), interpolation=self.interpolation)
        kpatch2 = TF.resized_crop(knn_img, kbox2[1], kbox2[0], kbox2[3], kbox2[2],
            (self.patch_size, self.patch_size), interpolation=self.interpolation)
        img1 = self.image_pipeline1(img)
        img2 = self.image_pipeline2(img)
        ipatch1 = self.patch_pipeline1(ipatch1)
        ipatch2 = self.patch_pipeline2(ipatch2)
        kpatch1 = self.patch_pipeline1(kpatch1)
        kpatch2 = self.patch_pipeline2(kpatch2)
        if self.prefetch:
            img1 = torch.from_numpy(to_numpy(img1))
            img2 = torch.from_numpy(to_numpy(img2))
            ipatch1 = torch.from_numpy(to_numpy(ipatch1))
            ipatch2 = torch.from_numpy(to_numpy(ipatch2))
            kpatch1 = torch.from_numpy(to_numpy(kpatch1))
            kpatch2 = torch.from_numpy(to_numpy(kpatch2))
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        ipatch_cat = torch.cat((ipatch1.unsqueeze(0), ipatch2.unsqueeze(0)), dim=0)
        kpatch_cat = torch.cat((kpatch1.unsqueeze(0), kpatch2.unsqueeze(0)), dim=0)
        return dict(img=img_cat, ipatch=ipatch_cat, kpatch=kpatch_cat)

    def evaluate(self, scores, keyword, logger=None, **kwargs):
        raise NotImplemented
