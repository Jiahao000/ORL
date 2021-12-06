import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from openselfsup.utils import build_from_cfg

from torchvision.transforms import Compose
import torchvision.transforms.functional as TF

from .registry import DATASETS, PIPELINES
from .builder import build_datasource
from .utils import to_numpy


def get_max_iou(pred_boxes, gt_box):
    """
    pred_boxes : multiple coordinate for predict bounding boxes (x, y, w, h)
    gt_box :   the coordinate for ground truth bounding box (x, y, w, h)
    return :   the max iou score about pred_boxes and gt_box
    """
    # 1.get the coordinate of inters
    ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
    ixmax = np.minimum(pred_boxes[:, 0] + pred_boxes[:, 2], gt_box[0] + gt_box[2])
    iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
    iymax = np.minimum(pred_boxes[:, 1] + pred_boxes[:, 3], gt_box[1] + gt_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (pred_boxes[:, 2] * pred_boxes[:, 3] + gt_box[2] * gt_box[3] - inters)

    # 4. calculate the overlaps and find the max overlap between pred_boxes and gt_box
    iou = inters / uni
    iou_max = np.max(iou)

    return iou_max


def box_filter(boxes, min_size=20, max_ratio=None, topN=None, max_iou_thr=None):
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

        # Filter for overlap
        if max_iou_thr:
            if len(proposal):
                iou_max = get_max_iou(np.array(proposal), np.array(box))
                if iou_max > max_iou_thr:
                    continue

        proposal.append(box)

    if not len(proposal):  # ensure at least one box for each image
        proposal.append(boxes[0])

    if topN:
        if topN <= len(proposal):
            return proposal[:topN]
        else:
            return proposal
    else:
        return proposal


@DATASETS.register_module
class CorrespondenceDataset(Dataset):
    """Dataset for generating corresponding intra- and inter-RoIs.
    """

    def __init__(self,
                 data_source,
                 format_pipeline,
                 patch_size=224,
                 min_size=96,
                 max_ratio=3,
                 topN=100,
                 max_iou_thr=0.5,
                 knn_image_num=10,
                 topk_bbox_ratio=0.1,
                 prefetch=False):
        self.data_source = build_datasource(data_source)
        format_pipeline = [build_from_cfg(p, PIPELINES) for p in format_pipeline]
        self.format_pipeline = Compose(format_pipeline)
        self.patch_size = patch_size
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.topN = topN
        self.max_iou_thr = max_iou_thr
        self.knn_image_num = knn_image_num
        self.topk_bbox_ratio = topk_bbox_ratio
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, knn_imgs, box, knn_boxes = self.data_source.get_sample(idx)
        filtered_box = box_filter(box, self.min_size, self.max_ratio, self.topN, self.max_iou_thr)
        filtered_knn_boxes = [
            box_filter(knn_box, self.min_size, self.max_ratio, self.topN, self.max_iou_thr)
            for knn_box in knn_boxes
        ]
        patch_list = []
        for x, y, w, h in filtered_box:
            patch = TF.resized_crop(img, y, x, h, w, (self.patch_size, self.patch_size))
            if self.prefetch:
                patch = torch.from_numpy(to_numpy(patch))
            else:
                patch = self.format_pipeline(patch)
            patch_list.append(patch)
        knn_patch_lists = []
        for k in range(len(knn_imgs)):
            knn_patch_list = []
            for x, y, w, h in filtered_knn_boxes[k]:
                patch = TF.resized_crop(knn_imgs[k], y, x, h, w, (self.patch_size, self.patch_size))
                if self.prefetch:
                    patch = torch.from_numpy(to_numpy(patch))
                else:
                    patch = self.format_pipeline(patch)
                knn_patch_list.append(patch)
            knn_patch_lists.append(torch.stack(knn_patch_list))
        filtered_box = torch.from_numpy(np.array(filtered_box))
        filtered_knn_boxes = [torch.from_numpy(np.array(knn_box)) for knn_box in filtered_knn_boxes]
        knn_img_keys = ['{}nn_img'.format(k) for k in range(len(knn_imgs))]
        knn_bbox_keys = ['{}nn_bbox'.format(k) for k in range(len(knn_imgs))]
        # img: BCHW, knn_img: K BCHW, bbox: Bx4, knn_bbox= K Bx4
        # K is the number of knn images, B is the number of filtered bboxes
        dict1 = dict(img=torch.stack(patch_list))
        dict2 = dict(bbox=filtered_box)
        dict3 = dict(zip(knn_img_keys, knn_patch_lists))
        dict4 = dict(zip(knn_bbox_keys, filtered_knn_boxes))
        return {**dict1, **dict2, **dict3, **dict4}

    def evaluate(self, json_file, intra_bbox, inter_bbox, **kwargs):
        assert (len(intra_bbox) == len(inter_bbox)), \
            "Mismatch the number of images in part training set, got: intra: {} inter: {}".format(
                len(intra_bbox), len(inter_bbox))
        data = mmcv.load(json_file)
        # dict
        data_new = {}
        # sub-dict
        info = {}
        image_info = {}
        pseudo_anno = {}
        info['bbox_min_size'] = self.min_size
        info['bbox_max_aspect_ratio'] = self.max_ratio
        info['bbox_max_iou'] = self.max_iou_thr
        info['intra_bbox_num'] = self.topN
        info['knn_image_num'] = self.knn_image_num
        info['knn_bbox_pair_ratio'] = self.topk_bbox_ratio
        image_info['file_name'] = data['images']['file_name']
        image_info['id'] = data['images']['id']
        pseudo_anno['image_id'] = data['pseudo_annotations']['image_id']
        pseudo_anno['bbox'] = intra_bbox
        pseudo_anno['knn_image_id'] = data['pseudo_annotations']['knn_image_id']
        pseudo_anno['knn_bbox_pair'] = inter_bbox
        data_new['info'] = info
        data_new['images'] = image_info
        data_new['pseudo_annotations'] = pseudo_anno
        return data_new
