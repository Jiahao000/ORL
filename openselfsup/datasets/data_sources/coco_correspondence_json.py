import os

import mmcv
import numpy as np
from PIL import Image

from ..registry import DATASOURCES
from .utils import McLoader


@DATASOURCES.register_module
class COCOCorrespondenceJson(object):

    def __init__(self,
                 root,
                 knn_json_file,
                 ss_json_file,
                 knn_image_num,
                 part,
                 num_parts=10,
                 data_len=118287,
                 memcached=False,
                 mclient_path=None):
        assert part in np.arange(num_parts).tolist(), \
            "part order must be within [0, num_parts)"
        print('loading knn json file...')
        data = mmcv.load(knn_json_file)
        print('loaded knn json file!')
        print('loading selective search json file, this may take several minutes...')
        if isinstance(ss_json_file, list):
            data_ss_list = [mmcv.load(ss) for ss in ss_json_file]
            self.bboxes = []
            for ss in data_ss_list:
                self.bboxes += ss['bbox']
        else:
            data_ss = mmcv.load(ss_json_file)
            self.bboxes = data_ss['bbox']
        print('loaded selective search json file!')
        # divide the whole dataset into several parts to enable parallel roi pair retrieval.
        # each part should be run on single gpu and all parts can be run on multiple gpus in parallel.
        part_len = int(data_len / num_parts)
        print('processing part {}...'.format(part))
        if part == num_parts - 1:  # last part
            self.fns = data['images']['file_name']
            self.fns = [os.path.join(root, fn) for fn in self.fns]
            self.part_fns = self.fns[part * part_len:]
            self.part_labels = data['pseudo_annotations']['knn_image_id'][part * part_len:]
            self.part_bboxes = self.bboxes[part * part_len:]
        else:
            self.fns = data['images']['file_name']
            self.fns = [os.path.join(root, fn) for fn in self.fns]
            self.part_fns = self.fns[part * part_len:(part + 1) * part_len]
            self.part_labels = data['pseudo_annotations']['knn_image_id'][part * part_len:(part + 1) * part_len]
            self.part_bboxes = self.bboxes[part * part_len:(part + 1) * part_len]
        self.knn_image_num = knn_image_num
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.part_fns)

    def get_sample(self, idx):
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.part_fns[idx])
        else:
            img = Image.open(self.part_fns[idx])
        img = img.convert('RGB')
        # load knn images
        target = self.part_labels[idx][:self.knn_image_num]
        if self.memcached:
            knn_imgs = [self.mc_loader(self.fns[i]) for i in target]
        else:
            knn_imgs = [Image.open(self.fns[i]) for i in target]
        knn_imgs = [knn_img.convert('RGB') for knn_img in knn_imgs]
        # load selective search proposals
        bbox = self.part_bboxes[idx]
        knn_bboxes = [self.bboxes[i] for i in target]
        return img, knn_imgs, bbox, knn_bboxes
