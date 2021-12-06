import os
import random

from PIL import Image
import mmcv

from ..registry import DATASOURCES
from .utils import McLoader


@DATASOURCES.register_module
class COCOORLJson(object):

    def __init__(self, root, json_file, topk_knn_image, memcached=False, mclient_path=None):
        data = mmcv.load(json_file)
        self.fns = data['images']['file_name']
        self.intra_bboxes = data['pseudo_annotations']['bbox']
        self.total_knn_image_num = data['info']['knn_image_num']
        self.knn_image_ids = data['pseudo_annotations']['knn_image_id']
        self.knn_bbox_pairs = data['pseudo_annotations']['knn_bbox_pair']  # NxKx(topk_bbox_num)x8
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.topk_knn_image = topk_knn_image
        assert self.topk_knn_image <= self.total_knn_image_num, \
            "Top-k knn image number exceeds total number of knn images available. Abort!"
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            assert self.mclient_path is not None
            self.mc_loader = McLoader(self.mclient_path)
            self.initialized = True

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        # randomly select one knn image
        rnd = random.randint(0, self.topk_knn_image - 1)
        target_id = self.knn_image_ids[idx][rnd]
        if self.memcached:
            self._init_memcached()
        if self.memcached:
            img = self.mc_loader(self.fns[idx])
            knn_img = self.mc_loader(self.fns[target_id])
        else:
            img = Image.open(self.fns[idx])
            knn_img = Image.open(self.fns[target_id])
        img = img.convert('RGB')
        knn_img = knn_img.convert('RGB')
        # load proposals
        intra_bbox = self.intra_bboxes[idx]
        knn_bbox = self.knn_bbox_pairs[idx][rnd]  # (topk_bbox_num)x8
        return img, knn_img, intra_bbox, knn_bbox
