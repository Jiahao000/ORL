import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class Correspondence(nn.Module):
    """Correspondence discovery in Stage 2 of ORL.

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
        base_momentum (float): The base momentum coefficient for the target network.
            Default: 0.99.
        knn_image_num (int): The number of KNN images. Default: 10.
        topk_bbox_ratio (float): The ratio of retrieved top-ranked RoI pairs. Default: 0.1.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.99,
                 knn_image_num=10,
                 topk_bbox_ratio=0.1,
                 **kwargs):
        super(Correspondence, self).__init__()
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        self.neck = self.online_net[1]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

        self.knn_image_num = knn_image_num
        self.topk_bbox_ratio = topk_bbox_ratio

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained) # backbone
        self.online_net[1].init_weights(init_linear='kaiming') # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights(init_linear='kaiming')

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0].clone().detach()

        loss = self.head(proj_online_v1, proj_target_v2)['loss'] + \
               self.head(proj_online_v2, proj_target_v1)['loss']
        return dict(loss=loss)

    def forward_test(self, img, bbox, **kwargs):
        knn_imgs = [kwargs.get('{}nn_img'.format(k)) for k in range(self.knn_image_num)]
        knn_bboxes = [kwargs.get('{}nn_bbox'.format(k)) for k in range(self.knn_image_num)]
        assert img.size(0) == 1, \
            "Input batch size must be 1, got: {}".format(img.size(0))
        assert img.dim() == 5, \
            "Input img must have 5 dims, got: {}".format(img.dim())
        assert bbox.dim() == 3, \
            "Input bbox must have 3 dims, got: {}".format(bbox.dim())
        assert knn_imgs[0].dim() == 5, \
            "Input knn_img must have 5 dims, got: {}".format(knn_imgs[0].dim())
        assert knn_bboxes[0].dim() == 3, \
            "Input knn_bbox must have 3 dims, got: {}".format(knn_bboxes[0].dim())
        img = img.view(
            img.size(0) * img.size(1), img.size(2), img.size(3),
            img.size(4))  # (1B)xCxHxW
        knn_imgs = [knn_img.view(
            knn_img.size(0) * knn_img.size(1), knn_img.size(2),
            knn_img.size(3), knn_img.size(4)) for knn_img in knn_imgs]  # K (1B)xCxHxW
        with torch.no_grad():
            feat = self.backbone(img)[0].clone().detach()
            knn_feats = [self.backbone(knn_img)[0].clone().detach() for knn_img in knn_imgs]
            feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            knn_feats = [nn.functional.adaptive_avg_pool2d(knn_feat, (1, 1)) for knn_feat in knn_feats]
            feat = feat.view(feat.size(0), -1)  # (1B)xC
            knn_feats = [knn_feat.view(knn_feat.size(0), -1) for knn_feat in knn_feats]  # K (1B)xC
            feat_norm = nn.functional.normalize(feat, dim=1)
            knn_feats_norm = [nn.functional.normalize(knn_feat, dim=1) for knn_feat in knn_feats]
            # smaps: a list containing K similarity matrix (BxB Tensor)
            smaps = [torch.mm(feat_norm, knn_feat_norm.transpose(0, 1)) for knn_feat_norm in knn_feats_norm]  # K BxB
            top_query_inds = []
            top_key_inds = []
            for smap in smaps:
                topk_num = int(self.topk_bbox_ratio * smap.size(0))
                top_value, top_ind = torch.topk(smap.flatten(), topk_num if topk_num > 0 else 1)
                top_query_ind = top_ind // smap.size(1)
                top_key_ind = top_ind % smap.size(1)
                top_query_inds.append(top_query_ind)
                top_key_inds.append(top_key_ind)
            bbox = bbox.view(bbox.size(0) * bbox.size(1), bbox.size(2))  # (1B)x4
            knn_bboxes = [knn_bbox.view(
                knn_bbox.size(0) * knn_bbox.size(1), knn_bbox.size(2)) for knn_bbox in knn_bboxes]  # K (1B)x4
            topk_box_pairs_list = [torch.cat((bbox[qind], kbox[kind]), dim=1).cpu()
                for kbox, qind, kind in zip(knn_bboxes, top_query_inds, top_key_inds)]  # K (topk_bbox_num)x8
            knn_bbox_keys = ['{}nn_bbox'.format(k) for k in range(len(topk_box_pairs_list))]
            dict1 = dict(intra_bbox=bbox.cpu())
            dict2 = dict(zip(knn_bbox_keys, topk_box_pairs_list))
        # intra_bbox: Bx4, inter_bbox: K (topk_bbox_num)x8
        # B is the number of filtered bboxes, K is the number of knn images,
        return {**dict1, **dict2}

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
