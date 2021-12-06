_base_ = '../../../../../base.py'

# model settings
model = dict(
    type='Correspondence',
    pretrained=None,
    base_momentum=0.99,
    knn_image_num=10,
    topk_bbox_ratio=0.1,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckSimCLR',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2,
        sync_bn=True,
        with_bias=False,
        with_last_bn=False,
        with_avg_pool=True),
    head=dict(type='LatentPredictHead',
        predictor=dict(type='NonLinearNeckSimCLR',
            in_channels=256, hid_channels=4096,
            out_channels=256, num_layers=2, sync_bn=True,
            with_bias=False, with_last_bn=False, with_avg_pool=False)))
# dataset settings
data_source_cfg = dict(
    type='COCOCorrespondenceJson',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_knn_json = 'data/cocoplus/meta/trainplus2017_10nn_instance.json'
data_train_ss_json = ['data/coco/meta/train2017_selective_search_proposal.json',
                      'data/cocoplus/meta/unlabeled2017_selective_search_proposal.json']
data_train_root = 'data/cocoplus/trainplus2017'
dataset_type = 'CorrespondenceDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
format_pipeline = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
]
data = dict(
    imgs_per_gpu=1,  # support single-image single-gpu inference only
    workers_per_gpu=0,
    val=dict(
        type=dataset_type,
        data_source=dict(
            knn_json_file=data_train_knn_json,
            ss_json_file=data_train_ss_json,
            root=data_train_root,
            knn_image_num=10,
            part=4,  # [0, num_parts)
            num_parts=10,
            data_len=241690,  # train2017 (118287) + unlabeled2017 (123403)
            **data_source_cfg),
        format_pipeline=format_pipeline,
        patch_size=224,
        min_size=96,
        max_ratio=3,
        max_iou_thr=0.5,
        topN=100,
        knn_image_num=10,
        topk_bbox_ratio=0.1,
        prefetch=False
    ))
# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]
# optimizer
optimizer = dict(type='SGD', lr=0.4, weight_decay=0.0001, momentum=0.9)
# apex
use_fp16 = False
optimizer_config = dict(update_interval=update_interval, use_fp16=use_fp16)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=4,
    warmup_ratio=0.0001, # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 800
