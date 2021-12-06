import copy
_base_ = '../../../../base.py'

# model settings
model = dict(
    type='ORL',
    pretrained=None,
    base_momentum=0.99,
    global_loss_weight=1.,
    local_intra_loss_weight=1.,
    local_inter_loss_weight=1.,
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
    type='COCOORLJson',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_json = 'data/coco/meta/train2017_10nn_instance_correspondence.json'
data_train_root = 'data/coco/train2017'
dataset_type = 'ORLDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_image_pipeline = [
    dict(type='RandomResizedCrop', size=224, interpolation=2),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
train_patch_pipeline = [
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=1.),
    dict(type='RandomAppliedTrans',
         transforms=[dict(type='Solarization')], p=0.),
]
# prefetch
prefetch = True
if not prefetch:
    train_image_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    train_patch_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_image_pipeline1 = copy.deepcopy(train_image_pipeline)
train_image_pipeline2 = copy.deepcopy(train_image_pipeline)
train_patch_pipeline1 = copy.deepcopy(train_patch_pipeline)
train_patch_pipeline2 = copy.deepcopy(train_patch_pipeline)
train_image_pipeline2[4]['p'] = 0.1 # gaussian blur
train_image_pipeline2[5]['p'] = 0.2 # solarization
train_patch_pipeline2[3]['p'] = 0.1 # gaussian blur
train_patch_pipeline2[4]['p'] = 0.2 # solarization
data = dict(
    imgs_per_gpu=64,  # total 64*8(gpu)*1(interval)=512
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            json_file=data_train_json, root=data_train_root,
            topk_knn_image=10,
            **data_source_cfg),
        image_pipeline1=train_image_pipeline1,
        image_pipeline2=train_image_pipeline2,
        patch_pipeline1=train_patch_pipeline1,
        patch_pipeline2=train_patch_pipeline2,
        patch_size=96,
        interpolation=2,
        shift=(-0.5, 0.5),
        scale=(0.5, 2.),
        ratio=(0.5, 2.),
        iou_thr=0.5,
        attempt_num=200,
        prefetch=prefetch,
    ))
# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval)
]
# optimizer
optimizer = dict(type='SGD', lr=0.4, weight_decay=0.0001, momentum=0.9)
# apex
use_fp16 = True
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
