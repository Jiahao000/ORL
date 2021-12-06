import copy
_base_ = '../../../../base.py'

# model settings
model = dict(
    type='BYOL',
    pretrained=None,
    base_momentum=0.99,
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
    type='COCO',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/cocoplus/meta/trainplus2017.txt'
data_train_root = 'data/cocoplus/trainplus2017'
dataset_type = 'BYOLDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
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
extract_pipeline = [
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
]
# prefetch
prefetch = True
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
    extract_pipeline.extend([dict(type='ToTensor'), dict(type='Normalize', **img_norm_cfg)])
train_pipeline1 = copy.deepcopy(train_pipeline)
train_pipeline2 = copy.deepcopy(train_pipeline)
train_pipeline2[4]['p'] = 0.1 # gaussian blur
train_pipeline2[5]['p'] = 0.2 # solarization

data = dict(
    imgs_per_gpu=64,  # total 128*2(gpu)*1(interval)=256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
        pipeline1=train_pipeline1,
        pipeline2=train_pipeline2,
        prefetch=prefetch,
    ))
# additional hooks
update_interval = 1  # interval for accumulate gradient
custom_hooks = [
    dict(type='BYOLHook', end_momentum=1., update_interval=update_interval),
    dict(
        type='ExtractorHook',
        extractor=dict(
            imgs_per_gpu=256,
            workers_per_gpu=6,
            dataset=dict(
                type='ExtractDataset',
                data_source=dict(
                    list_file=data_train_list,
                    root=data_train_root,
                    **data_source_cfg),
                pipeline=extract_pipeline,
                prefetch=prefetch),
            prefetch=prefetch,
            img_norm_cfg=img_norm_cfg),
        normalize=True)
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
