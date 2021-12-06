_base_ = '../../../../base.py'

# model settings
model = dict(type='SelectiveSearch')

# dataset settings
data_source_cfg = dict(
    type='COCOSelectiveSearchJson',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_json = 'data/coco/annotations/image_info_unlabeled2017.json'
data_train_root = 'data/coco/unlabeled2017'
dataset_type = 'SelectiveSearchDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data = dict(
    imgs_per_gpu=1,  # support single-image single-gpu inference only
    workers_per_gpu=8,
    val=dict(
        type=dataset_type,
        data_source=dict(
            json_file=data_train_json, root=data_train_root,
            **data_source_cfg),
        method='fast',
        min_size=None,
        max_ratio=None,
        topN=None
    ))
