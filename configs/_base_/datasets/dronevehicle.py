# dataset settings
dataset_type = 'DroneVehicleDataset'  # load paired image
data_root = '/home/yuanmaoxun/Datasets/DroneVehiclUse/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(512, 640)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PairedImageDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_tir', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 640),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PairedImageDefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'img_tir'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_total/trainMatchedLabelTxtMVP_total',
        img_prefix=data_root + 'train_total/trainMatchedImg_total',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_total/valMatchedLabelTxtMVP_total',
        img_prefix=data_root + 'val_total/valMatchedImg_total',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_total/testMatchedLabelTxtMVP_total',
        img_prefix=data_root + 'test_total/testMatchedImg_total',
        pipeline=test_pipeline))