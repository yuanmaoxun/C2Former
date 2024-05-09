
_base_ = [
    '../_base_/datasets/dronevehicle.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='Two_Stream_S2ANet',
    backbone=dict(
        type='C2FormerResNet',
        fmap_size=(128, 160),
        dims_in=[256, 512, 1024, 2048],
        dims_out=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        cca_strides=[3, 3, 3, 3],
        groups=[1, 2, 3, 6],
        offset_range_factor=[2, 2, 2, 2],
        no_offs=[False, False, False, False],
        attn_drop_rate=0.0,
        drop_rate=0.0,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        pretrained='pretrain_weights/resnet50-2stream.pth',
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    fam_head=dict(
        type='RotatedRetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
            # scales=[2],
            # ratios=[2.0],
            # strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    align_cfgs=dict(
        type='AlignConv',
        kernel_size=3,
        channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    odm_head=dict(
        type='ODMRefineHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    train_cfg=dict(
        fam_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                # pos_iou_thr=0.5,
                # neg_iou_thr=0.3,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        odm_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                # pos_iou_thr=0.7,
                # neg_iou_thr=0.5,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000
        # nms_pre=6000,
        # min_bbox_size=0,
        # score_thr=0.01,
        # nms=dict(iou_thr=0.425),
        # max_per_img=1000
    )
)

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(512, 640)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline, version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)