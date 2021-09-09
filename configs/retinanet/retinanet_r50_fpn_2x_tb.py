_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/tb_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=2))

# learning policy
optimizer = dict(lr=0.005)
lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=0.001,
        step=[25600, 32000])

# Runner type
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=38400)
checkpoint_config = dict(interval=2000)
evaluation = dict(interval=2000, metric='bbox')
