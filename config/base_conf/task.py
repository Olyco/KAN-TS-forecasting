task_conf = dict(
    hist_len=48,
    pred_len=12,

    batch_size=64,
    max_epochs=10,
    lr=0.0001,
    optimizer="Adam",
    optimizer_betas=(0.95, 0.9),
    optimizer_weight_decay=1e-5,
    lr_scheduler='ReduceLROnPlateau',
    lr_step_size=1,
    lr_gamma=0.5,
    gradient_clip_val=5,
    val_metric="val/loss",
    test_metric="test/mae",
    es_patience=10,
    lrs_factor=0.5,
    lrs_patience=5,

    norm_variable = True,
    norm_time_feature=False,
    include_time_feature = False,
    time_feature_cls=["tod", "dow"],
    file_format="csv",

    num_workers=2,
)
