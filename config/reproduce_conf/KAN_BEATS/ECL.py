from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
        
exp_conf = dict(
    model_name="KAN_BEATS",
    dataset_name='ECL1_short',
    var_cut=1,

    norm_variable=True,
    batch_sampler='synchronized',

    hist_len=10,
    pred_len=5,

    max_epochs=5,

    grid_size=3,
    spline_order=3,

    stack_types=['generic'],
    num_blocks=[2],
    num_block_layers=[2],
    widths=[5],
    sharing=False,
    expansion_coefficient_lengths=[10],
    backcast_loss_ratio=0.1,
    loss=MAE(),
    # logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE(), customR2Score()]),

    val_metric="val_loss",
    test_metric="test_mae",
    batch_size=64,

    log_interval=10,
    log_gradient_flow=False,
    weight_decay=1e-2,

    learning_rate=0.001,
    lr=0.001,
    reduce_on_plateau_patience=1,#
    reduce_on_plateau_min_lr=1e-12,
)