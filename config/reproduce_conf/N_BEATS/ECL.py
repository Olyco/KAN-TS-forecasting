from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
from torch import nn
from torcheval.metrics import R2Score

class customR2Score(MultiHorizonMetric):

    def loss(self, y_pred, target):
        metric = R2Score(multioutput="uniform_average")
        metric.update(self.to_prediction(y_pred), target)
        r2 = metric.compute()
        return r2

exp_conf = dict(
    model_name="N_BEATS",
    dataset_name='ECL',
    var_cut=10,

    norm_variable=True,
    batch_sampler='synchronized',

    hist_len=24,
    pred_len=12,

    max_epochs=3,

    stack_types=['generic'],
    num_blocks=[2],
    num_block_layers=[2],
    widths=[64],
    sharing=False,
    expansion_coefficient_lengths=[32],
    backcast_loss_ratio=0.1,
    loss=MAE(),
    logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE(), customR2Score()]),

    val_metric="val_loss",
    test_metric="test_mae",
    batch_size=128,

    log_interval=10,
    log_gradient_flow=False,
    weight_decay=1e-2,

    lr=0.001,
    learning_rate=0.001,
    reduce_on_plateau_patience=3,#
    reduce_on_plateau_min_lr=1e-12,
)