exp_conf = dict(
    model_name="eff_KAN",
    dataset_name='ECL',

    hist_len=24,
    pred_len=6,

    max_epochs=50,

    layers_hidden=[24, 5, 6],
    grid_size=3,#
    spline_order=3,#

    lr=0.0001,
)