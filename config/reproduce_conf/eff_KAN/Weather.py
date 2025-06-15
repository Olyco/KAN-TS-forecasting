exp_conf = dict(
    model_name="eff_KAN",
    dataset_name='Weather',

    hist_len=24,
    pred_len=6,

    max_epochs=50,

    layers_hidden=[24, 2, 6],
    grid_size=3,
    spline_order=3,

    lr=0.0001,
)