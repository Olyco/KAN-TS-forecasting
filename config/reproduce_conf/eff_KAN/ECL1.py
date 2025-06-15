exp_conf = dict(
    model_name="eff_KAN",
    dataset_name='ECL',
    var_cut=1,
    data_split=[21043, 2630, 2631],

    hist_len=48,
    pred_len=24,

    max_epochs=150,

    layers_hidden=[48, 20, 20, 24],
    grid_size=5,#
    spline_order=4,#

    lr=0.001,
)