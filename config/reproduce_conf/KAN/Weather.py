exp_conf = dict(
    model_name="KAN",
    dataset_name='Weather',
    # var_cut=1,

    hist_len=24,
    pred_len=6,

    max_epochs=2,

    width=[24 * 20, 2, 6 * 20],
    grid=3,
    k=3,
    symbolic_enabled=False,
    auto_save=True,

    lr=0.0001,
)