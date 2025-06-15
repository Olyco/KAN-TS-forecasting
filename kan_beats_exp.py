import argparse
import os
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

import torch
import lightning.pytorch as L
import torch.optim.lr_scheduler as lrs
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from pytorch_forecasting.data import NaNLabelEncoder, GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet, Baseline, NBeats, GroupNormalizer, MultiNormalizer, EncoderNormalizer
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch.tuner import Tuner

from train import load_config
from model.KAN_BEATS import KANBeats


def prepare_data(config):
    data_path = os.path.join(config['data_root'], "{}.{}".format(config['dataset_name'], config['file_format']))
    data = pd.read_csv(data_path)
    data = data.rename(columns={'date': 'timestamp'})
    variable = data.iloc[:, 1:config['var_cut'] + 1].to_numpy()

    scaler = MinMaxScaler()
    if config['norm_variable']:
        scaler.fit(variable[:config['data_split'][0]])

        var = scaler.transform(variable)
    else:
      var = variable

    concatenated_df = pd.DataFrame()
    for i in range(var.shape[1]):
        ts = pd.DataFrame(data={'variable': var.T[i], 'id': [i for _ in range(var.shape[0])]})
        concatenated_df = pd.concat([concatenated_df, ts])
    concatenated_df = concatenated_df.reset_index()

    train_cut = concatenated_df.loc[concatenated_df['index'] < config['data_split'][0]]
    print(train_cut)

    val_cut = concatenated_df.loc[concatenated_df['index'].isin(range(config['data_split'][0], config['data_split'][0] + config['data_split'][1]))]
    print(val_cut)

    test_cut = concatenated_df.loc[concatenated_df['index'] >= config['data_split'][0] + config['data_split'][1]]
    print(test_cut)

    train_data = TimeSeriesDataSet(
        data=train_cut,
        time_idx="index",
        target="variable",
        categorical_encoders={"id": NaNLabelEncoder().fit(train_cut.id)},
        group_ids=['id'],
        max_encoder_length=config['hist_len'],
        max_prediction_length=config['pred_len'],
        time_varying_unknown_reals=["variable"],

        # target_normalizer=GroupNormalizer(),
        target_normalizer=None,
        scalers=None,
    )
    print(train_data.get_parameters())
    val_data = TimeSeriesDataSet.from_dataset(train_data, val_cut)
    test_data = TimeSeriesDataSet.from_dataset(train_data, test_cut)

    train_dataloader = train_data.to_dataloader(
        train=True, 
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'], 
        # batch_sampler=config['batch_sampler'],
    )
    x, y = next(iter(train_dataloader))
    # print(x)
    for key, value in x.items():
        print(f"\t{key} = {value.size()}")

    val_dataloader = val_data.to_dataloader(
        train=False, 
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
        )
    test_dataloader = test_data.to_dataloader(
        train=False, 
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
    )

    return train_dataloader, val_dataloader, test_dataloader, train_data


def train_func(hyper_conf, conf):
    if hyper_conf is not None: # add training config
        for k, v in hyper_conf.items():
            conf[k] = v
    conf['exp_time'] = datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d%m%y_%H%M")

    L.seed_everything(conf["seed"])
    save_dir = os.path.join(conf["save_root"], '{}_{}'.format(conf["model_name"], conf["dataset_name"]))
    
    if "use_wandb" in conf and conf["use_wandb"]:
        run_logger = WandbLogger(save_dir=save_dir, name=conf["exp_time"], version='seed_{}'.format(conf["seed"]))
    else:
        run_logger = CSVLogger(save_dir=save_dir, name=conf["exp_time"], version='seed_{}'.format(conf["seed"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["exp_time"], 'seed_{}'.format(conf["seed"]))


    callbacks = [
        ModelCheckpoint(
            monitor=conf["val_metric"],
            mode="min",
            save_top_k=1,
            save_last=False,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor=conf["val_metric"],
            mode='min',
            patience=conf["es_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]

    trainer = L.Trainer(
        devices=conf["devices"],
        precision=conf["precision"] if "precision" in conf else "32-true",
        logger=run_logger,
        callbacks=callbacks,
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf["gradient_clip_algorithm"] if "gradient_clip_algorithm" in conf else "norm",
        gradient_clip_val=conf["gradient_clip_val"],
        default_root_dir=conf["save_root"],
        deterministic=True,
    )

    train_dataloader, val_dataloader, test_dataloader, train_data = prepare_data(conf)

    if conf['model_name'] == "KAN_BEATS":
        model = KANBeats.from_dataset(
            train_data,
            grid_size=conf['grid_size'],
            spline_order=conf['spline_order'],
            stack_types=conf['stack_types'],
            num_blocks=conf['num_blocks'],
            num_block_layers=conf['num_block_layers'],
            widths=conf['widths'],
            sharing=conf['sharing'],
            expansion_coefficient_lengths=conf['expansion_coefficient_lengths'],
            backcast_loss_ratio=conf['backcast_loss_ratio'],
            loss=conf['loss'],

            logging_metrics=conf['logging_metrics'],
            optimizer=conf['optimizer'],

            log_interval=3,
            log_gradient_flow=conf['log_gradient_flow'],
            weight_decay=conf['weight_decay'],
            learning_rate=conf['learning_rate'],
            reduce_on_plateau_patience=4,
            reduce_on_plateau_min_lr=conf['reduce_on_plateau_min_lr'],
            reduce_on_plateau_reduction=10.0,
            )
    elif conf['model_name'] == "N_BEATS":
        model = NBeats.from_dataset(
            train_data,
            stack_types=conf['stack_types'],
            num_blocks=conf['num_blocks'],
            num_block_layers=conf['num_block_layers'],
            widths=conf['widths'],
            sharing=conf['sharing'],
            expansion_coefficient_lengths=conf['expansion_coefficient_lengths'],
            backcast_loss_ratio=conf['backcast_loss_ratio'],
            loss=conf['loss'],

            logging_metrics=conf['logging_metrics'],
            optimizer=conf['optimizer'],

            log_interval=3,
            log_gradient_flow=conf['log_gradient_flow'],
            weight_decay=conf['weight_decay'],
            learning_rate=conf['learning_rate'],
            reduce_on_plateau_patience=2,
            reduce_on_plateau_min_lr=conf['reduce_on_plateau_min_lr'],
            reduce_on_plateau_reduction=2.0,
        )

    print(ModelSummary(model, max_depth=-1))

    #
    res = Tuner(trainer).lr_find(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-3)
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=False, suggest=True)
    fig.savefig(f"{conf['exp_dir']}/lr_{res.suggestion():.010f}.png")
    model.hparams.learning_rate = res.suggestion()
    
    #change lr everywhere
    for g in trainer.optimizers[0].param_groups:
      g['lr'] = model.hparams.learning_rate
    lr_state = trainer.lr_scheduler_configs[0].scheduler.state_dict()
    lr_state['_last_lr'] = model.hparams.learning_rate
    trainer.lr_scheduler_configs[0].scheduler.load_state_dict(lr_state)

    print(model.hparams)
    print(trainer.optimizers[0])
    print(trainer.lr_scheduler_configs[0].scheduler.state_dict())
    #

    start = time.time()
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
        )
    end = time.time()
    train_time = end-start
    print(f"Training time: {train_time:.03f} s ({(train_time / 60):.03f} min, {(train_time / 3600):.03f} h)")

    trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')

    # Visualize Predictions
    preds = model.predict(test_dataloader)

    # Aggregate inputs, ground truth and classes into tensor alligned with predictions
    input_list, true_list, class_list = [], [], []
    for x, y in test_dataloader: 
        input_list.append(x["encoder_target"])
        true_list.append(y[0])
        class_list.append(x["groups"])

    inputs = torch.cat(input_list)
    trues = torch.cat(true_list)
    classes = torch.cat(class_list)

    # Select indices of samples to visualize
    n_samples = 10
    ss_indices = np.random.choice(range(preds.shape[0]), n_samples, replace=False)
    ss_pred = preds[ss_indices]
    ss_true = trues[ss_indices]
    ss_input = inputs[ss_indices]
    ss_class = classes[ss_indices]

    f, axarr = plt.subplots(n_samples, 1, figsize=(20, 80))
    for i in range(n_samples):
        series_preds = ss_pred[i, :].squeeze()
        series_trues = ss_true[i, :].squeeze()
        series_inputs = ss_input[i, :].squeeze()

        feat_name = str(ss_class[i].item())
        
        input_len = series_inputs.shape[0]
        pred_gt_len = series_preds.shape[0]
        input_x = np.array([i for i in range(input_len)])
        x = np.array([i for i in range(input_len, input_len+pred_gt_len)])
        axarr[i].plot(x, series_preds, c="green", label="predictions")
        axarr[i].plot(x, series_trues, c="red", label="ground truth")
        axarr[i].plot(input_x, series_inputs, c="blue", label="input")
        axarr[i].legend()
        axarr[i].set_title(feat_name)
    f.savefig(f"{conf['exp_dir']}/preds.png")

    # loss
    loss_f, ax = plt.subplots(figsize=(10, 10))

    df = pd.read_csv(f"{conf['exp_dir']}/metrics.csv")
    loss_epoch = df[df['train_loss_epoch'].notnull()]['train_loss_epoch']
    val_loss_epoch = df[df['val_loss'].notnull()]['val_loss']


    epochs = np.arange(0,loss_epoch.shape[0])
    ax.plot(epochs, val_loss_epoch, label='validation')
    ax.plot(epochs, loss_epoch, label='train')
    ax.set_title('model loss')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend(loc='upper right')
    loss_f.savefig(f"{conf['exp_dir']}/loss.png")

    # lr
    lr_f, ax = plt.subplots(figsize=(10, 10))

    df = pd.read_csv(f"{conf['exp_dir']}/metrics.csv")
    lr_epoch = df[df['lr-Adam'].notnull()]['lr-Adam']

    epochs = np.arange(0,lr_epoch.shape[0])
    ax.plot(epochs, lr_epoch)
    ax.set_title('learning rate')
    ax.set_ylabel('lr')
    ax.set_xlabel('epoch')
    ax.legend(loc='upper right')
    lr_f.savefig(f"{conf['exp_dir']}/lr.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str) # experiment config
    parser.add_argument("-d", "--data_root", default="drive/MyDrive/VKR/Data/Time series", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="drive/MyDrive/VKR/Results/save", help="save root")
    parser.add_argument("--devices", default='auto', type=str, help="device' id to use")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    args = parser.parse_args()

    training_conf = {
        "seed": int(args.seed),
        "data_root": args.data_root,
        "save_root": args.save_root,
        "devices": args.devices,
        "use_wandb": args.use_wandb,
    }
    init_exp_conf = load_config(args.config)
    train_func(training_conf, init_exp_conf)