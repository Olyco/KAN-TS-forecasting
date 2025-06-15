import argparse
import importlib
import os
import sys
from datetime import datetime
import pytz
import time

import lightning.pytorch as L
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from runner.data_runner import DataInterface
from runner.exp_runner import LTSFRunner


def load_module_from_path(module_name, exp_conf_path):
    spec = importlib.util.spec_from_file_location(module_name, exp_conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_config(exp_conf_path):
    # load exp_conf
    exp_conf = load_module_from_path("exp_conf", exp_conf_path).exp_conf

    # load task_conf
    task_conf_module = importlib.import_module('config.base_conf.task')
    task_conf = task_conf_module.task_conf

    # load data_conf
    data_conf_module = importlib.import_module('config.base_conf.datasets')
    data_conf = eval('data_conf_module.{}_conf'.format(exp_conf['dataset_name']))

    # combine
    fused_conf = {**task_conf, **data_conf}
    fused_conf.update(exp_conf)

    return fused_conf


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

    if conf["model_name"] == "KAN":
        conf["ckpt_path"] = os.path.join(conf["exp_dir"], "model")

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

    data_module = DataInterface(**conf)
    model = LTSFRunner(**conf)
    print(ModelSummary(model, max_depth=-1))
    print(model.hparams)


    start = time.time()
    trainer.fit(model=model, datamodule=data_module)
    end = time.time()
    train_time = end-start
    print(f"Training time: {train_time:.03f} s ({(train_time / 60):.03f} min, {(train_time / 3600):.03f} h)")

    # if conf["model_name"] == "KAN":
    #     model.model.saveckpt()

    trainer.test(model, datamodule=data_module, ckpt_path='best')


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
