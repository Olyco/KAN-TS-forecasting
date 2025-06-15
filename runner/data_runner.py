import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class GeneralTSFDataset(Dataset):
    def __init__(self, hist_len, pred_len, variable, time_feature, model_name, include_time_feature):
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.variable = variable
        self.time_feature = time_feature
        self.model_name = model_name
        self.include_time_feature = include_time_feature

    def __getitem__(self, index):
        hist_start = index
        hist_end = index + self.hist_len
        pred_end = hist_end + self.pred_len

        var_x = self.variable[hist_start:hist_end, ...]
        tf_x = self.time_feature[hist_start:hist_end, ...]

        var_y = self.variable[hist_end:pred_end, ...]
        tf_y = self.time_feature[hist_end:pred_end, ...]

        if self.include_time_feature: # static features should not be forecasted?
          # concatenate all features
          var_x = np.concatenate((var_x, tf_x), axis=-1)
          var_y = np.concatenate((var_y, tf_y), axis=-1)

        
        if self.model_name == "KAN":
          var_x = var_x.reshape(-1)
          var_y = var_y.reshape(-1)

        return var_x, var_y

    def __len__(self):
        return len(self.variable) - (self.hist_len + self.pred_len) + 1


class DataInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.hist_len = kwargs['hist_len']
        self.pred_len = kwargs['pred_len']
        self.norm_variable = kwargs['norm_variable']
        self.norm_time_feature = kwargs['norm_time_feature']
        self.train_len, self.val_len, self.test_len = kwargs['data_split']
        self.time_feature_cls = kwargs['time_feature_cls']
        self.file_format = kwargs['file_format']

        self.data_path = os.path.join(kwargs['data_root'], "{}.{}".format(kwargs['dataset_name'], self.file_format))
        self.config = kwargs

        self.variable, self.time_feature, self.scaler = self.__read_data__()

    def __read_data__(self):
        if self.file_format == "npz":
            data = np.load(self.data_path)
            variable = data['variable']
        elif self.file_format == "csv":
            data = pd.read_csv(self.data_path)
            data = data.rename(columns={'date': 'timestamp'})
            variable = data.iloc[:, 1:self.config['var_cut'] + 1].to_numpy()

        timestamp = pd.DatetimeIndex(data['timestamp'])

        # scale data
        scaler = MinMaxScaler()
        if self.norm_variable:
            scaler.fit(variable[:self.train_len])

        # time_feature
        time_feature = []
        for tf_cls in self.time_feature_cls:
            if tf_cls == "tod":
                tod_size = int((24 * 60) / self.config['freq']) - 1
                tod = np.array(list(map(lambda x: ((60 * x.hour + x.minute) / self.config['freq']), timestamp)))
                if self.norm_time_feature:
                    time_feature.append(tod / tod_size)
                else:
                    time_feature.append(tod)
            elif tf_cls == "dow":
                dow_size = 7 - 1
                dow = np.array(timestamp.dayofweek)  # 0 ~ 6
                if self.norm_time_feature:
                    time_feature.append(dow / dow_size)
                else:
                    time_feature.append(dow)
            elif tf_cls == "dom":
                dom_size = 31 - 1
                dom = np.array(timestamp.day) - 1  # 0 ~ 30
                if self.norm_time_feature:
                    time_feature.append(dom / dom_size)
                else:
                    time_feature.append(dom)
            elif tf_cls == "doy":
                doy_size = 366 - 1
                doy = np.array(timestamp.dayofyear) - 1  # 0 ~ 181
                if self.norm_time_feature:
                    time_feature.append(doy / doy_size)
                else:
                    time_feature.append(doy)
            else:
                raise NotImplementedError

        return variable, np.stack(time_feature, axis=-1), scaler
  

    def train_dataloader(self):
        scaled_train_var = self.scaler.transform(self.variable[:self.train_len])
        return DataLoader(
            dataset=GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                scaled_train_var,
                self.time_feature[:self.train_len].copy(),
                self.config["model_name"],
                self.config['include_time_feature'],
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True
        )

    def val_dataloader(self):
        scaled_val_var = self.scaler.transform(self.variable[self.train_len - self.hist_len:self.train_len + self.val_len])
        return DataLoader(
            dataset=GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                scaled_val_var,
                self.time_feature[self.train_len - self.hist_len:self.train_len + self.val_len].copy(),
                self.config["model_name"],
                self.config['include_time_feature'],
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self):
        scaled_test_var = self.scaler.transform(self.variable[self.train_len + self.val_len - self.hist_len:])
        return DataLoader(
            dataset=GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                scaled_test_var,
                self.time_feature[self.train_len + self.val_len - self.hist_len:].copy(),
                self.config["model_name"],
                self.config['include_time_feature'],
            ),
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False
        )
