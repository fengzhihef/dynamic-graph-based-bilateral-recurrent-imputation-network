import os
import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils import add_miss_mask_produce

class Fly(PandasDataset):
    def __init__(self):
        df, mask = self.load()
        super().__init__(dataframe=df, mask=mask, name='Fly', freq='5T', aggr='nearest')

    def load(self):
        path = os.path.join(datasets_path['Fly'], 'Fly.h5')
        df = pd.read_hdf(path)
        datetime_idx = sorted(df.index)
        date_range = pd.date_range(datetime_idx[0], datetime_idx[-1], freq='5T')
        df = df.reindex(index=date_range)
        mask = ~np.isnan(df.values)
        return df.astype('float32'), mask.astype('uint8')

    @property
    def mask(self):
        return self.df.values != 0. if self._mask is None else self._mask

class MissingValuesData(Fly):
    SEED = 56789

    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng(self.SEED)
        eval_mask = add_miss_mask_produce(self.mask, missing_rate=0.1)
        self.eval_mask = (self.mask & ~eval_mask).astype('uint8')

    @property
    def training_mask(self):
        processed_mask = self.process_mask(np.array(self.mask).copy())
        result_mask = processed_mask - self.eval_mask
        return result_mask

    def process_mask(self, mask):
        rows, cols = mask.shape
        processed_mask = mask.astype(float)

        # Forward and backward scanning to compute distances
        for direction in [1, -1]:
            for j in range(cols):
                nearest_one = None
                range_iter = range(rows) if direction == 1 else range(rows - 1, -1, -1)
                for i in range_iter:
                    if mask[i, j] == 1:
                        nearest_one = i
                    elif nearest_one is not None:
                        distance = abs(i - nearest_one)
                        value = np.exp(-distance)
                        current_value = processed_mask[i, j]
                        processed_mask[i, j] = max(value, current_value) if direction == -1 else value

        return processed_mask

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
        test_len = int(test_len * len(idx)) if test_len < 1 else test_len
        val_len = int(val_len * (len(idx) - test_len)) if val_len < 1 else val_len
        test_start = len(idx) - test_len
        val_start = test_start - val_len
        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]
