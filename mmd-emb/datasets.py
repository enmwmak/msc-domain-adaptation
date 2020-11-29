from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


class ExtractDataset(Dataset):
    def __init__(self, f, meta_data=None):
        self.mfccs = f['mfcc']
        if meta_data:
            with h5py.File(meta_data, 'r') as f_meta:
                self.positions = f_meta['positions'][:]
                self.spk_ids = f_meta['spk_ids'][:]
                self.utt_ids = f_meta['utt_ids'][:]
        else:
            self.positions = f['positions'][:]
            self.spk_ids = f['spk_ids'][:]
            self.utt_ids = f['utt_ids'][:]

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), spk_id, utt_id

    def __len__(self):
        return len(self.positions)


class PartialExtractDataset(Dataset):
    def __init__(self, df, mfcc):
        self.mfccs = mfcc
        self.positions = df[['starts', 'ends']].values
        self.spk_ids = df['spk_ids'].values
        self.utt_ids = df['utt_ids'].values

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), spk_id, utt_id

    def __len__(self):
        return len(self.positions)


class NestedPartialExtractDataset(Dataset):
    def __init__(self, df, mfcc, chunk=50):
        self.mfccs = mfcc
        positions = df[['starts', 'ends']].values
        n_subset = len(positions) // chunk + 1
        self.positions = np.array_split(positions, n_subset)

        self.spk_ids = np.array_split(df['spk_ids'].values, n_subset)
        self.utt_ids = np.array_split(df['utt_ids'].values, n_subset)
        self.chunk = chunk

    def __getitem__(self, idx):
        position = self.positions[idx]
        start, end = position.min(), position.max()
        mfcc = self.mfccs[start:end]

        data = {
            'positions': position-position.min(),
            'spk_ids': self.spk_ids[idx],
            'utt_ids': self.utt_ids[idx],
            'mfcc': mfcc,
        }

        inner = InnerDataset(data)
        inner_loader = torch.utils.data.DataLoader(inner,
                                                   batch_size=1)
        return inner_loader

    def __len__(self):
        return len(self.positions)


class InnerDataset(Dataset):
    def __init__(self, data):
        assert type(data) in [dict, h5py._hl.files.File]
        self.mfccs = data['mfcc']
        self.positions = data['positions'][:]
        self.spk_ids = data['spk_ids'][:]
        self.utt_ids = data['utt_ids'][:]

    def __getitem__(self, idx):
        start, end = self.positions[idx]
        spk_id = self.spk_ids[idx]
        utt_id = self.utt_ids[idx]
        mfcc = self.mfccs[start:end].T
        return torch.tensor(mfcc), torch.tensor(spk_id), utt_id

    def __len__(self):
        return len(self.positions)


class RandomSampleDataset(Dataset):
    def __init__(
        self,
        f,
        sample_length_range,
        n_blocks,
        sample_per_epoch,
        min_frames_per_utt,
        meta_data_file=None,
        balance_class=True,
        min_utts_per_spk=8,
    ):
        self.n_blocks = n_blocks
        if meta_data_file:
            with h5py.File(meta_data_file, 'r') as f_meta:
                self.df = self.get_df(f_meta, min_utts_per_spk, min_frames_per_utt)
        else:
            self.df = self.get_df(f, min_utts_per_spk, min_frames_per_utt)

        self.smp_per_epo = sample_per_epoch
        self.mfccs = f["mfcc"]
        self.balance_class = balance_class
        self.smp_len_min, self.smp_len_max = sample_length_range

    def sample(self):
        np.random.seed()
        sample_length = np.random.randint(low=self.smp_len_min, high=self.smp_len_max)
        if self.balance_class:
            prob = (1 / self.df["spk_ids"].nunique() / self.df["n_utts"]).values
            idxes = np.random.choice(len(self.df), p=prob, size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                self.df.iloc[idxes], sample_length
            )
        else:
            idxes = np.random.randint(len(self.df), size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                self.df.iloc[idxes], sample_length
            )

    def sample_original(self):
        np.random.seed()
        sample_length = np.random.randint(low=self.smp_len_min, high=self.smp_len_max)
        if self.balance_class:
            # remove augment data
            mask = ~self.df.utt_ids.str.endswith(('babble', 'music', 'noise', 'reverb'))
            df = self.df[mask]
            utt_counts = df.spk_ids.value_counts()
            df["n_utts"] = df.spk_ids.map(utt_counts)

            prob = (1 / df["spk_ids"].nunique() / df["n_utts"]).values
            idxes = np.random.choice(len(df), p=prob, size=self.smp_per_epo)
            self.spk_ids, self.utt_ids, self.positions = self.sample_segments(
                df.iloc[idxes], sample_length
            )
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        spk_id = self.spk_ids[i]
        utt_id = self.utt_ids[i]
        sub_positions = self.positions[i]
        mfccs = []
        for start, end in sub_positions:
            mfccs.append(self.mfccs[start:end])
        mfccs = np.concatenate(mfccs).T
        return torch.tensor(mfccs), torch.tensor(spk_id), utt_id

    def __len__(self):
        return self.smp_per_epo

    def sample_segments(self, df_smp, sample_length):
        spk_ids = df_smp.spk_ids.values
        utt_ids = df_smp.utt_ids.values
        positions = []
        for start, end in df_smp[["starts", "ends"]].values:
            sub_positions = []
            for _ in range(self.n_blocks):
                smp_start = np.random.randint(low=start, high=end - sample_length)
                smp_end = smp_start + sample_length
                sub_positions.append([smp_start, smp_end])
            positions.append(sub_positions)
        positions = np.array(positions)
        return spk_ids, utt_ids, positions

    @staticmethod
    def get_df(f, min_utts_per_spk, min_frames_per_utt):
        df = pd.DataFrame(
            {
                "spk_ids": f["spk_ids"][:],
                "utt_ids": f["utt_ids"][:],
                "starts": f["positions"][:, 0],
                "ends": f["positions"][:, 1],
            }
        )
        df = df[(df.ends - df.starts) > min_frames_per_utt]
        utt_counts = df.spk_ids.value_counts()
        df["n_utts"] = df.spk_ids.map(utt_counts)
        df = df[df.n_utts > min_utts_per_spk]
        df["spk_ids"] = LabelEncoder().fit_transform(df["spk_ids"])
        # df['utt_ids'] = LabelEncoder().fit_transform(df['utt_ids'])
        return df