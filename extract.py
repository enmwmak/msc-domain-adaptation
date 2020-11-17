import numpy as np
import h5py
import torch
from datasets import ExtractDataset

def extract_collate(batch):
    assert len(batch) == 1
    x = batch[0][0]
    spk_id = batch[0][1]
    utt_id = batch[0][2]
    return [x[None, ...], spk_id, utt_id]


def sequential_extract(model, mfcc_file, save_xvec_to):
    print(f"Reading mfcc from {mfcc_file}")
    with h5py.File(mfcc_file, "r") as fr:
        dset = ExtractDataset(fr)
        loader = torch.utils.data.DataLoader(
            dset, batch_size=1, shuffle=False, collate_fn=extract_collate
        )
        print(f"Saving xvectors to {save_xvec_to}")
        with h5py.File(save_xvec_to, "w") as fw:
            fw["X"], fw["spk_ids"], fw["spk_path"] = _extraction(model, loader)

def _extraction(model, loader):
    unicode = h5py.special_dtype(vlen=str)
    model.eval()
    X, spk_ids, utt_ids = [], [], []
    with torch.no_grad():
        for batch_idx, (mfcc, spk_id, utt_id) in enumerate(loader):
            mfcc = mfcc.to(torch.device(torch.cuda.current_device()))
            x = model.extract(mfcc)
            spk_ids.append(spk_id)
            utt_ids.append(utt_id)
            X.append(x)

        X = torch.cat(X).to("cpu").numpy()
        spk_ids = np.stack(spk_ids).astype(unicode)
        utt_ids = np.stack(utt_ids).astype(unicode)
    return X, spk_ids, utt_ids