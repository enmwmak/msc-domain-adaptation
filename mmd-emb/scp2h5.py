"""
Read Kaldi .scp files and convert to .h5
Require to setup PYTHONPATH as follows:
export PYTHONPATH=$HOME/so/python/kaldi_io
kaldio_io can be downloaded from https://github.com/vesis84/kaldi-io-for-python

@author: M.W. Mak
@date: Nov. 2020
"""

import kaldi_io
import numpy as np
import sys
import h5py as h5
from tqdm import tqdm

    # Stripe or append text in key 
def adjust_key(key, strip_after_chars=None, append_str=None):
    new_key = key.copy()
    if strip_after_chars is not None:
        for i in range(len(key)):
            if key[i][0:3] == 'sw_':                  # Switchboard speaker id
                fields = key[i].split('_')
                new_key[i] = fields[0] + '_' + fields[1].split('_')[0].split('-')[0]
            else:                                     # Others speaker id
                for ch in strip_after_chars:                
                    new_key[i] = new_key[i].split(ch)[0]
    if append_str is not None:
        for i in range(len(key)):
            new_key[i] = new_key[i] + append_str
    return new_key

# Save key and val to h5 file
def save_h5(spk_ids, mfcc, positions, utt_ids, h5_file, compression=None):
    unicode = h5.special_dtype(vlen=str)
    print('Saving %s' % h5_file)
    f = h5.File(h5_file, 'w')
    f.create_dataset('spk_ids', data=spk_ids.astype(unicode), compression=compression)
    f.create_dataset('mfcc', data=mfcc, compression=compression)
    f.create_dataset('positions', data=positions, compression=compression)
    f.create_dataset('utt_ids', data=utt_ids.astype(unicode), compression=compression)    
    f.close()

    
# Convert all .ark files (indexed by .scp) in a given dir to a single .h5 file
# The .h5 file should contain the  fields: 'mfcc', 'positions', 'spk_ids, 'utt_ids'
# positions is an N x 2 matrix, with positions[i,0] contains the starting position
# and positions[:,1] containing the ending position of utterance i.    
def scp2h5(scpfile, h5file, strip_after_chars=None, append_str=None, compression=None):
    utt_ids, val, pos = load_ark_from_scp(scpfile)
    mfcc = np.vstack(val)
    positions = np.vstack(pos)
    spk_ids = np.asarray(adjust_key(utt_ids, strip_after_chars=strip_after_chars, 
                                    append_str=append_str))
    utt_ids = np.asarray(utt_ids)
    save_h5(spk_ids, mfcc, positions, utt_ids, h5file, compression=compression)


# Load matrix-type .ark files based on the .scp file
def load_ark_from_scp(scpfile):
    key = list()
    val = list()
    pos = list()
    start = 0
    num_lines = sum(1 for line in open(scpfile))
    pbar = tqdm(total=num_lines)
    for i, (k, v) in enumerate(kaldi_io.read_mat_scp(scpfile)):
        if i % 100 == 0:
            pbar.update(100)
        n_frames = v.shape[0]
        end = start + n_frames - 1
        key.append(k)
        val.append(v)
        pos.append([start, end])
        start = end + 1
    pbar.close()
    return key, val, pos

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <.scp file> <.h5 file> <strip_after_chars>")
        print("Example: python3 scp2h5.py ../tmp/test.scp ../tmp/test.h5 _,-")
        exit()
    sac = sys.argv[3].split(',')
    scp2h5(sys.argv[1], sys.argv[2], strip_after_chars=sac, compression='gzip')


if __name__ == '__main__':
    main()
