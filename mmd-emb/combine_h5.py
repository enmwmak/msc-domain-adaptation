#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:48 2020
@author: mwmak

Combine the .h5 files in a given folder and save the combined file in .h5 using
virtual datasets.
"""
import sys
import pathlib
import h5py
import numpy as np

def combine_h5(h5dir, out_h5file):
    filelist = list(pathlib.Path(h5dir).glob('*.h5'))
    unicode = h5py.special_dtype(vlen=str)
    n_files = len(filelist)

    # Get total no. of utts (spks) and no. of frames in .h5 files in the folder
    n_utts = list()
    n_frames = list()
    for i in range(n_files):
        with h5py.File(filelist[i], 'r') as f:
            n_utts.append(len(f['utt_ids']))
            n_frames.append(f['mfcc'].shape[0])
            mfcc_dim = f['mfcc'].shape[1]
    tot_n_utts = np.sum(n_utts)
    tot_n_frames = np.sum(n_frames)    
    print(f"Total no. of utts = {tot_n_utts}")
    print(f"Total no. of frames = {tot_n_frames}")
    print(f"MFCC dim = {mfcc_dim}")

    # Assemble virtual dataset
    utt_layout = h5py.VirtualLayout(shape=(tot_n_utts,), dtype=unicode)
    spk_layout = h5py.VirtualLayout(shape=(tot_n_utts,), dtype=unicode)
    pos_layout = h5py.VirtualLayout(shape=(tot_n_utts,2), dtype="int64")
    mfc_layout = h5py.VirtualLayout(shape=(tot_n_frames, mfcc_dim), dtype="float32")    
    k1 = 0
    k2 = 0
    for i in range(n_files):
        print(f"Reading {filelist[i]}")
        range1 = range(k1, k1 + n_utts[i])
        range2 = range(k2, k2 + n_frames[i])
        print(f"spk_ids: {range1}")
        print(f"mfcc: {range2}")
        utt_layout[range1] = h5py.VirtualSource(filelist[i], "utt_ids", shape=(n_utts[i],))
        spk_layout[range1] = h5py.VirtualSource(filelist[i], "spk_ids", shape=(n_utts[i],))
        pos_layout[range1] = h5py.VirtualSource(filelist[i], "positions", shape=(n_utts[i],2))
        mfc_layout[range2] = h5py.VirtualSource(filelist[i], "mfcc", shape=(n_frames[i],mfcc_dim))
        k1 = k1 + n_utts[i]
        k2 = k2 + n_frames[i]

    # Add virtual dataset to output file
    with h5py.File(out_h5file, "w", libver="latest") as f:
        print(f"Writing combined file {out_h5file}")
        f.create_virtual_dataset("utt_ids", utt_layout, fillvalue=None)
        f.create_virtual_dataset("spk_ids", spk_layout, fillvalue=None)
        f.create_virtual_dataset("positions", pos_layout, fillvalue=None)
        f.create_virtual_dataset("mfcc", mfc_layout, fillvalue=None)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <h5 folder> <output .h5 file>")
        print("Example: python3 combine_h5.py ../tmp combine.h5")
        exit()
    combine_h5(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()