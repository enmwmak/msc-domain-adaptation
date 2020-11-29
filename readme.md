This repository contains the programs for the paper 
W.W. Lin, M.W. Mak, N. Li, D. Su, and D. Yu, "A Framework for Adapting DNN Speaker
Embedding Across Languages, IEEE/ACM Transactions on Audio, Speech and Language Processing, 2020.

* "run.sh" is the main script. It converts waveforms to Kaldi scp/ark files, trains an x-vector or a DenseNet, and performs scoring.
* "scripts/" folder contains the shell and Perl scripts used by run.sh
* "mmd-emb/run.py" is the starting point for training the x-vector network or the Densent described in the paper. 
* "mmd-emb/xvectornet.py" defines the x-vector network.
* "mmd-emb/densenet.py" defines the DenseNet.
* "mmd-emb/trainer.py" contains codes and classes for training the x-vector network or the densenet.
* "mmd-emb/scorer.py" contains codes and classes for scoring the x-vectors and densenet-based embedding vectors.
* "mmd-emb/datasets.py" contains codes and classes for loading data.
* "mmd-emb/scp2h5.py" converts Kaldi .scp files to .h5 files
* "mmd-emb/combine_h5.py" combines all .h5 files in a given folder to a single .h5 file using virtual datasets.
* "models/readme" contains a URL from which an example Pytorch model file (not fully trained) can be found.
* "h5/readme" contains a URL from which some example .h5 files for training the networks can be found.

You may need to generate MFCC data in .h5 format. The .h5 file should have 4 fields:

* **mfcc**: A 2-D array with dimension T x MFCC_dim, where T is the total number of frames
* **positions**: A 2-D array with dimension N x 2 containing the indexes to the position of each utterance, where N is the total number of utterances. For example, positions[0, 0] is the index of the starting position of the first utterance, and positions[0, 1] is the index of the ending position of the first utterance. 
* **spk_ids**: Strings containing the speaker IDs
* **utt_ids**: Strings containing the utterance IDs

You may also need to create the following symbolic links:
* ln -s /usr/local/kaldi/egs/sre08/v1/sid .
* ln -s /usr/local/kaldi/egs/wsj/s5/steps .
* ln -s /usr/local/kaldi/egs/wsj/s5/utils .


The programs were developed by W.W. Lin. Some parts of the coded were modified by M.W. Mak.
