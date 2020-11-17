This folder contains the programs for the paper 
W.W. Lin, M.W. Mak, N. Li, D. Su, and D. Yu, "A Framework for Adapting DNN Speaker
Embedding Across Languages, IEEE/ACM Transactions on Audio, Speech and Language Processing, 2020.

* "run.py" is the starting point for training the x-vector network or the Densent described in the paper. 
* "xvectornet.py" defines the x-vector network.
* "densenet.py" defines the DenseNet.
* "trainer.py" contains codes and classes for training the x-vector network or the densenet.
* "scorer.py" contains codes and classes for scoring the x-vectors and densenet-based embedding vectors.
* "datasets.py" contains codes and classes for loading data.
* "models/readme" contains a URL from which an example Pytorch model file (not fully trained) can be found.
* "h5/readme" contains a URL from which some example .h5 files for training the networks can be found.

You may need to generate MFCC data in .h5 format. The .h5 file should have 4 fields:

* **mfcc**: A 2-D array with dimension T x MFCC_dim, where T is the total number of frames
* **positions**: A 2-D array with dimension N x 2 containing the indexes to the position of each utterance, where N is the total number of utterances. For example, positions[0, 0] is the index of the starting position of the first utterance, and positions[0, 1] is the index of the ending position of the first utterance. 
* **spk_ids**: Strings containing the speaker IDs
* **utt_ids**: Strings containing the utterance IDs
   
The programs were developed by W.W. Lin. Some parts of the coded were modified by M.W. Mak
