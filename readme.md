This folder contains the programs for the paper 
W.W. Lin, M.W. Mak, N. Li, D. Su, and D. Yu, "A Framework for Adapting DNN Speaker
Embedding Across Languages, IEEE/ACM Transactions on Audio, Speech and Language Processing, 2020.

* "run.py" is the starting point for training the x-vector network or the Densent described in the paper. 
* "model.py" defines the x-vector network. 
* "densenets.py defines the densenets.
* "trainer.py" contains all the neceaary functions and classes.

You may need to generate MFCC data in Panda format and save the data in .h5 file. 
The .h5 file should have 4 fields:

* mfccs: A 2-D array with dimension T x MFCC_dim, where T is the total number of frames
* positions: A 2-D array with dimension N x 2 containing the indexes to the position of each utterance, where N is the total number of utterances. For example, positions[0, 0] is the index of the starting position of first utterance. positions[0, 1] is the index of the ending position of first utterance. 
* spk_ids: Strings containing the speaker IDs
* utt_ids: Strings containing the utterance IDs
   
