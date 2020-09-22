This folder contains the programs for the paper 
W.W. Lin, M.W. Mak, N. Li, D. Su, and D. Yu, "A Framework for Adapting DNN Speaker
Embedding Across Languages, IEEE/ACM Transactions on Audio, Speech and Language Processing, 2020.

"run.py" is the starting point for training the x-vector network or the Densent described in the paper. 

* model.py contains X-vector network. densenets.py contains all densenets.
* trainer.py contains all the neceaary functions and classes.
* .h5 file should have four fields, mfccs, positions, spk_ids, utt_ids. mfccs is a two dimentional TxMFCC_dim array like .
T is the total frames in all data. positions is a two dimentional NX2 array indexing the position of each utterance, where N is the total numbers of utterance.
For example, positions[0, 0] is the index of the starting position of first utterance. positions[0, 1] is the index of the ending position of first utterance.

You may need to generate MFCC data in Panda format and save the data in .h5 file. 
The .h5 file should have 4 fields:
   
