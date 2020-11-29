#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# This script should be called by ../run.sh
##########################################################################################

#============================================================================================
# Now we prepare the features to generate examples for DNN training.
#============================================================================================
# This script applies CMVN and removes nonspeech frames from SWBW+SRE+MX6+Augmented.
# Output the VAD- and CMVN-mfcc to exp/swbd_sre_combined_no_sil, which are indexed
# by data/swbd_sre_combined_no_sil/feats.scp
local/nnet3/xvector/prepare_feats_for_egs.sh --nj $num_jobs --cmd "$train_cmd" \
    data/swbd_sre_combined data/swbd_sre_combined_no_sil exp/swbd_sre_combined_no_sil
utils/fix_data_dir.sh data/swbd_sre_combined_no_sil

# Now, we need to remove features that are too short after removing silence
# frames.  We want atleast 5s (500 frames) per utterance.
min_len=500
mv data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2num_frames.bak
awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_sre_combined_no_sil/utt2num_frames.bak > \
    data/swbd_sre_combined_no_sil/utt2num_frames
utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2spk > \
		      data/swbd_sre_combined_no_sil/utt2spk.new
mv data/swbd_sre_combined_no_sil/utt2spk.new data/swbd_sre_combined_no_sil/utt2spk
utils/fix_data_dir.sh data/swbd_sre_combined_no_sil

# We also want several utterances per speaker. Now we'll throw out speakers
# with fewer than 8 utterances.
min_num_utts=8
awk '{print $1, NF-1}' data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2num
awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' \
	data/swbd_sre_combined_no_sil/spk2num | \
    utils/filter_scp.pl - data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2utt.new
mv data/swbd_sre_combined_no_sil/spk2utt.new data/swbd_sre_combined_no_sil/spk2utt
utils/spk2utt_to_utt2spk.pl data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/utt2spk

utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2spk data/swbd_sre_combined_no_sil/utt2num_frames > \
		      data/swbd_sre_combined_no_sil/utt2num_frames.new
mv data/swbd_sre_combined_no_sil/utt2num_frames.new data/swbd_sre_combined_no_sil/utt2num_frames

# Now we're ready to create training examples.
utils/fix_data_dir.sh data/swbd_sre_combined_no_sil



