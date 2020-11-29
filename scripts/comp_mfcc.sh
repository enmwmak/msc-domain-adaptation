#!/bin/bash -e
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# This script should be called by ../run.sh
##########################################################################################

#======================================================
# Convert speech to MFCC and store MFCC in mfcc/
#======================================================
# Make MFCCs and compute the energy-based VAD for each dataset
# make_mfcc.sh will create data/$name/feats.scp, which indicates the starting
# position of each .sph file in the .ark file.  
# Note that for some dataset (e.g. swbd), the no. of processed files may be
# smaller than the no. of .sph files. In that case, make_mfcc.sh may return with
# an error code, causing bash to terminate. As a result, the compute_vad_decision
# needs to be exectued manually.
list16="sre16_dev_enroll sre16_dev_test sre16_eval_enroll"
list18="sre18_dev_cmn2 sre18_dev_enroll sre18_dev_test sre18_dev_unlabeled sre18_dev_vast sre18_eval_enroll sre18_eval_test"
list_sre_swbd="sre04-10-mx6 swbd"
num_jobs=16
for name in "$list16 $list18 $list_sre_swbd"; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $num_jobs \
			   --cmd "$train_cmd" data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj $num_jobs --cmd "$train_cmd" \
				    data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
done

# Need to reduce the number of jobs for sre18_dev_vast because the no. of utt is too small
num_jobs=4
for name in "sre18_dev_vast"; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $num_jobs \
			   --cmd "$train_cmd" data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj $num_jobs --cmd "$train_cmd" \
				    data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
done


    
    



