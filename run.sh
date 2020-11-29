#!/bin/bash -e
# This script converts NIST SRE and Switchboard speech files to Kaldi's MFCC and VAD. 
# Then, it converts the .ark files to .h5 files for DNN training and scoring.
# The scripts implements the DenseNets and x-vector networks in the paper
# W.W. Lin, M.W. Mak, N. Li, D. Su, and D. Yu, "A Framework for Adapting DNN Speaker
# Embedding Across Languages, IEEE/ACM Transactions on Audio, Speech and Language Processing, 2020.
##########################################################################################

# Get run level
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <stage>"
    exit
fi
stage=$1

#======================================================
# Define some constants and variables
#======================================================
multi_machines="N"          # See ./.queue/machines for a list of machines to be used
skip_data_aug="N"           # Skip data augmentation, Y|N

#======================================================
# Set up machines to run jobs in parallel; enmcomp2 and enmcomp12 have 2 CPUs
#======================================================
if [ "$multi_machines" == "Y" ]; then
    mkdir -p ./.queue
    echo -n "" > ./.queue/machines
    for host in enmcomp2 enmcomp12; do
	echo $host >> ./.queue/machines
	num_jobs=16
    done
else
    host=`hostname`
    if [ "$host" == "enmcomp2" ] || [ "$host" == "enmcomp12" ]; then
	num_jobs=16
    else
	num_jobs=8
    fi	
fi

#======================================================
# Set up environment
#======================================================
. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

#======================================================
# Prepare speech data in data/
#======================================================
if [ $stage -eq 0 ]; then
    . ./scripts/prepare_data.sh
fi

#======================================================
# Convert speech to MFCC and store MFCC in mfcc/
#======================================================
if [ $stage -eq 1 ]; then
    . ./scripts/comp_mfcc.sh    
fi

#============================================================================================
# Augment the SRE data with reverberation, noise, music, and babble, and combined it
# with the clean SRE. Produce data/sre_combined/ and augmented MFCC in mfcc/
#============================================================================================
if [ $stage -eq 2 ]; then
    . ./scripts/data_augmentation.sh
fi

#============================================================================================
# Now we prepare the features to generate examples for DNN training. Use vad.scp to remove
# silence. Produce data/swbd_sre_combined_no_sil
#============================================================================================
if [ $stage -eq 3 ]; then
    . ./scripts/prepare_dnn_training.sh
fi

#============================================================================================
# Convert .scp files in data/swbd_sre_combined_no_sil to .h5 files in h5/swbd_sre_combined_no_sil/
# Combine the .h5 files in h5/swbd_sre_combined_no_sil/.
# Store the combined .h5 file (with virtual dataset) as h5/swbd_sre_combined_no_sil.h5
#============================================================================================
if [ $stage -eq 4 ]; then

    # Split and convert data/swbd_sre_combined_no_sil/feats.scp to .h5 files
    rm -f h5/swbd_sre_combined_no_sil/*.h5
    scripts/scp_to_h5.sh data/swbd_sre_combined_no_sil/feats.scp 32
    mkdir -p h5/swbd_sre_combined_no_sil
    mv data/swbd_sre_combined_no_sil/split32/*.h5 h5/swbd_sre_combined_no_sil

    # Create virtual datasets by combining the .h5 files in h5/swbd_sre_combined_no_sil/
    workdir=`pwd`
    cd h5
    python3 ../mmd-emb/combine_h5.py swbd_sre_combined_no_sil swbd_sre_combined_no_sil.h5
    cd $workdir

    # Convert data/sre18_dev_unlabeled/feats.scp to .h5 file
    python3 mmd-emb/scp2h5.py data/sre18_dev_unlabeled/feats.scp h5/sre18_dev_unlabeled.h5 -
    
    # Create augmentation data for data/sre18_dev_unlabeled and stored the info in data/sre18_dev_unlabeled_aug
    scripts/target_domain_aug.sh

    # Convert data/sre18_dev_unlabeled_aug to .h5 file
    python3 mmd-emb/scp2h5.py data/sre18_dev_unlabeled_aug/feats.scp h5/sre18_dev_unlabeled_aug.h5 - 

    # Convert data/sre18_eval_enroll to .h5 file
    python3 mmd-emb/scp2h5.py data/sre18_eval_enroll/feats.scp h5/sre18_eval_enroll.h5 -
    
    # Convert data/sre18_eval_test to .h5 file
    python3 mmd-emb/scp2h5.py data/sre18_eval_test/feats.scp h5/sre18_eval_test.h5 -
    
fi    

#============================================================================================
# Train either x-vector network or DenseNet
# net_type: 'XvectorNet' or 'DenseNet'
# run_mode: 'training'
#============================================================================================
if [ $stage -eq 5 ]; then
    python3 mmd-emb/run.py
fi

#============================================================================================
# Score either x-vector network or DenseNet
# net_type: 'XvectorNet' or 'DenseNet'
# run_mode: 'scoring'
#============================================================================================
if [ $stage -eq 6 ]; then
    python3 mmd-emb/run.py
fi

