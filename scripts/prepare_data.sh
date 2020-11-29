#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# This script should be called by ../run.sh
# Modified by M.W. Mak
##########################################################################################

# Path to training corpora
data_root=/corpus  

#-----------------------------------------
# Prepare data in swbd, swbcell and sre04-10+mx6
#-----------------------------------------
# Prepare telephone and microphone speech from Mixer6 (/corpus/mx6_speech)
# This will create data/mix6_* folders
local/make_mx6.sh $data_root data

# Prepare SRE10 test and enroll. Includes microphone interview speech.
# This will create data/sre10
local/make_sre10.pl /corpus/nist10 data

# Prepare SRE08 test and enroll. Includes some microphone speech. Create data/sre08
local/make_sre08.pl /corpus/nist08/test /corpus/nist08/train data

# This prepares the older NIST SREs from 2004-2006. Create data/sre2004, data/sre2005_train,
# data/sre2005_test, data/sre2006_train, and data/sre2006_test
local/make_sre.sh /corpus data

# Combine SRE04-10 and Mixer6 into one dataset. Create data/sre04-10-mx6
utils/combine_data.sh data/sre04-10-mx6 \
  data/sre2004 data/sre2005_train data/sre2005_test data/sre2006_train data/sre2006_test \
  data/sre08 data/sre10 data/mx6
utils/validate_data_dir.sh --no-text --no-feats data/sre04-10-mx6
utils/fix_data_dir.sh data/sre04-10-mx6

# Prepare SWBD corpora. Create data/swbd2_phase?_train, data/swbd_cellular?_train
local/make_swbd_cellular1.pl /corpus/swbcell1 data/swbd_cellular1_train  
local/make_swbd_cellular2.pl /corpus/swbcell2 data/swbd_cellular2_train
local/make_swbd2_phase1.pl /corpus/swb2ph1/ data/swbd2_phase1_train
local/make_swbd2_phase2.pl /corpus/swb2ph2/ data/swbd2_phase2_train
local/make_swbd2_phase3.pl /corpus/swb2ph3/ data/swbd2_phase3_train

# Combine all SWB corpora into one dataset. Create data/swbd
utils/combine_data.sh data/swbd \
   data/swbd_cellular1_train data/swbd_cellular2_train \
   data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

# Combine SRE04-10, Mixer6, and SWB
utils/combine_data.sh data/swbd_sre_combined data/sre04-10-mx6 data/swbd
utils/fix_data_dir.sh data/swbd_sre_combined

#-----------------------------------------
# Prepare data in sre16-dev and sre16-eval
#-----------------------------------------  
# Creat data/sre16_dev_major and data/sre16_dev_minor without speaker grouping
local/make_sre16_dev_unlabeled.pl /corpus/sre16-dev data

# Create data/sre16_dev_enroll and data/sre16_dev_test without speaker grouping in spk2utt
local/make_sre16_dev.pl /corpus/sre16-dev data

# Create data/sre16_eval_enroll and data/sre16_eval_test without speaker grouping in spk2utt
local/make_sre16_eval.pl /corpus/sre16-eval data

#---------------------------------------------------------------------------------------------
# Prepare data in sre18-dev and sre18-eval
#---------------------------------------------------------------------------------------------  
# Create data/sre18_dev_enroll with speaker grouping and data/sre18_dev_test without speaker grouping
local/make_sre18_dev.pl /corpus/sre18-dev data

# Create data/sre18_dev_unlabeled without speaker grouping
local/make_sre18_dev_unlabeled.pl /corpus/sre18-dev data

# Create data/sre18_eval_enroll with speaker grouping and data/sre18_eval_test without speaker grouping
local/make_sre18_eval.pl /corpus/sre18-eval data
  
# Combine all sre18-dev subsets to data/sre18_dev
utils/combine_data.sh data/sre18_dev data/sre18_dev_enroll data/sre18_dev_test data/sre18_dev_unlabeled

# Combine all sre18-eval subsets to data/sre18_dev (not for computing plda as no speaker labels)
utils/combine_data.sh data/sre18_eval data/sre18_eval_enroll data/sre18_eval_test

# Combine Switchboard II Phase1-3 into data/swbd2
utils/combine_data.sh data/swbd2 data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

