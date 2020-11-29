#!/bin/bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# This script should be called by ../run.sh
##########################################################################################

#============================================================================================
# If not skipping the data augumentation and DNN training, we produce a DNN in
# exp/xvector_nnet_2a; otherwise, we create a symbolic link to the old DNN in sre16-eval/v2-1
#============================================================================================
# Augment the SRE data with reverberation, noise, music, and babble, and combined it with the clean SRE

echo "Performing data augmentation"  

# Combine Switchboard and sre04-10-mx6 into the folder data/swbd_sre. Note that
# this step should be done after computing the MFCC of sre04-10-mx6 and swbd_sre.  
utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre data/swbd data/sre04-10-mx6
utils/fix_data_dir.sh data/swbd_sre
    
frame_shift=0.01
awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/swbd_sre/utt2num_frames > \
    data/swbd_sre/reco2dur

if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
fi

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

# Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
# additive noise here.
python steps/data/reverberate_data_dir.py \
  "${rvb_opts[@]}" \
  --speech-rvb-probability 1 \
  --pointsource-noise-addition-probability 0 \
  --isotropic-noise-addition-probability 0 \
  --num-replications 1 \
  --source-sampling-rate 8000 \
  data/swbd_sre data/swbd_sre_reverb
cp data/swbd_sre/vad.scp data/swbd_sre_reverb/
utils/copy_data_dir.sh --utt-suffix "-reverb" data/swbd_sre_reverb data/swbd_sre_reverb.new
rm -rf data/swbd_sre_reverb
mv data/swbd_sre_reverb.new data/swbd_sre_reverb

# Prepare the MUSAN corpus, which consists of music, speech, and noise
# suitable for augmentation.
local/make_musan.sh /corpus/musan data

# Get the duration of the MUSAN recordings.  This will be used by the
# script augment_data_dir.py.
for name in speech noise music; do
  utils/data/get_utt2dur.sh data/musan_${name}
  mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
done

# Augment with musan_noise
python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" \
	 --fg-noise-dir "data/musan_noise" data/swbd_sre data/swbd_sre_noise
# Augment with musan_music
python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" \
	 --bg-noise-dir "data/musan_music" data/swbd_sre data/swbd_sre_music
# Augment with musan_speech
python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" \
	   --num-bg-noises "3:4:5:6:7" \
	   --bg-noise-dir "data/musan_speech" data/swbd_sre data/swbd_sre_babble

# Combine reverb, noise, music, and babble into one directory.
utils/combine_data.sh data/swbd_sre_aug \
			data/swbd_sre_reverb data/swbd_sre_noise data/swbd_sre_music data/swbd_sre_babble

# Take a random subset of the augmentations (128k is somewhat larger than twice
# the size of the SWBD+SRE list)
utils/subset_data_dir.sh data/swbd_sre_aug 128000 data/swbd_sre_aug_128k
utils/fix_data_dir.sh data/swbd_sre_aug_128k

# Make filterbanks for the augmented data.  Note that we do not compute a new
# vad.scp file here.  Instead, we use the vad.scp from the clean version of
# the list.
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $num_jobs --cmd "$train_cmd" \
  data/swbd_sre_aug_128k exp/make_mfcc $mfccdir

# Combine the clean and augmented SWBD+SRE list.  This is now roughly
# double the size of the original clean list.
utils/combine_data.sh data/swbd_sre_combined data/swbd_sre_aug_128k data/swbd_sre

# Filter out the clean + augmented portion of the SRE list.  This will be used to
# train the PLDA model later in the script. data/sre_combined does not have SWBD
utils/copy_data_dir.sh data/swbd_sre_combined data/sre_combined
utils/filter_scp.pl data/sre04-10-mx6/spk2utt data/swbd_sre_combined/spk2utt | \
    utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
utils/fix_data_dir.sh data/sre_combined

# Combine data/sre_combined with data/sre16_dev and data/sre16_eval
# This has not been used for training PLDA models in SRE18 submission
#utils/combine_data.sh data/sre_combined_tmp data/sre_combined data/sre16_dev data/sre16_eval
#rm -rf data/sre_combined
#utils/copy_data_dir.sh data/sre_combined_tmp data/sre_combined
#utils/fix_data_dir.sh data/sre_combined
#rm -rf data/sre_combined_tmp
