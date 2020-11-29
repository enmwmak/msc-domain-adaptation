#!/bin/bash -e
#============================================================================================
# Augment the unlabeled target-domain data in SRE18 (sre18_dev_unlabeled) with
# with reverberation, noise, music, and babble, and combined it with the clean SRE
#============================================================================================
echo "Performing  unlabeled target-domain data augmentation"  

frame_shift=0.01
awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/sre18_dev_unlabeled/utt2num_frames > \
    data/sre18_dev_unlabeled/reco2dur

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
  data/sre18_dev_unlabeled data/sre18_dev_unlabeled_reverb
cp data/sre18_dev_unlabeled/vad.scp data/sre18_dev_unlabeled_reverb/
utils/copy_data_dir.sh --utt-suffix "-reverb" data/sre18_dev_unlabeled_reverb data/sre18_dev_unlabeled_reverb.new
rm -rf data/sre18_dev_unlabeled_reverb
mv data/sre18_dev_unlabeled_reverb.new data/sre18_dev_unlabeled_reverb

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
	 --fg-noise-dir "data/musan_noise" data/sre18_dev_unlabeled data/sre18_dev_unlabeled_noise
# Augment with musan_music
python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" \
	 --bg-noise-dir "data/musan_music" data/sre18_dev_unlabeled data/sre18_dev_unlabeled_music
# Augment with musan_speech
python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" \
	   --num-bg-noises "3:4:5:6:7" \
	   --bg-noise-dir "data/musan_speech" data/sre18_dev_unlabeled data/sre18_dev_unlabeled_babble

# Combine reverb, noise, music, and babble into one directory.
utils/combine_data.sh data/sre18_dev_unlabeled_aug data/sre18_dev_unlabeled_reverb \
		      data/sre18_dev_unlabeled_noise data/sre18_dev_unlabeled_music \
		      data/sre18_dev_unlabeled_babble

# Make filterbanks for the augmented data.  Note that we do not compute a new
# vad.scp file here.  Instead, we use the vad.scp from the clean version of
# the list.
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 16 --cmd "$train_cmd" \
  data/sre18_dev_unlabeled_aug exp/make_mfcc $mfccdir


