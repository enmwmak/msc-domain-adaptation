#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
# Apache 2.0
#
# Create data/sre18_dev_enroll with speaker grouping in spk2utt file
# Create data/sre18_dev_test without speaker grouping in spk2utt file

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-SRE18-dev> <path-to-output>\n";
  print STDERR "e.g. $0 /corpus/sre18-dev data/\n";
  exit(1);
}

($db_base, $out_dir) = @ARGV;

#---------------------
# Handle enroll
#---------------------
$out_dir_enroll = "$out_dir/sre18_dev_enroll";
if (system("mkdir -p $out_dir_enroll")) {
  die "Error making directory $out_dir_enroll";
}

$tmp_dir_enroll = "$out_dir_enroll/tmp";
if (system("mkdir -p $tmp_dir_enroll") != 0) {
  die "Error making directory $tmp_dir_enroll";
}

open(SPKR, ">$out_dir_enroll/utt2spk") || die "Could not open the output file $out_dir_enroll/utt2spk";
open(WAV, ">$out_dir_enroll/wav.scp") || die "Could not open the output file $out_dir_enroll/wav.scp";
open(META, "<$db_base/docs/sre18_dev_enrollment.tsv") or die "cannot open enrollment list";
%utt2fixedutt = ();
<META>;                             # Skip header
while (<META>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];                  # e.g., 1001_sre18
  $utt = $toks[1];                  # e.g., dlrdnskt_sre18.sph
  if ($utt ne "segment") {
    print SPKR "${spk}-${utt} $spk\n";
    $utt2fixedutt{$utt} = "${spk}-${utt}";
  }
}
#&print_hash(%utt2fixedutt); exit;

if (system("find $db_base/data/enrollment/ -name '*.sph' > $tmp_dir_enroll/sph.list") != 0) {
  die "Error getting list of sph files";
}
if (system("find $db_base/data/enrollment/ -name '*.flac' > $tmp_dir_enroll/flac.list") != 0) {
  die "Error getting list of flac files";
}

open(SPHLIST, "<$tmp_dir_enroll/sph.list") or die "cannot open wav list";
open(FLACLIST, "<$tmp_dir_enroll/flac.list") or die "cannot open flac list";

while(<SPHLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  $utt=$utt2fixedutt{$t[$#t]};
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
}
while(<FLACLIST>) {
  chomp;
  $flac = $_;
  @t = split("/",$flac);
  $utt=$utt2fixedutt{$t[$#t]};
  print WAV "$utt"," sox $flac -r 8000 -t wav - |\n";
}
close(WAV) || die;
close(SPKR) || die;
close(FLACLIST) || die;
close(SPHLIST) || die;

#-------------------
# Handle test
#-------------------
$out_dir_test= "$out_dir/sre18_dev_test";
if (system("mkdir -p $out_dir_test")) {
  die "Error making directory $out_dir_test";
}

$tmp_dir_test = "$out_dir_test/tmp";
if (system("mkdir -p $tmp_dir_test") != 0) {
  die "Error making directory $tmp_dir_test";
}

open(SPKR, ">$out_dir_test/utt2spk") || die "Could not open the output file $out_dir_test/utt2spk";
open(WAV, ">$out_dir_test/wav.scp") || die "Could not open the output file $out_dir_test/wav.scp";
open(TRIALS, ">$out_dir_test/trials") || die "Could not open the output file $out_dir_test/trials";
open(CMN2_TRIALS, ">$out_dir_test/trials_cmn2") || die "Could not open the output file $out_dir_test/trials_cmn2";
open(VAST_TRIALS, ">$out_dir_test/trials_vast") || die "Could not open the output file $out_dir_test/trials_vast";
open(PSTN_TRIALS, ">$out_dir_test/trials_pstn") || die "Could not open the output file $out_dir_test/trials_pstn";
open(VOIP_TRIALS, ">$out_dir_test/trials_voip") || die "Could not open the output file $out_dir_test/trials_voip";
open(AFV_TRIALS, ">$out_dir_test/trials_afv") || die "Could not open the output file $out_dir_test/trials_avf";


if (system("find $db_base/data/test/ -name '*.sph' > $tmp_dir_test/sph.list") != 0) {
  die "Error getting list of sph files";
}
if (system("find $db_base/data/test/ -name '*.flac' > $tmp_dir_test/flac.list") != 0) {
  die "Error getting list of flac files";
}

open(KEY, "<$db_base/docs/sre18_dev_trial_key.tsv") || die "Could not open trials file $db_base/docs/sre18_dev_trial_key.tsv.  It might be located somewhere else in your distribution.";
open(SEG_KEY, "<$db_base/docs/sre18_dev_segment_key.tsv") || die "Could not open trials file $db_base/docs/sre18_dev_segment_key.tsv.  It might be located somewhere else in your distribution.";

open(SPHLIST, "<$tmp_dir_test/sph.list") or die "cannot open wav list";
open(FLACLIST, "<$tmp_dir_test/flac.list") or die "cannot open flac list";

%utt2subject = ();      # This hash table is not useful at this moment. But will be useful in future
%utt2source = ();       # This table is used for mapping utt to data source (cmn2 or vast)
<SEG_KEY>;                          # Skip header
while(<SEG_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $utt = $toks[0];
  $subject = $toks[1];
  $source = $toks[6];
  if ($utt ne "segment") {
      $utt2subject{$utt} = $subject;
      $utt2source{$utt} = $source;
  }
}
close(SEG_KEY) || die;

while(<SPHLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  $utt = $t[$#t];
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  print SPKR "$utt $utt\n";
}
while(<FLACLIST>) {
  chomp;
  $flac = $_;
  @t = split("/",$flac);
  $utt = $t[$#t];
  print WAV "$utt"," sox $flac -r 8000 -t wav - |\n";
  print SPKR "$utt $utt\n";
}

close(WAV) || die;
close(SPKR) || die;
close(FLACLIST) || die;
close(SPHLIST) || die;

<KEY>;                        # Skip header
while (<KEY>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];            # 1001_sre18
  $utt = $toks[1];            # aadxhatk_sre18.sph
  $target_type = $toks[3];
  $sub_source = $toks[7];     # pstn|voip|afv
  if ($utt ne "segment") {
    print TRIALS "${spk} ${utt} ${target_type}\n";
    if ($utt2source{$utt} eq "cmn2") {
      print CMN2_TRIALS "${spk} ${utt} ${target_type}\n";
    } elsif ($utt2source{$utt} eq "vast") {
      print VAST_TRIALS "${spk} ${utt} ${target_type}\n";
    } else {
      die "Unexpected data source $utt2source{$utt} for utterance $utt.";
    }
    if ($sub_source eq "pstn") {
      print PSTN_TRIALS "${spk} ${utt} ${target_type}\n";
    } elsif ($sub_source eq "voip") {
	print VOIP_TRIALS "${spk} ${utt} ${target_type}\n";
    } elsif ($sub_source eq "afv") {
	print AFV_TRIALS "${spk} ${utt} ${target_type}\n";
    } else {
	die "Unexpected data sub-source $sub_source for utterance $utt.";
    }
  }
}
close(TRIALS) || die;
close(CMN2_TRIALS) || die;
close(VAST_TRIALS) || die;
close(PSTN_TRIALS) || die;
close(VOIP_TRIALS) || die;
close(AFV_TRIALS) || die;

if (system("utils/utt2spk_to_spk2utt.pl $out_dir_enroll/utt2spk >$out_dir_enroll/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_enroll";
}
if (system("utils/utt2spk_to_spk2utt.pl $out_dir_test/utt2spk >$out_dir_test/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir_test";
}
if (system("utils/fix_data_dir.sh $out_dir_enroll") != 0) {
  die "Error fixing data dir $out_dir_enroll";
}
if (system("utils/fix_data_dir.sh $out_dir_test") != 0) {
  die "Error fixing data dir $out_dir_test";
}


sub print_hash {
    my (%hashtbl) = @_;
    foreach my $k (sort keys %hashtbl) {
        print "$k: $hashtbl{$k}\n";
    }
}
