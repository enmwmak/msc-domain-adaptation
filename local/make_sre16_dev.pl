#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
# Apache 2.0
#
# For each utt, we find its call_id from the file sre16_dev_segment_key.tsv.
# From the call_id, we find the speaker_id from the subject_id field of
# the file call_sides.tsv

# Create data/sre16_dev_enroll with speaker grouping
# Create data/sre16_dev_test without speaker grouping (a requirment for scoring)
# Example:
#   local/make_sre16_dev.pl /corpus/sre16-dev data

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-SRE16-dev> <path-to-output>\n";
  print STDERR "e.g. $0 /corpus/callmynet data/\n";
  exit(1);
}

($db_base, $out_dir) = @ARGV;

#--------------------------------
# Handle enroll
#--------------------------------
$out_dir_enroll = "$out_dir/sre16_dev_enroll";
if (system("mkdir -p $out_dir_enroll")) {
  die "Error making directory $out_dir_enroll";
}

$tmp_dir_enroll = "$out_dir_enroll/tmp";
if (system("mkdir -p $tmp_dir_enroll") != 0) {
  die "Error making directory $tmp_dir_enroll";
}

open(SPKR, ">$out_dir_enroll/utt2spk") || die "Could not open the output file $out_dir_enroll/utt2spk";
open(WAV, ">$out_dir_enroll/wav.scp") || die "Could not open the output file $out_dir_enroll/wav.scp";
open(META, "<$db_base/docs/sre16_dev_enrollments.tsv") or die "cannot open sre16_dev_enrollments.tsv";

%utt2fixedutt = ();
while (<META>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];
  $utt = $toks[1];
  if ($utt ne "segment") {
    print SPKR "${spk}-${utt} $spk\n";        # With speaker grouping
    $utt2fixedutt{$utt} = "${spk}-${utt}";
  }
}

if (system("find $db_base/data/enrollment/ -name '*.sph' > $tmp_dir_enroll/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir_enroll/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$utt2fixedutt{$t1[0]};
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
}
close(WAV) || die;
close(SPKR) || die;

#--------------------------------
# Handle test
#--------------------------------
$out_dir_test= "$out_dir/sre16_dev_test";
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
open(CEB_TRIALS, ">$out_dir_test/trials_ceb") || die "Could not open the output file $out_dir_test/trials_ceb";
open(CMN_TRIALS, ">$out_dir_test/trials_cmn") || die "Could not open the output file $out_dir_test/trials_cmn";

if (system("find $db_base/data/test/ -name '*.sph' > $tmp_dir_test/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(KEY, "<$db_base/docs/sre16_dev_trial_key.tsv") || die "Could not open trials file $db_base/docs/sre16_dev_trial_key.tsv.  It might be located somewhere else in your distribution.";
open(SEG_KEY, "<$db_base/docs/sre16_dev_segment_key.tsv") || die "Could not open trials file $db_base/docs/sre16_dev_segment_key.tsv.  It might be located somewhere else in your distribution.";
open(LANG_KEY, "<$db_base/metadata/calls.tsv") || die "Could not open trials file $db_base/metadata/calls.tsv.  It might be located somewhere else in your distribution.";
open(WAVLIST, "<$tmp_dir_test/sph.list") or die "cannot open wav list";
open(CALLSIDE, "<$db_base/metadata/call_sides.tsv") or die "cannot open call_sides.tsv";

%utt2call = ();
while(<SEG_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $utt = $toks[0];
  $call = $toks[1];
  if ($utt ne "segment") {
    $utt2call{$utt} = $call;
  }
}
close(SEG_KEY) || die;

%call2spk = ();
$line = <CALLSIDE>;    # Skip header
while(<CALLSIDE>) {
    chomp;
    $line = $_;
    @toks = split(' ', $line);
    $call = $toks[0];
    $spk = $toks[2];
    $call2spk{$call} = $spk;
}
close(CALLSIDE) || die;

%call2lang = ();
while(<LANG_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $call = $toks[0];
  $lang = $toks[1];
  $call2lang{$call} = $lang;
}
close(LANG_KEY) || die;

# utt_id in wav.scp and utt2spk files should be preceeded by spk_id
while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt = $t1[0];
  $call = $utt2call{$utt};
  $spk = $call2spk{$call};
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  print SPKR "$utt $utt\n";          # No speaker grouping
}
close(WAV) || die;
close(SPKR) || die;

#======================================================================
# Prepare trial files based on sre16_dev_trials.tsv
#======================================================================
while (<KEY>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];     # e.g., 1001_sre16
  $utt = $toks[1];     # e.g., dtadhlw_sre16
  $call = $utt2call{$utt};
  $target_type = $toks[3];   # target or nontarget
  if ($utt ne "segment") {
    print TRIALS "${spk} ${utt} ${target_type}\n";
    if ($call2lang{$call} eq "ceb") {
      print CEB_TRIALS "${spk} ${utt} ${target_type}\n";
    } elsif ($call2lang{$call} eq "cmn") {
      print CMN_TRIALS "${spk} ${utt} ${target_type}\n";
    } else {
      die "Unexpected language $call2lang{$call} for utterance $utt.";
    }
  }
}
close(KEY) || die;
close(TRIALS) || die;
close(CEB_TRIALS) || die;
close(CMN_TRIALS) || die;

#---------------------------------
# Create spk2utt and check errors
#---------------------------------
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

exit;

sub print_hash {
    my (%hashtbl) = @_;
    foreach my $k (sort keys %hashtbl) {
	print "$k: $hashtbl{$k}\n";
    }
}
