#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright   2017   David Snyder
# Apache 2.0

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-sre18-dev> <path-to-output>\n";
  print STDERR "e.g. $0 /corpus/sre18-dev data\n";
  exit(1);
}

($db_base, $out_dir) = @ARGV;

# Handle unlabeled data
$out_dir = "$out_dir/sre18_dev_unlabeled";
if (system("mkdir -p $out_dir")) {
  die "Error making directory $out_dir";
}

$tmp_dir = "$out_dir/tmp";
if (system("mkdir -p $tmp_dir") != 0) {
  die "Error making directory $tmp_dir";
}

open(SPKR, ">$out_dir/utt2spk") || die "Could not open the output file $out_dir/utt2spk";
open(WAV, ">$out_dir/wav.scp") || die "Could not open the output file $out_dir/wav.scp";

if (system("find $db_base/data/unlabeled/ -name '*.sph' > $tmp_dir/sph.list") != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  $utt=$t[$#t];
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  print SPKR "$utt $utt\n";
}

close(WAV) || die;
close(SPKR) || die;

if (system("utils/utt2spk_to_spk2utt.pl $out_dir/utt2spk >$out_dir/spk2utt") != 0) {
  die "Error creating spk2utt file in directory $out_dir";
}
if (system("utils/fix_data_dir.sh $out_dir") != 0) {
  die "Error fixing data dir $out_dir";
}

