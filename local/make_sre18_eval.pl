#!/usr/bin/perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2017   David Snyder
# Apache 2.0
#
# Make file for sre18-eval
# Create data/sre18_eval_enroll with speaker grouping
# Create data/sre18_eval_test without speaker grouping (a requirement for scoring)

if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-SRE18-eval> <path-to-output>\n";
  print STDERR "e.g. $0 /corpus/sre18-eval data/\n";
  exit(1);
}

($db_base, $out_dir) = @ARGV;

#============================
# Handle enrollment utts
#============================
$out_dir_enroll = "$out_dir/sre18_eval_enroll";
if (system("mkdir -p $out_dir_enroll")) {
  die "Error making directory $out_dir_enroll";
}

$tmp_dir_enroll = "$out_dir_enroll/tmp";
if (system("mkdir -p $tmp_dir_enroll") != 0) {
  die "Error making directory $tmp_dir_enroll";
}

open(SPKR, ">$out_dir_enroll/utt2spk") || die "Could not open the output file $out_dir_enroll/utt2spk";
open(WAV, ">$out_dir_enroll/wav.scp") || die "Could not open the output file $out_dir_enroll/wav.scp";
open(META, "<$db_base/docs/sre18_eval_enrollment.tsv") or die "cannot open enrollment list";
%utt2fixedutt = ();
<META>;                             # Skip header
while (<META>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];                  # e.g., 1001_sre18
  $utt = $toks[1];                  # e.g., dlrdnskt_sre18.sph
  if ($utt ne "segment") {
    print SPKR "${spk}-${utt} $spk\n";    # With speaker grouping
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


#============================
# Handle test utts
#============================
$out_dir_test= "$out_dir/sre18_eval_test";
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

if (system("find $db_base/data/test/ -name '*.sph' > $tmp_dir_test/sph.list") != 0) {
  die "Error getting list of sph files";
}
if (system("find $db_base/data/test/ -name '*.flac' > $tmp_dir_test/flac.list") != 0) {
  die "Error getting list of flac files";
}

open(SPHLIST, "<$tmp_dir_test/sph.list") or die "cannot open wav list";
open(FLACLIST, "<$tmp_dir_test/flac.list") or die "cannot open flac list";

while(<SPHLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  $utt = $t[$#t];
  print WAV "$utt"," sph2pipe -f wav -p -c 1 $sph |\n";
  print SPKR "$utt $utt\n";                  # No speaker grouping
}
while(<FLACLIST>) {
  chomp;
  $flac = $_;
  @t = split("/",$flac);
  $utt = $t[$#t];
  print WAV "$utt"," sox $flac -r 8000 -t wav - |\n";
  print SPKR "$utt $utt\n";                 # No speaker grouping
}

close(WAV) || die;
close(SPKR) || die;
close(FLACLIST) || die;
close(SPHLIST) || die;

#======================================================================
# Prepare trial files for CMN2 and VAST based on sre18_eval_trials.tsv
# The last field (target|nontarget) is set to NA as key file is not available
#======================================================================
open(EVALTRIALS, "<$db_base/docs/sre18_eval_trials.tsv") || die "Could not open eval trial file $db_base/docs/sre18_eval_trials.tsv";
<EVALTRIALS>;
while (my $line = <EVALTRIALS>) {
    chomp($line);
    my ($modelid, $testfile, $channel) = split(/\s+/, $line);
    my ($segment, $ext) = split(/\./, $testfile);
    if ($ext eq "sph") {
	print CMN2_TRIALS "${modelid} ${testfile} NA\n";
    } else {
	print VAST_TRIALS "${modelid} ${testfile} NA\n";
    }
    print TRIALS "${modelid} ${testfile} NA\n";
}    
close(EVALTRIALS) || die;
close(TRIALS) || die;
close(CMN2_TRIALS) || die;
close(VAST_TRIALS) || die;

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

# Private functions
sub print_hash {
    my (%hashtbl) = @_;
    foreach my $k (sort keys %hashtbl) {
        print "$k: $hashtbl{$k}\n";
    }
}
