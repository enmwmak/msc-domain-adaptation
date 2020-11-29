#!/bin/bash -e

# Convert an input.scp file to a number of .h5 files and store the .h5 files in the same folder as the
# .scp file
# Example usage:
#   scripts/scp_to_h5.sh <fullscpfile> <no. of splits>
#   scripts/scp_to_h5.sh data/swbd_sre_combined_no_sil/feats.scp 16

fullscpfile=$1
ns=$2

datadir=`dirname $fullscpfile`
base=`basename $fullscpfile | sed 's/.scp//'`

mkdir -p $datadir/split${ns}
nlines=`cat $fullscpfile | wc -l`
nps=`expr $nlines / $ns`
ns1=`expr $ns - 1`
end=0
for i in `seq $ns1`; do	
    scpfile=$datadir/split${ns}/${base}-${i}.scp
    start=`expr $end + 1`
    end=`expr $start + $nps`
    echo "$start,$end"
    sed -n ${start},${end}p $fullscpfile > $scpfile
    h5file=$datadir/split${ns}/${base}-${i}.h5
    echo "$scpfile --> $h5file"
    python3 ../T-ASLP20/mmd-emb/scp2h5.py $scpfile $h5file _,-
done

scpfile=$datadir/split${ns}/${base}-${ns}.scp
start=`expr $end + 1`
end=$nlines
echo "$start,$end"
sed -n ${start},${end}p $fullscpfile > $scpfile
h5file=$datadir/split${ns}/${base}-${ns}.h5
echo "$scpfile --> $h5file"
python3 ../T-ASLP20/mmd-emb/scp2h5.py $scpfile $h5file _,-

