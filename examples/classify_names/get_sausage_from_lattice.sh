#!/usr/bin/env bash

exp_dir=exp/chain_all/cnn_tdnn_all/
lat_dir=exp/chain_all/cnn_tdnn_all/decode_safe_t_dev1/
lang=data/lang_nosp_test
log=data/log
[ -f ./path.sh ] && . ./path.sh
stage=0
cmd=run.pl

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh
set -e -o pipefail
set -o nounset
prefix=train
model=$exp_dir/final.mdl
mkdir -p $log
nj=$(ls $lat_dir/lat.*.gz | wc -l)

if [ $stage -le 0 ] ; then
  for i in $(seq 1 $nj);
  do
      lattice-align-words $lang/phones/word_boundary.int $model \
          "ark:gunzip -c $lat_dir/lat.${i}.gz|" ark:- | \
          lattice-to-phone-lattice $model ark:- ark:- | \
          lattice-mbr-decode ark:- ark,t:$log/${prefix}_${i}.text  ark:/dev/null ark,t:$log/${prefix}_${i}.sau ark,t:$log/${prefix}_${i}.time
  done
fi

if [ $stage -le 1 ] ; then
  for i in $(seq 1 $nj);
  do
          cat $log/${prefix}_${i}.text | utils/int2sym.pl -f 2- data/lang_nosp_test/phones.txt > $log/${prefix}_${i}.text.phones
  done
fi

prefix=word
if [ $stage -le 2 ] ; then
  for i in $(seq 1 $nj);
  do
      lattice-mbr-decode ark:"gunzip -c $lat_dir/lat.${i}.gz|" ark,t:$log/${prefix}_${i}.text  ark:/dev/null ark,t:$log/${prefix}_${i}.sau ark,t:$log/${prefix}_${i}.time
      cat $log/${prefix}_${i}.text | utils/int2sym.pl -f 2- data/lang_nosp_test/words.txt > $log/${prefix}_${i}.text.words
  done
fi