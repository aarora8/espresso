#!/usr/bin/env bash

exp_dir=exp/chain_a/tdnn_a_spec/
lat_dir=exp/chain_a/tdnn_a_spec/decode_train_sp
lang=data/lang_nosp_test
log=data/decode_lats
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


if [ $stage -le -5 ]; then
  steps/nnet3/align_lats.sh --nj 32 --cmd "$train_cmd" \
    --acoustic-scale 1.0 \
    --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' \
    --online-ivector-dir exp/nnet3_a/ivectors_train_sp_hires \
    data/train_sp_hires data/lang_nosp_test exp/chain_a/tdnn_a_spec exp/lats
fi

if [ $stage -le -4 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_nosp_test exp/chain_a/tdnn_a_spec exp/chain_a/tdnn_a_spec/graph
fi

if [ $stage -le -3 ]; then
  steps/nnet3/decode.sh --num-threads 4 --nj 20 --cmd "$decode_cmd" \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --online-ivector-dir exp/nnet3_a/ivectors_train_sp_hires \
     exp/chain_a/tdnn_a_spec/graph data/train_sp_hires exp/chain_a/tdnn_a_spec/decode_train_sp || exit 1;
fi


if [ $stage -le -2 ]; then
  steps/nnet3/align.sh --nj 32 --cmd "$train_cmd" \
      --use-gpu true \
      --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0 --acoustic-scale=1.0' \
      --online-ivector-dir exp/nnet3_a/ivectors_train_sp_hires \
      data/train_sp_hires data/lang_nosp_test exp/chain_a/tdnn_a_spec/ exp/chain_a/tdnn_a_spec/align_train_sp
fi

if [ $stage -le -1 ]; then
#  for i in $(seq 1 $nj);
#  do
#      ali-to-phones --ctm-output exp/chain_a/tdnn_a_spec/final.mdl ark:"gunzip -c exp/chain_a/tdnn_a_spec/align_train_sp/ali.${i}.gz|" exp/chain_a/tdnn_a_spec/align_train_sp/${prefix}_${i}.ctm.int
#      cat exp/chain_a/tdnn_a_spec/align_train_sp/${prefix}_${i}.ctm.int | int2sym.pl -f 5 data/lang_nosp_test/phones.txt > exp/chain_a/tdnn_a_spec/align_train_sp/${prefix}_${i}.ctm.words
#  done
  steps/get_train_ctm.sh data/train_sp_hires/ data/lang_nosp_test/ exp/chain_a/tdnn_a_spec/align_train_sp/ exp/chain_a/tdnn_a_spec/align_train_sp/
fi
exit
if [ $stage -le 0 ] ; then
  for i in $(seq 1 $nj);
  do
      lattice-align-words $lang/phones/word_boundary.int $model \
          "ark:gunzip -c $lat_dir/lat.${i}.gz|" ark:- | \
          lattice-to-phone-lattice $model ark:- ark:- | \
          lattice-align-phones
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
