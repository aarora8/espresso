#!/usr/bin/env bash

# This script creates traing and validations splits, downloads text corpus for language modeling,
#  prepares the training, validation and test data for rimes dataset 
# (i.e text, images.scp, utt2spk and spk2utt). It calls process_data.py.
#  Eg. local/prepare_data.sh

stage=2
download_dir=data/local
data_url="https://dl.fbaipublicfiles.com/fairseq/data/tutorial_names.tar.gz"
. ./cmd.sh
. ./path.sh
#. ./utils/parse_options.sh || exit 1;

if [ ${stage} -le 1 ]; then
  wget -P $download_dir $data_url
  tar -xzf $download_dir/tutorial_names.tar.gz

  fairseq-preprocess \
    --trainpref names/train --validpref names/valid --testpref names/test \
    --source-lang input --target-lang label \
    --destdir names-bin --dataset-impl raw
fi

if [ ${stage} -le 2 ]; then
  mkdir -p log
  log_file=log/train.log
  CUDA_VISIBLE_DEVICES=$free_gpu fairseq-train names-bin \
    --task simple_classification \
    --arch pytorch_tutorial_rnn \
    --optimizer adam --lr 0.001 --lr-shrink 0.5 \
    --max-tokens 1000 | tee $log_file
fi
