#!/usr/bin/env python3

import os
import numpy as np
from collections import namedtuple


utt_sau_dict = dict()
phones_count_sau = dict()
text_file = os.path.join("/Users/ashisharora/data_prep_siamese/", 'train.sau')
text_fh = open(text_file, 'r', encoding='utf-8')
for line in text_fh:
    uttid = line.split(" ")[0]
    sauasge_transcription = list()
    line_wo_uttid = " ".join(line.strip().split(" ")[1:])
    line_wo_uttid = line_wo_uttid.strip().split("[ 0 ")
    for element in line_wo_uttid:
        count = element.count("[")
        if count != 0:
            start_index = element.index("[")
            element = element[start_index:]
            sauasge_transcription.append(element)
    utt_sau_dict[uttid] = sauasge_transcription
    phones_count_sau[uttid] = len(sauasge_transcription)


reco_sau_dict = dict()
for uttid in sorted(utt_sau_dict.keys()):
    recoid = "_".join(uttid.strip().split("_")[3:])
    if recoid not in list(reco_sau_dict.keys()):
        reco_sau_dict[recoid] = list()
    for sausage_element in utt_sau_dict[uttid]:
        reco_sau_dict[recoid].append(sausage_element)


start_end_set = namedtuple('start_end_set', 'uttid start end')
utt_time_dict = dict()
utt_time_elements_count_dict = dict()
text_file = os.path.join("/Users/ashisharora/data_prep_siamese/", 'train.time')
text_fh = open(text_file, 'r', encoding='utf-8')
for line in text_fh:
    uttid = line.split(" ")[0]
    count = line.count(";")
    line_wo_uttid = " ".join(line.strip().split(" ")[1:])
    start_end_list = line_wo_uttid.strip().split(";")
    utt_time_list = list()
    for element in start_end_list:
        element = element.strip()
        utt_time_list.append(start_end_set(
        uttid = uttid,
        start = element.split(" ")[0],
        end = element.split(" ")[1]
    ))
    utt_time_dict[uttid] = utt_time_list
    utt_time_elements_count_dict[uttid] = len(utt_time_list)

reco_start_end_set = namedtuple('reco_start_end_set', 'uttid start end')
reco_time_dict = dict()
for uttid in sorted(utt_time_dict.keys()):
    recoid = "_".join(uttid.strip().split("_")[3:])
    utt_start_frame = float(uttid.strip().split("_")[1])/3
    if recoid not in list(reco_time_dict.keys()):
        reco_time_dict[recoid] = list()
    for start_end_set in utt_time_dict[uttid]:
        start = float(start_end_set.start) + utt_start_frame
        end = float(start_end_set.end) + utt_start_frame
        reco_time_dict[recoid].append(reco_start_end_set(
            uttid = uttid,
            start = start,
            end = end
        ))

Segment = namedtuple('Segment', 'recoid word start end')
utt_stend_time_ctm_dict = dict()
text_file = os.path.join("/Users/ashisharora/data_prep_siamese/align_train_sp", 'ctm')
text_fh = open(text_file, 'r', encoding='utf-8')
for line in text_fh:
    segment_fields = line.strip().split()
    if "sp0.9" in segment_fields[0]:
        continue
    if "sp1.1" in segment_fields[0]:
        continue
    start = float(segment_fields[2])
    start_frame = (start * 100) / 3
    duration = float(segment_fields[3])
    end = start + duration
    end_frame = (end * 100) / 3
    if segment_fields[0] not in list(utt_stend_time_ctm_dict.keys()):
        utt_stend_time_ctm_dict[segment_fields[0]] = list()
    utt_stend_time_ctm_dict[segment_fields[0]].append(Segment(
        recoid = segment_fields[0],
        word = segment_fields[4],
        start = start_frame,
        end = end_frame
    ))


train_text_handle = open("/Users/ashisharora/data_prep_siamese/train.input-label.input.3", 'w', encoding='utf8')
train_text_handle2 = open("/Users/ashisharora/data_prep_siamese/train.input-label.label.3", 'w', encoding='utf8')
uttword_sauseg_dict = dict()
for recoid in sorted(utt_stend_time_ctm_dict.keys()):
    for Segment in utt_stend_time_ctm_dict[recoid]:
        word = Segment.word
        start_frame = Segment.start
        end_frame = Segment.end
        if word == "<UNK>":
            continue
        sauasge_list = reco_sau_dict[recoid]
        time_list = reco_time_dict[recoid]
        assert len(sauasge_list) == len(time_list)
        count_start = 0
        for time in time_list:
            start_time = time.start
            # print(start_time)
            if start_time >= start_frame:
                # print(start_time)
                # print(start_frame)
                break
            count_start += 1
        count_end = 0
        for time in time_list:
            count_end += 1
            end_time = time.start
            if end_time >= end_frame:
                # print(end_time)
                # print(end_frame)
                break
        # print(word, count_start, count_end)
        # print(recoid, start_frame, end_frame)
        # print(time_list[count_start-1])
        # print(time_list[count_end])
        if count_start != 0:
            count_start -= 1
        train_text_handle.write(' '.join(sauasge_list[count_start:count_end]) + '\n')
        train_text_handle2.write(word + '\n')