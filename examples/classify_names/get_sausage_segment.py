#!/usr/bin/env python3

import argparse
import io
import os
import sys
import numpy as np


def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices

lexicon_handle = open("/Users/ashisharora/data_prep_siamese/lexicon.txt", 'r', encoding='utf8')
phone_handle = open("/Users/ashisharora/data_prep_siamese/phones.txt", 'r', encoding='utf8')
word_phone_dict = dict()
phone_index_dict = dict()
valid_word_dict = dict()
lexicon_data_vect = lexicon_handle.read().strip().split("\n")
phones_data_vect = phone_handle.read().strip().split("\n")

for key_val in lexicon_data_vect:
  key_val = key_val.split(" ")
  if key_val[0] in word_phone_dict:
      valid_word_dict[key_val[0]] = 'false'
  word_phone_dict[key_val[0]] = key_val[1:]

for key_val in phones_data_vect:
  key_val = key_val.split(" ")
  phone_index_dict[key_val[0]] = key_val[1:]

text_file = os.path.join("/Users/ashisharora/data_prep_siamese/", 'text.words')
text_fh = open(text_file, 'r', encoding='utf-8')
utt_word_dict = dict()
valid_text_dict = dict()
unique_words_train = dict()
for line in text_fh:
    line = line.strip().split(" ")
    utt_id = line[0]
    transcription = line[1:]
    utt_word_dict[utt_id] = transcription
    for word in transcription:
        word = word.strip()
        if word in valid_word_dict:
            valid_text_dict[utt_id] = 'false'
        if word not in unique_words_train:
            unique_words_train[word] = 1
        else:
            unique_words_train[word] += 1


print(len(unique_words_train))


text_file = os.path.join("/Users/ashisharora/data_prep_siamese", 'text.phones')
text_fh = open(text_file, 'r', encoding='utf-8')
utt_phone_dict = dict()
phones_count_text_phones = dict()
for line in text_fh:
    line = line.strip().split(" ")
    utt_id = line[0]
    phone_transcription = line[1:]
    # remove silence
    # utterance = utterance.replace("<sil>", "")
    # remove empty entries in list
    phone_transcription = list(filter(None, phone_transcription))
    count_phones = len(phone_transcription)
    # stores uttid and number of phones
    phone_transcription_wo_word_position = list()
    for phone in phone_transcription:
        phone_transcription_wo_word_position.append(phone.split('_')[0])  # Removing the world-position markers(e.g. _B)
    utt_phone_dict[utt_id] = phone_transcription_wo_word_position
    phones_count_text_phones[utt_id] = count_phones


utt_sau_dict = dict()
phones_count_sau = dict()
text_file = os.path.join("/Users/ashisharora/data_prep_siamese", 'train.sau')
text_fh = open(text_file, 'r', encoding='utf-8')
for line in text_fh:
    utt_id = line.split(" ")[0]
    count = line.count("[")
    count_eps = line.count("[ 0")
    sauasge_transcription = list()
    line_wo_uttid = " ".join(line.strip().split(" ")[1:])
    line_wo_uttid = line_wo_uttid.strip().split("[ 0 ")
    for element in line_wo_uttid:
        count = element.count("[")
        if count != 0:
            start_index = element.index("[")
            element = element[start_index:]
            sauasge_transcription.append(element)
    utt_sau_dict[utt_id] = sauasge_transcription
    phones_count_sau[utt_id] = len(sauasge_transcription)


count = 0
count_total =0
for key_val in phones_count_sau:
    key_val = key_val.split(" ")
    count_total += 1
    if phones_count_sau[key_val[0]] != phones_count_text_phones[key_val[0]]:
        print(key_val[0])
        print(phones_count_sau[key_val[0]], phones_count_text_phones[key_val[0]])
        count += 1


output_text_handle = open("/Users/ashisharora/data_prep_siamese/output", 'w', encoding='utf8')
uttword_sauseg_dict = dict()
unique_words = dict()
for utt_id in sorted(utt_word_dict.keys()):
    transcription = utt_word_dict[utt_id]
    phone_transcription = utt_phone_dict[utt_id]
    sauasge_transcription = utt_sau_dict[utt_id]
    print(transcription)
    print(phone_transcription)
    print(sauasge_transcription)
    print(phones_count_sau[utt_id], phones_count_text_phones[utt_id])
    if utt_id in valid_text_dict:
        continue
    phone_start = 0
    large_string = ' '.join(phone_transcription)
    print(large_string)
    uttword_sauseg_dict[utt_id] = dict()
    for index, word in enumerate(transcription):
        if word == "<UNK>":
            continue
        word2phone = word_phone_dict[word]
        substring_to_find = ' '.join(word2phone)
        result = large_string.find(substring_to_find)
        count = large_string.count(" ", 0, result)
        # print(substring_to_find)
        # print(result)
        # print(count)
        # print(count + len(word2phone) - 1)
        key = str(index) + "_" + word
        uttword_sauseg_dict[utt_id][key] = sauasge_transcription[count:(count + len(word2phone))]
        # print(key, uttword_sauseg_dict[utt_id][key])
        # output_text_handle.write(word + " " + ' '.join(uttword_sauseg_dict[utt_id][key]) + '\n')
        if word not in unique_words:
            unique_words[word] = 1
        else:
            unique_words[word] += 1
        # input()

print(len(unique_words_train))
print(len(unique_words))

# count_words = 0
# for word in unique_words:
#     count_words += unique_words[word]
#     output_text_handle.write(word + " " + str(unique_words[word]) + '\n')
# print(count_words)
#
# count_words = 0
# for word in unique_words_train:
#     count_words += unique_words_train[word]
#     output_text_handle.write(word + " " + str(unique_words_train[word]) + '\n')
# print(count_words)