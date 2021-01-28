#!/usr/bin/env python3

lexicon_handle = open("/Users/ashisharora/espresso/examples/classify_names/names-bin2/train.input-label.input.2", 'r', encoding='utf8')
unique_char = dict()
lexicon_data_vect = lexicon_handle.read().strip().split("\n")

for word in lexicon_data_vect:
    for char in word:
        if char not in unique_char:
            unique_char[char] = 1
        else:
            unique_char[char] += 1

count_words = 0
for char in unique_char.keys():
    print(char, unique_char[char])
