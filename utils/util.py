#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 下午2:19
import time
import codecs
from collections import defaultdict
from datetime import timedelta

def get_time_dif(start_time):
    """Compute time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def read_data(file_path):
    """Read definitions file"""
    content = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            defs=[]
            definition=line[-1].split(" ")
            for d in definition:
                defs.append(d)
            content.append([line[0], defs])
    return content


def read_hypernyms(file_path):
    """Read hypernyms file"""
    hyp_token = []
    hnym_data = defaultdict(list)
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            word = line[0]
            hyp_token.append(word)
            line = line[1:]
            assert len(line) % 2 == 0
            for i in range(int(len(line) / 2)):
                hnym = line[2 * i]
                hyp_token.append(hnym)
                weight = line[2 * i + 1]
                hnym_data[word].append([hnym, weight])
    return hnym_data, hyp_token


if __name__=="__main__":
    a=read_hypernyms("../data/Wn_Gcide/bag_of_hypernyms.txt")
    print()