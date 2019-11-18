# !/usr/bin/env python3
# encoding=utf-8


def load_data(path):
    print('Loading data...')
    # filename = 'cluster_data.txt'
    with open(path, 'r') as fin:
        lines = fin.readlines()
        
    raw_data = []
    for line in lines:
        data = list(map(int, line.strip().split()))
        raw_data.append(data)
    # print(raw_data)
    return raw_data