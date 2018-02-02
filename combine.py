import sys
import os
import collections

allRating=collections.defaultdict(list)
with open(sys.argv[1], 'r') as f:
    for line in f:
        tks=line.strip().split('\t')
        allRating[tks[0]].append(line.strip())
with open(sys.argv[2], 'r') as f:
    for line in f:
        tks=line.strip().split('\t')
        allRating[tks[0]].append(line.strip())

with open(sys.argv[3], 'w') as f:
    for u in allRating:
        for l in allRating[u]:
            f.write(l+'\n')

