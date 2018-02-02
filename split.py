import collections
import random

ratings=collections.defaultdict(list)
user2item=collections.defaultdict(set)
itemset=set()
import sys
userdict=collections.defaultdict(int)
itemdict=collections.defaultdict(int)
df = sys.argv[1]
data_name=sys.argv[2]#"home"#df.strip().split('.rating')[0]
print data_name
with open(df, 'r') as f:
    for line in f:
        user, item, rating, time=line.strip().split(',')
        if user not in userdict:
            userdict[user]=len(userdict)
        user=str(userdict[user])
        if item not in itemdict:
            itemdict[item]=len(itemdict)
        item=str(itemdict[item])
        ratings[user].append([user, item, rating, time])
        user2item[user].add(item)
        itemset.add(int(item))


max_item=max(itemset)
train=collections.defaultdict(list)
test=collections.defaultdict(list)
negs=collections.defaultdict(list)

num_negs=99
print '#user ', len(userdict)
print '#item ', len(itemdict)
print len(ratings)
deleted_users=[]
for u in ratings:
    if len(ratings[u])<2:
        deleted_users.append(u)

for u in deleted_users:
    del ratings[u]
records=collections.defaultdict(list)
user2item=collections.defaultdict(set)
user2dict=collections.defaultdict(int)
item2dict=collections.defaultdict(int)
itemset=set()
for u in ratings:
    for r in ratings[u]:
        user, item, rating, time=r
        if user not in user2dict:
            user2dict[user]=len(user2dict)
        user=user2dict[user]
        if item not in item2dict:
            item2dict[item]=len(item2dict)
        item=item2dict[item]
        records[user].append([str(user), str(item), rating, time])
        user2item[user].add(item)
        itemset.add(item)

max_item=max(itemset)
'''
with open('./'+data_name+'.82.rating', 'w') as f:
    for u in records:
        for j in records[u]:
            f.write('\t'.join(j)+'\n')
'''
print '#user ', len(user2dict)
print '#item ', len(item2dict)
print len(records)
count=0
for u in records:
    rs=sorted(records[u], key=lambda x:x[-1])
    ters=len(rs)/5
    trrs=len(rs)-ters
    for r in rs[:trrs]:
        train[u].append('\t'.join(r))
    if ters:
        test_r=[]
        for r in rs[trrs:]:
            test_r.append(r[1])
        test[u].append('\t'.join([str(u)]+test_r))
    neg_candidate=set([random.randint(0,max_item) for _ in xrange(num_negs*2)])
    neg_candidate=list(neg_candidate-user2item[u])[:(num_negs)]
    negs[(u,rs[-1][1])].append(neg_candidate)
    if count%100000==0:
        print count
    count+=1
#data_name+="_all"
print 'Split complete'
with open('./'+data_name+'.82.train.rating', 'w') as f:
    for u in sorted(train.keys()):
        for r in train[u]:
            f.write(r+'\n')

with open('./'+data_name+'.82.test.rating', 'w') as f:
    for u in sorted(test.keys()):
        for r in test[u]:
            f.write(r+'\n')

with open('./'+data_name+'.82.test.negative', 'w') as f:
    for u in sorted(negs.keys(), key=lambda x: x[0]):
        for r in negs[u]:
            f.write('\t'.join([str(u)]+map(str, r))+'\n')

