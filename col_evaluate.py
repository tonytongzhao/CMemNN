import math, random
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from keras.utils import multi_gpu_model
from time import time
#from numba import jit, autojit
import collections
# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def sample_collaboratives(u, i, upass, ipass, user2item, item2user, user2user, item2item, sims=None):
    if sims=='common':
        gpu,gpi=sample_common_collaboratives(u, i, upass, ipass, user2user, item2item)
    else:
        gpu,gpi=sample_random_collaboratives(u, i, upass, ipass, user2item, item2user)
    return gpu, gpi



def sample_random_collaboratives(user, item, upass, ipass, user2item, item2user):

    gpu=list(item2user[item])+[user]*(upass-len(item2user[item])) if upass>len(item2user[item]) else random.sample(item2user[item], upass)

    gpi=list(user2item[user])+[item]*(ipass-len(user2item[user])) if ipass>len(user2item[user]) else random.sample(user2item[user], ipass)

    return gpu, gpi

def sample_common_collaboratives(user, item, upass, ipass, user2user, item2item):

    gpi=list(item2item[item])+[item]*(ipass-len(item2item[item])) if ipass>len(item2item[item]) else item2item[item][:ipass]
    gpu=list(user2user[user])+[user]*(upass-len(user2user[user])) if upass>len(user2user[user]) else user2user[user][:upass]
    return gpu, gpi


def evaluate_model(dataset, model, upass, ipass, user2item, item2user, user2user, item2item, testRatings, testNegatives, K, num_thread, sims):
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _upass
    global _ipass
    global _user2item
    global _item2user
    global _user2user
    global _item2item
    global _dataset
    global _sims
    _dataset= dataset
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _user2item = user2item
    _item2user = item2user
    _user2user = user2user
    _item2item = item2item
    _upass = upass
    _ipass = ipass
    _sims = sims
    num_thread=1

    precision, recall, ndcgs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    test_cands=np.random.permutation(range(len(_testRatings)))[:3000]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, test_cands)
        pool.close()
        pool.join()
        #hits = [r[0] for r in res]
        #ndcgs = [r[1] for r in res]
        #return (hits, ndcgs)
    # Single thread
    idxs=collections.defaultdict(int)
    for idx in test_cands:
        (prec, rec, ndcg) = eval_one_rating(idx)
        for i in xrange(len(prec)):
            precision[i].append(prec[i])
            recall[i].append(rec[i])
            ndcgs[i].append(ndcg[i])
    return (precision, recall, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    all_items=list(set(range(_dataset.num_item))- set(_dataset.user2item[u]))
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(all_items), u, dtype = 'int32')
    gp_user_input=[]
    gp_item_input=[]
    for it in xrange(len(users)):
        u,i=u,all_items[it]
        gpu,gpi=sample_collaboratives(u, i, _upass, _ipass, _user2item, _item2user, _user2user, _item2item, _sims)
        gp_user_input.append(np.array(gpu))
        gp_item_input.append(np.array(gpi))
    # _model_multiple = multi_gpu_model(_model, gpus=8)
    predictions = _model.predict([users, np.array(all_items), np.array(gp_user_input), np.array(gp_item_input)], batch_size=len(users), verbose=0)
    for i in xrange(len(all_items)):
        item = all_items[i]
        map_item_score[item] = predictions[i]
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    precision, recall = getPrecisionRecall(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (precision, recall, ndcg)

def getPrecisionRecall(ranklist, gtItem):
    recall=[]
    precision=[]
    rec=0.0
    for i, item in enumerate(ranklist):
        if item in gtItem:
            rec+=1.0
        recall.append(rec/len(gtItem))
        precision.append(rec/(i+1))
    return precision, recall

def getNDCG(ranklist, gtItem):
    ndcgs=[]
    dcg=0.0
    ndcg=0.0
    for i in xrange(len(ranklist)):
        item = ranklist[i]
        if item in gtItem:
            dcg+=math.log(2) / math.log(i+2)
        ndcg+=math.log(2)/math.log(i+2)
        ndcgs.append(dcg/ndcg)
    return ndcgs






