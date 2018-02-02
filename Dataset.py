import scipy.sparse as sp
import numpy as np
import collections, time

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, sims=None):
        '''
        Constructor
        '''
        self.num_user=0
        self.num_item=0
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        self.user2item=collections.defaultdict(list)
        self.item2user=collections.defaultdict(list)
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")

        self.num_users, self.num_items = self.trainMatrix.shape
        print self.num_users, self.num_items
        self.user2user, self.item2item=None, None
        if sims:
            self.user2user, self.item2item=self.calSims(self.item2user), self.calSims(self.user2item)

    def calSims(self, common):
        print 'calculate sims'
        stats={}#collections.defaultdict(float)
        res=collections.defaultdict(list)
        t1=time.time()
        for c in common:
            for i in common[c]:
                for j in common[c]:
                    if i==j:
                        continue
                    if i not in stats:
                        stats[i]=collections.defaultdict(int)
                    if j not in stats:
                        stats[j]=collections.defaultdict(int)
                    stats[i][j]+=1
                    stats[j][i]+=1
        ranked=[]
        n=0
        for i in stats:
            n+=1
            if n%500==0:
                t1=time.time()
            ranked=sorted(stats[i].items(), key=lambda x:-x[1])[:50]
            for r in ranked:
                res[i].append(r[0])
        return res

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, items = int(arr[0]), [int(x) for x in arr[1:]]
                self.num_user=max(user, self.num_user)
                for item in items:
                    self.num_item=max(item, self.num_item)
                ratingList.append([user, items])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                    self.num_item=max(self.num_item, int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                self.num_user = max(self.num_user, u)
                self.num_item = max(self.num_item, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((self.num_user+1, self.num_item+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                self.user2item[user].append(item)
                self.item2user[item].append(user)
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat
