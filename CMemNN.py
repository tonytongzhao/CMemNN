import numpy as np
import sys

mf_pretrain = ''
mlp_pretrain = ''
upass=int(sys.argv[1])
ipass=int(sys.argv[2])
gpu=int(sys.argv[3])
import os
'''
from keras.utils.training_utils import multi_gpu_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
'''

os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout, RepeatVector,TimeDistributed, BatchNormalization
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from col_evaluate import evaluate_model, sample_collaboratives
from Dataset import Dataset
from time import time
import sys


def get_model(num_users, num_items, upass, ipass, mf_dim=10, layers=1, reg_layers=[0], reg_mf=0, enable_dropout=False):
    num_layer = layers#layers#len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    #Collaborative variables
    group_user_input = Input(shape=(upass,), dtype='int32', name = 'group_user_input')
    group_item_input = Input(shape=(ipass,), dtype='int32', name = 'group_item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01,seed=None), name = 'mf_embedding_user',
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), name = 'mf_embedding_item',
                                  input_length=1)

    GP_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = "gp_embedding_user", embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                  input_length=upass)
    GP_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'gp_embedding_item',embeddings_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                                  input_length=ipass)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = keras.layers.Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

    # CMEMNN part
    group_user_latent = GP_Embedding_User(group_user_input)
    group_item_latent = GP_Embedding_Item(group_item_input)

    gp_user_latent = Flatten()(MF_Embedding_User(user_input))
    gp_item_latent = Flatten()(MF_Embedding_Item(item_input))
    for idx in xrange(num_layer):
        #user_collaborative
        gp_item_repeated=RepeatVector(1)(gp_item_latent)
        attu=keras.layers.Dot(axes=-1)([group_user_latent, gp_item_repeated])
        attu= Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), output_shape=lambda s: (s[0],s[2],s[1]))(attu)
        attu=TimeDistributed(Activation('softmax'))(attu)
        group_user_permuted_latent= Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), output_shape=lambda s: (s[0],s[2],s[1]) )(group_user_latent)
        col_u_vec=keras.layers.Dot(axes=-1)([attu, group_user_permuted_latent])
        col_u_vec=BatchNormalization()(col_u_vec)


        #item_collaborative
        gp_user_repeated=RepeatVector(1)(gp_user_latent)
        atti=keras.layers.Dot(axes=-1)([group_item_latent, gp_user_repeated])
        atti= Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), output_shape=lambda s: (s[0],s[2],s[1]))(atti)
        atti=TimeDistributed(Activation('softmax'))(atti)
        group_item_permuted_latent= Lambda(lambda x: K.permute_dimensions(x, (0,2,1)), output_shape=lambda s: (s[0],s[2],s[1]))(group_item_latent)
        col_i_vec=keras.layers.Dot(axes=-1)([atti, group_item_permuted_latent])
        col_i_vec=BatchNormalization()(col_i_vec)

        gp_user_latent=Flatten()(col_u_vec)
        gp_item_latent=Flatten()(col_i_vec)

    # Concatenate MF and MLP parts
    #gp_vector = keras.layers.Multiply()([gp_user_latent, gp_item_latent]) # element-wise multiply
    #gp_vector = keras.layers.Multiply()([gp_user_latent, mf_item_latent]) # element-wise multiply
    gp_vector = keras.layers.Multiply()([mf_user_latent, gp_item_latent]) # element-wise multiply

    predict_vector = keras.layers.Concatenate()([mf_vector, gp_vector])
	'''
    if enable_dropout:
        predict_vector = Dropout(0.5)(predict_vector)
	'''
    # Final prediction layer
    prediction = Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid', name = "prediction")(predict_vector)

    model = Model(inputs=[user_input, item_input, group_user_input, group_item_input],
                  outputs=[prediction])

    return model

def get_train_instances(train, upass, ipass, user2item, item2user, num_negatives, weight_negatives, user_weights):
    user_input, item_input, gp_user_input, gp_item_input, labels, weights = [],[],[],[],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        gpu,gpi=sample_collaboratives(u, i, upass, ipass, user2item, item2user, None, None)
        gp_user_input.append(gpu)
        gp_item_input.append(gpi)
        labels.append(1)
        weights.append(user_weights[u])
        # negative instances
        for t in xrange(num_negatives):
            j = np.random.randint(num_items)
            while train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            gpu,gpj=sample_collaboratives(u, j, upass, ipass, user2item, item2user, None, None)
            gp_user_input.append(np.array(gpu))
            gp_item_input.append(np.array(gpj))
            labels.append(0)
            weights.append(weight_negatives * user_weights[u])
    return user_input, item_input, gp_user_input, gp_item_input, labels, weights

if __name__ == '__main__':
    dataset_name ="ml-1m.82"
    print dataset_name
    mf_dim =32   #embedding size
    reg_mf = 0
    num_negatives = 6  #number of negatives per positive instance
    weight_negatives = 1.0
    learner = "adam"
    reg_layers=0.001
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 128
    layers=1
    verbose = 1
    enable_dropout = False

    topKs = range(1,11)
    evaluation_threads = 1#mp.cpu_count()
    import logging
    log_name="CMemNN(%s) Dropout %s: mf_dim=%d, layers=%d, regs=%f, reg_mf=%.1e, num_negatives=%d, weight_negatives=%.2f, learning_rate=%.1e, num_epochs=%d, batch_size=%d, verbose=%d, upass=%d, ipass=%d"%(dataset_name, enable_dropout, mf_dim, layers, reg_layers, reg_mf, num_negatives, weight_negatives, learning_rate, num_epochs, batch_size, verbose, upass, ipass)
    logging_mark=0
    logger = logging.getLogger('cmemnn')
    if not logging_mark:
        logger.setLevel(logging.DEBUG)
    else:
        if not os.path.exists('/data/zhaoton/col/results/'+dataset_name):
            os.makedirs('/data/zhaoton/col/results/'+dataset_name)
        hdlr = logging.FileHandler(os.path.join('/data/zhaoton/col/results/'+dataset_name, log_name+'.log'))
        logger.addHandler(hdlr)
        logger.setLevel(logging.DEBUG)


    # Loading data
    t1 = time()
    dataset = Dataset("/data1/col/data/"+dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    user2item=dataset.user2item
    item2user=dataset.item2user
    num_users, num_items = train.shape
    total_weight_per_user = train.nnz / float(num_users)
    train_csr, user_weights = train.tocsr(), []
    for u in xrange(num_users):
        #user_weights.append(total_weight_per_user / float(train_csr.getrow(u).nnz))
        user_weights.append(1)
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, upass, ipass,  mf_dim, layers, reg_layers, reg_mf, enable_dropout)
    #model = multi_gpu_model(model, gpus=8)

    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        print 'sgd'
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    
	# Init performance
    precisions=[0 for _ in xrange(len(topKs))]
    recalls=[0 for _ in xrange(len(topKs))]
    ndcgs=[0 for _ in xrange(len(topKs))]
    (precision, recall, ndcgss) = evaluate_model(dataset, model, upass, ipass, user2item, item2user, None, None, testRatings, testNegatives, topKs[-1], evaluation_threads, None)
    for i in xrange(len(topKs)):
        prec, rec, ndcg = np.array(precision[i]).mean(), np.array(recall[i]).mean(), np.array(ndcgss[i]).mean()
        precisions[i]=prec
        recalls[i]=rec
        ndcgs[i]=ndcg
    # Training model
    for epoch in xrange(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, gp_user_input, gp_item_input, labels, weights = get_train_instances(train, upass, ipass, user2item, item2user, num_negatives, weight_negatives, user_weights)
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input), np.array(gp_user_input), np.array(gp_item_input)], #input
                         np.array(labels), # labels
                         sample_weight=np.array(weights), # weight of samples
                         batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (precision, recall, ndcgss) = evaluate_model(dataset, model, upass, ipass, user2item, item2user, None, None, testRatings, testNegatives, topKs[-1], evaluation_threads, None)
            for i in xrange(len(topKs)):
                prec, rec, ndcg = np.array(precision[i]).mean(), np.array(recall[i]).mean(), np.array(ndcgss[i]).mean()
				precisions[i] = prec
				ndcgs[i] = ndcg
				recalls[i]=rec
            logger.info("CMemNN(%s) Dropout %s: upass=%d, ipass=%d, mf_dim=%d, layers=%d, regs=%f, reg_mf=%.1e, num_negatives=%d, weight_negatives=%.2f, learning_rate=%.1e, num_epochs=%d, batch_size=%d, verbose=%d"%(dataset_name, enable_dropout, upass, ipass, mf_dim, layers, reg_layers, reg_mf, num_negatives, weight_negatives, learning_rate, num_epochs, batch_size, verbose))
            logger.info(precisions)
            logger.info(recalls)
            logger.info(ndcgs)

