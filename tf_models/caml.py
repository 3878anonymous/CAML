#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import gzip
import json
from tqdm import tqdm
import random
from collections import Counter
import operator
import timeit
import time

import datetime
from keras.preprocessing import sequence

from .utilities import *
from keras.utils import np_utils
import numpy as np

from tylib.lib.att_op import *
from tylib.lib.seq_op import *

def multi_pointer_coattention_networks(self,
                                q1_output, q2_output,
                                q1_len, q2_len,
                                o1_embed, o2_embed,
                                o1_len, o2_len,
                                rnn_type='',
                                reuse=None):
    """ Multi-Pointer Co-Attention Networks

    This function excepts a base model object, along with q1_output (user),
    and q2_output (item) and all their meta info (lengths etc.)

    o1_embed and o2_embed are original word embeddings, which have
    not been procesed by review-level encoders.

    Returns q1_output, q2_output, which are the final user/item reprs.
    """
    _odim = o1_embed.get_shape().as_list()[2]

    # for visualisation purposes only
    self.afm = []
    self.afm2 = []
    self.word_att1 = []
    self.word_att2 = []
    self.att1 = []
    self.att2 = []
    self.hatt1 = []
    self.hatt2 = []
    self.word_u = []
    self.word_i = []

    print("========================================")
    print("Multi-Pointer Co-Attention Network Model")
    o1_embed = tf.reshape(o1_embed, [-1, self.args.dmax, _odim])
    o2_embed = tf.reshape(o2_embed, [-1, self.args.dmax, _odim])
    f1, f2 = [],[] 
    r1, r2 = [],[]

    if self.args.masking == 1:
        q1_mask = tf.sequence_mask(q1_len, self.args.dmax, dtype=tf.float32)
        q2_mask = tf.sequence_mask(q2_len, self.args.dmax, dtype=tf.float32)
    else:
        q1_mask = None
        q2_mask = None

    tmp_reuse = reuse
    for i in range(self.args.num_heads):
        sub_afm = []
        """ Review-level Co-Attention
        """

        if self.args.att_reuse == 0:
            name = 'caml_{}'.format(i)
            #reuse = None
        else:
            name = 'caml'
            if i == 1:
            #    reuse = None
            #else:
                tmp_reuse = True

        _q1, _q2, a1, a2, sa1, sa2, afm = co_attention(
                q1_output, q2_output, att_type=self.args.att_type,
                pooling='MAX', mask_diag=False,
                kernel_initializer=self.initializer,
                activation=None, dropout=self.dropout,
                seq_lens=None, transform_layers=self.args.num_inter_proj,
                proj_activation=tf.nn.relu, name=name,
                reuse=tmp_reuse, gumbel=True,
                hard=1, model_type=self.args.rnn_type,
                mask_a=q1_mask, mask_b=q2_mask
                )

        if self.args.average_embed == 1:
            _q1 = tf.reduce_sum(_q1, 1)
            _q2 = tf.reduce_sum(_q2, 1)
            f1.append(_q1)
            f2.append(_q2)

        self.att1.append(sa1)
        self.att2.append(sa2)
        self.hatt1.append(a1)
        self.hatt2.append(a2)
        self.afm.append(afm)

        print("=====================")
        """ Word-level Co-Attention Layer
        """
        print(o1_embed)
        # _dim = o1_embed.get_shape().as_list()[1]
        o1_embed = tf.reshape(o1_embed, [-1, self.args.dmax,
                            self.args.smax * self.args.emb_size])
        o2_embed = tf.reshape(o2_embed, [-1, self.args.dmax,
                            self.args.smax * self.args.emb_size])
        _a1 = tf.expand_dims(a1, 2)
        _a2 = tf.expand_dims(a2, 2)
        #_o1 = tf.reduce_sum(o1_embed * _a1,1)
        #_o2 = tf.reduce_sum(o2_embed * _a2,1)
        #review_word1 = tf.reduce_sum(tf.reshape(self.q1_inputs, [-1, self.args.dmax, self.args.smax]) * tf.cast(_a1, dtype=tf.int32),1)
        #review_word2 = tf.reduce_sum(tf.reshape(self.q2_inputs, [-1, self.args.dmax, self.args.smax]) * tf.cast(_a2, dtype=tf.int32),1)
        review_concept1 = tf.reduce_sum(tf.reshape(self.c1_inputs, [-1, self.args.dmax, self.args.smax]) * tf.cast(_a1, dtype=tf.int32),1)
        review_concept2 = tf.reduce_sum(tf.reshape(self.c2_inputs, [-1, self.args.dmax, self.args.smax]) * tf.cast(_a2, dtype=tf.int32),1)

        _o1 = tf.nn.embedding_lookup(self.embeddings,
                                     review_concept1)
        _o2 = tf.nn.embedding_lookup(self.embeddings,
                                     review_concept2)

        # Reshape back to get original document
        #_o1 = tf.reshape(_o1, [-1, self.args.smax, _odim])
        #_o2 = tf.reshape(_o2, [-1, self.args.smax, _odim])

        print("Lengths:")
        print(q1_len)
        # bsz x dmax
        # olen should be bsz x dmax
        _o1_len = tf.reshape(self.c1_len, [-1, self.args.dmax])
        _o2_len = tf.reshape(self.c2_len, [-1, self.args.dmax])
        _o1_len = tf.reduce_sum(_o1_len * tf.cast(a1, tf.int32),1)
        _o2_len = tf.reduce_sum(_o2_len * tf.cast(a2, tf.int32),1)
        _o1_len = tf.reshape(_o1_len,[-1])
        _o2_len = tf.reshape(_o2_len, [-1])

        if self.args.masking == 1:
            o1_mask = tf.sequence_mask(_o1_len, self.args.smax, dtype=tf.float32)
            o2_mask = tf.sequence_mask(_o2_len, self.args.smax, dtype=tf.float32)
        else:
            o1_mask = None
            o2_mask = None

        if self.args.att_reuse == 0:
            name = 'inner_{}'.format(i)
            #reuse = None
        else:
            name = 'inner'
            #if i == 0:
            #    reuse = None
            #else:
            #    reuse = True


        if self.args.word_gumbel == 1:
            word_hard = True
        else:
            word_hard = False

        z1, z2, wa1, wa2, swa1, swa2, wm = co_attention(
                _o1, _o2, att_type=self.args.att_type,
                pooling=self.args.word_pooling, mask_diag=False,
                kernel_initializer=self.initializer, activation=None,
                dropout=self.dropout, seq_lens=None,
                transform_layers=self.args.num_inter_proj,
                proj_activation=tf.nn.relu, name=name,
                reuse=tmp_reuse, model_type=self.args.rnn_type,
                hard=1, gumbel=word_hard,
                mask_a=o1_mask, mask_b=o2_mask
                )
        sub_afm.append(wm)
        z1 = tf.reduce_sum(z1, 1)
        z2 = tf.reduce_sum(z2, 1)

        if self.args.concept == 1:
            f1.append(z1)
            f2.append(z2)
        # These below are for visualisation only.
        self.afm2.append(wm)
        self.word_att1.append(swa1)
        self.word_att2.append(swa2)
        if self.args.word_gumbel == 1:
            word1 = tf.expand_dims(tf.reduce_sum(review_concept1 * tf.cast(wa1, dtype=tf.int32), 1), 1)
            word2 = tf.expand_dims(tf.reduce_sum(review_concept2 * tf.cast(wa2, dtype=tf.int32), 1), 1)
            self.word_u.append(word1)
            self.word_i.append(word2)


    #if self.args.average_embed == 1:
    #    f1.append(tf.reduce_sum(q1_output, 1))
    #    f2.append(tf.reduce_sum(q2_output, 1))
    if self.args.word_gumbel == 1:
        self.word_u = tf.concat(self.word_u, 1)
        self.word_i = tf.concat(self.word_i, 1)


    if('FN' in rnn_type):
        # Neural Network Multi-Pointer Learning
        q1_output = tf.concat(f1, 1)
        q2_output = tf.concat(f2, 1)
        q1_output = ffn(q1_output, _odim,
                  self.initializer, name='final_proj',
                  reuse=reuse,
                  num_layers=self.args.num_com,
                  dropout=None, activation=tf.nn.relu)
        q2_output = ffn(q2_output, _odim,
                self.initializer, name='final_proj',
                reuse=True,
                num_layers=self.args.num_com,
                dropout=None, activation=tf.nn.relu)
    elif('ADD' in rnn_type):
        # Additive Multi-Pointer Aggregation
        q1_output = tf.add_n(f1)
        q2_output = tf.add_n(f2)
    else:
        # Concat Multi-Pointer Aggregation
        q1_output = tf.concat(f1, 1)
        q2_output = tf.concat(f2, 1)

    print(q1_output)
    print(q2_output)
    print("================================================")
    return q1_output, q2_output
