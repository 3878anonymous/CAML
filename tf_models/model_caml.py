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
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper
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
from tylib.lib.cnn import *
from tylib.lib.compose_op import *
from tylib.exp.exp_ops import *
from tf_models.caml import *

class Model:
    ''' Base model class.

    This model originally supported multiple prediction types and tasks,
    such as MSE-based prediction (regression), classification (softmax)
    and even ranking. I stripped down the more irrelevant details for
    this repository but you may still find some artifacts of previous enabled
    features.

    This model also originally supports char-level representations, POS tag,
    external features and even the CoVe vectors. Since I do not use them, 
    I have removed them.
    '''
    def __init__(self, vocab_size, args, char_vocab=0, pos_vocab=0,
                    mode='RANK', num_user=0, num_item=0):
        self.vocab_size = vocab_size
        self.char_vocab = char_vocab
        self.pos_vocab = pos_vocab
        self.graph = tf.Graph()
        self.args = args
        self.imap = {}
        self.inspect_op = []
        self.mode=mode
        self.write_dict = {}
        self.PAD_tag = 0
        self.SOS_tag = 2
        self.EOS_tag = 3
        self.UNK_tag = 1
        # For interaction data only (disabled and removed from this repo)
        self.num_user = num_user
        self.num_item = num_item
        print('Creating Model in [{}] mode'.format(self.mode))
        self.feat_prop = None
        if(self.args.init_type=='xavier'):
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif(self.args.init_type=='normal'):
            self.initializer = tf.random_normal_initializer(0.0,
                                        self.args.init)
        elif(self.args.init_type=='uniform'):
            self.initializer = tf.random_uniform_initializer(
                                        maxval=self.args.init,
                                        minval=-self.args.init)

        self.cnn_initializer = tf.random_uniform_initializer(
                                        maxval=self.args.init,
                                        minval=-self.args.init)
        self.init = self.initializer
        self.temp = []
        self.att1, self.att2 = [],[]
        self.build_graph()

    def _get_pair_feed_dict(self, data, mode='training', lr=None):
        """ This is for pairwise ranking and not relevant to this repo.
        """
        data = zip(*data)
        labels = data[-1]

        if(lr is None):
            lr = self.args.learn_rate

        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.q1_len:data[self.imap['q1_len']],
            self.q2_len:data[self.imap['q2_len']],
            self.learn_rate:lr,
            self.dropout:self.args.dropout,
            self.rnn_dropout:self.args.rnn_dropout,
            self.emb_dropout:self.args.emb_dropout
        }
        if(mode=='training'):
            feed_dict[self.q3_inputs] = data[self.imap['q3_inputs']]
            feed_dict[self.q3_len]=data[self.imap['q3_len']]
        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rnn_dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        if(self.args.features):
            feed_dict[self.pos_features] = data[6]
            if(mode=='training'):
                feed_dict[self.neg_features] = data[7]
        return feed_dict

    def _check_model_type(self):
        if('SOFT' in self.args.rnn_type):
            return 'point'
        elif('SIG_MSE' in self.args.rnn_type \
                or 'RAW_MSE' in self.args.rnn_type):
            return 'point'
        else:
            return 'pair'

    def get_feed_dict(self, data, mode='training', lr=None):
        #mdl_type = self._check_model_type()
        #if(mdl_type=='point'):
        return self._get_point_feed_dict(data, mode=mode, lr=lr)
        #else:
        #    return self._get_pair_feed_dict(data, mode=mode, lr=lr)

    def _get_point_feed_dict(self, data, mode='training', lr=None):
        """ This is the pointwise feed-dict that is actually used.
        """
        data = list(zip(*data))
        labels = data[-1]
        soft_labels = np.array([[1 if t == i else 0
                            for i in range(self.args.num_class)] \
                            for t in labels])
        sig_labels = labels

        if(lr is None):
            lr = self.args.learn_rate
        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.q1_len:data[self.imap['q1_len']],
            self.q2_len:data[self.imap['q2_len']],
            self.c1_inputs:data[self.imap['c1_inputs']],
            self.c2_inputs:data[self.imap['c2_inputs']],
            self.c1_len:data[self.imap['c1_len']],
            self.c2_len:data[self.imap['c2_len']],
            self.learn_rate:lr,
            self.batch_size:len(data[self.imap['q2_len']]),
            self.dropout:self.args.dropout,
            self.rnn_dropout:self.args.rnn_dropout,
            self.emb_dropout:self.args.emb_dropout,
            self.soft_labels:soft_labels,
            self.sig_labels:sig_labels
        }

        if self.args.implicit == 1:
            feed_dict[self.user_id] = data[self.imap['user_id']]
            feed_dict[self.item_id] = data[self.imap['item_id']]
        #if('TNET' in self.args.rnn_type):
        #    # Use TransNet
        #    feed_dict[self.trans_inputs] = data[self.imap['trans_inputs']]
        #    feed_dict[self.trans_len] = data[self.imap['trans_len']]
        if (mode!='infer'):
            #feed_dict[self.gen_outputs] = self.imap['gen_outputs']
            feed_dict[self.gen_len] = data[self.imap['gen_len']]

            max_len = 0
            for i in range(len(feed_dict[self.gen_len])):
                if max_len < feed_dict[self.gen_len][i]:
                    max_len = feed_dict[self.gen_len][i]

            padding_outputs = []
            for i in range(len(data[self.imap['gen_outputs']])):
                padding_outputs.append(pad_to_max(data[self.imap['gen_outputs']][i], max_len))
            feed_dict[self.gen_outputs] = padding_outputs
            #print (len(feed_dict[self.gen_outputs]), len(feed_dict[self.gen_outputs][0]))
            

        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rnn_dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        #if(self.args.features):
        #    feed_dict[self.pos_features] = data[6]
        return feed_dict

    def register_index_map(self, idx, target):
        self.imap[target] = idx

    def _joint_representation(self, q1_embed, q2_embed, q1_len, q2_len, q1_max,
                    q2_max, force_model=None, score=1,
                    reuse=None, features=None, extract_embed=False,
                    side='', c1_embed=None, c2_embed=None, p1_embed=None,
                    p2_embed=None, i1_embed=None, i2_embed=None, o1_embed=None,
                    o2_embed=None, o1_len=None, o2_len=None, q1_mask=None,
                    q2_mask=None):
        """ Learns a joint representation given q1 and q2.
        """

        print("Learning Repr [{}]".format(side))
        print(q1_embed)
        print(q2_embed)

        # Extra projection layer
        if('HP' in self.args.rnn_type):
            # Review level Highway layer
            use_mode='HIGH'
        else:
            use_mode='FC'

        if(self.args.translate_proj==1):
            q1_embed = projection_layer(
                    q1_embed,
                    self.args.rnn_size,
                    name='trans_proj',
                    activation=tf.nn.relu,
                    initializer=self.initializer,
                    dropout=self.args.dropout,
                    reuse=reuse,
                    use_mode=use_mode,
                    num_layers=self.args.num_proj,
                    return_weights=True,
                    is_train=self.is_train
                    )
            q2_embed = projection_layer(
                    q2_embed,
                    self.args.rnn_size,
                    name='trans_proj',
                    activation=tf.nn.relu,
                    initializer=self.initializer,
                    dropout=self.args.dropout,
                    reuse=True,
                    use_mode=use_mode,
                    num_layers=self.args.num_proj,
                    is_train=self.is_train
                    )
        else:
            self.proj_weights = self.embeddings

        if(self.args.all_dropout):
            q1_embed = tf.nn.dropout(q1_embed, self.dropout)
            q2_embed = tf.nn.dropout(q2_embed, self.dropout)

        representation = None
        att1, att2 = None, None
        if(force_model is not None):
            rnn_type = force_model
        else:
            rnn_type = self.args.rnn_type
        rnn_size = self.args.rnn_size

        if self.args.masking == 1:
            _, q1_output = self.learn_single_repr(q1_embed, q1_len, q1_max,
                                            rnn_type,
                                            reuse=reuse, pool=False,
                                            name='main', mask=q1_mask)
            _, q2_output = self.learn_single_repr(q2_embed, q2_len, q2_max,
                                            rnn_type,
                                            reuse=True, pool=False,
                                            name='main', mask=q2_mask)
        else:
            _, q1_output = self.learn_single_repr(q1_embed, q1_len, q1_max,
                                            rnn_type,
                                            reuse=reuse, pool=False,
                                            name='main', mask=None)
            _, q2_output = self.learn_single_repr(q2_embed, q2_len, q2_max,
                                            rnn_type,
                                            reuse=True, pool=False,
                                            name='main', mask=None)

        print("==============================================")
        print('Single Repr:')
        print(q1_output)
        print(q2_output)
        print("===============================================")

        print (q1_output.get_shape(), q2_output.get_shape())
        # activate MPCN model
        q1_output, q2_output = multi_pointer_coattention_networks(
                                                self,
                                                q1_output, q2_output,
                                                q1_len, q2_len,
                                                o1_embed, o2_embed,
                                                o1_len, o2_len,
                                                rnn_type=self.args.rnn_type,
                                                reuse=reuse)
        
        try:
            # For summary statistics
            self.max_norm = tf.reduce_max(tf.norm(q1_output,
                                        ord='euclidean',
                                        keep_dims=True, axis=1))
        except:
            self.max_norm = 0

        if(extract_embed):
            self.q1_extract = q1_output
            self.q2_extract = q2_output

        if self.args.implicit == 1:
            q1_output = tf.concat([q1_output, self.user_batch], 1)
            q2_output = tf.concat([q2_output, self.item_batch], 1)

        q1_output = tf.nn.dropout(q1_output, self.dropout)
        q2_output = tf.nn.dropout(q2_output, self.dropout)

        ''''
        if(self.mode=='HREC'):
            # Use Rec Style output
            if('TNET' not in self.args.rnn_type):
                output = self._rec_output(q1_output, q2_output,
                                        reuse=reuse,
                                        side=side)
            #elif("TNET" in self.args.rnn_type):
            #     # Learn Repr with CNN
            #    input_vec = tf.concat([q1_output, q2_output], 1)
            #    dim = q1_output.get_shape().as_list()[1]
            #    trans_output = ffn(input_vec, dim,
            #              self.initializer, name='transform',
            #              reuse=reuse,
            #              num_layers=2,
            #              dropout=None, activation=tf.nn.tanh)
            #    trans_cnn = self.learn_single_repr(self.trans_embed,
            #                                     self.trans_len,
            #                                     self.args.smax * 2,
            #                                     rnn_type,
            #                                     reuse=True, pool=False,
            #                                     name='main')
            #    trans_cnn = tf.reduce_max(trans_cnn, 1)
            #    self.trans_loss = tf.nn.l2_loss(trans_output - trans_cnn)
            #    # Alternative predict op using transform
            #    output = self._rec_output(trans_output, None,
            #                                reuse=reuse,
            #                                side=side,
            #                                name='target')

        representation = output
        return output, representation, att1, att2
        '''
        return q1_output, q2_output, att1, att2

    def learn_single_repr(self, q1_embed, q1_len, q1_max, rnn_type,
                        reuse=None, pool=False, name="", mask=None):
        """ This is the single sequence encoder function.
        rnn_type controls what type of encoder is used.
        Supports neural bag-of-words (NBOW) and CNN encoder
        """
        if('NBOW' in rnn_type):
            if mask is not None:
                masks = tf.cast(mask, tf.float32)
                masks = tf.expand_dims(masks, 2)
                masks = tf.tile(masks, [1, 1, self.args.emb_size])

                q1_embd = q1_embed * masks

            q1_output = tf.reduce_sum(q1_embed, 1)
            if(pool):
                return q1_embed, q1_output
        elif('CNN' in rnn_type):
            q1_output = build_raw_cnn(q1_embed, self.args.rnn_size,
                filter_sizes=3,
                initializer=self.initializer,
                dropout=self.rnn_dropout, reuse=reuse, name=name)
            if(pool):
                q1_output = tf.reduce_max(q1_output, 1)
                return q1_output, q1_output
        else:
            q1_output = q1_embed

        return q1_embed, q1_output

    def _rec_output(self, q1_output, q2_output, reuse=None, side="",
                        name=''):
        """ This function supports the final layer outputs of
        recommender models.

        Four options: 'DOT','MLP','MF' and 'FM'
        (should be self-explanatory)
        """
        print("Rec Output")
        print(q1_output)
        dim = q1_output.get_shape().as_list()[1]
        with tf.variable_scope('rec_out', reuse=reuse) as scope:
            if('DOT' in self.args.rnn_type):
                output = q1_output * q2_output
                output = tf.reduce_sum(output, 1, keep_dims=True)
            elif('MLP' in self.args.rnn_type):
                output = tf.concat([q1_output, q2_output,
                                q1_output * q2_output], 1)
                output = ffn(output, self.args.hdim,
                            self.initializer,
                            name='ffn', reuse=None,
                            dropout=self.dropout,
                            activation=tf.nn.relu, num_layers=4)
                output = linear(output, 1, self.initializer)
            elif('MF' in self.args.rnn_type):
                output = q1_output * q2_output
                h = tf.get_variable(
                            "hidden", [dim, 1],
                            initializer=self.initializer,
                            )
                output = tf.matmul(output, h)
            elif('FM' in self.args.rnn_type):
                if(q2_output is None):
                    input_vec = q1_output
                else:
                    input_vec = tf.concat([q1_output, q2_output], 1)
                input_vec = tf.nn.dropout(input_vec, self.dropout)
                output, _ = build_fm(input_vec, k=self.args.factor,
                                    reuse=reuse,
                                    name=name,
                                    initializer=self.initializer,
                                    reshape=False)

            if('SIG' in self.args.rnn_type):
                output = tf.nn.sigmoid(output)
            return output


    def build_single_cell(self):
        #rnn_cell = LSTMCell
        #if (self.rnn_cell.lower() == 'gru'):
        rnn_cell = GRUCell

        #with tf.variable_scope('gen_review', reuse=None) as scope:
        cell = rnn_cell(self.args.rnn_dim)

        return cell

    def _cal_key_loss(self, preds):

        if self.args.word_gumbel == 0:
            return tf.constant(0.0, dtype=tf.float32)

        preds = -tf.log(preds + 1e-7)
        #preds = tf.expand_dims(preds, 1)
        #batch_dim = preds.get_shape().as_list()[0]
        word = tf.concat([self.word_u, self.word_i], 1)
        num_words = word.get_shape().as_list()[1]
        prefix = tf.range(self.batch_size)
        prefix = tf.tile(tf.expand_dims(prefix, 1), [1, num_words])
        indices = tf.concat([tf.expand_dims(prefix, 2), tf.expand_dims(word, 2)], 2)

        l = tf.gather_nd(preds, indices)

        #l = tf.reduce_sum(preds * word, 2)

        if self.args.word_aggregate == 'MEAN':
            l = tf.reduce_mean(l, 1)
        else:
            if self.args.word_aggregate == 'MAX':
                l = tf.reduce_max(l, 1)
            else:
                l = tf.reduce_min(l, 1)

        l = self.args.key_word_lambda * tf.reduce_mean(l)

        return l


    def _beam_search_infer(self, q1_output, q2_output, r_input, reuse=None):
        dim = q1_output.get_shape().as_list()[1]
        with tf.variable_scope('gen_review', reuse=reuse) as scope:
            #cal state
            self.review_user_mapping = tf.get_variable(name='review_user_mapping',
                                                      shape=[dim, self.args.rnn_dim],
                                                      initializer=self.initializer)#, dtype=self.dtype)

            self.review_item_mapping = tf.get_variable(name='review_item_mapping',
                                                      shape=[dim, self.args.rnn_dim],
                                                      initializer=self.initializer)#, dtype=self.dtype)

            if not (self.args.feed_rating == 0):
                self.review_rating_embeddings = tf.get_variable(name='review_rating_embeddings',
                                                      shape=[5, self.args.rnn_dim],
                                                      initializer=self.initializer)#, dtype=self.dtype)

            self.review_bias = tf.get_variable(name='review_bias',
                                                      shape=[self.args.rnn_dim],
                                                      initializer=self.initializer)#, dtype=self.dtype)

            self.rnn_cell = self.build_single_cell()

            #cal state
            if r_input is not None:
                r_embed = tf.nn.embedding_lookup(self.review_rating_embeddings, r_input)
                state = tf.nn.tanh(r_embed + tf.matmul(q1_output, self.review_user_mapping) + tf.matmul(q2_output, self.review_item_mapping) + self.review_bias)
            else:
                state = tf.nn.tanh(tf.matmul(q1_output, self.review_user_mapping) + tf.matmul(q2_output, self.review_item_mapping) + self.review_bias)

            self.beam_batch = self.args.beam_size * self.args.batch_size
            self.beam_batch_max = self.args.beam_size * self.args.beam_size * self.args.batch_size

            #max_val = self.max_val
            #initializer = tf.random_uniform_initializer(-max_val, max_val, dtype=self.dtype)

            neg_words = tf.Variable(tf.constant(0.0, shape=[self.vocab_size]), trainable=False)
            neg_words = tf.reshape(tf.scatter_update(neg_words, tf.constant(self.UNK_tag), tf.constant(1.0)), shape=[1, self.vocab_size])

            neg_words_batch = tf.tile(neg_words, [self.args.batch_size, 1])
            neg_words_beam_batch = tf.tile(neg_words, [self.beam_batch, 1])

            pad_batch = tf.constant(self.PAD_tag, shape=[self.beam_batch])
            eos_batch = tf.constant(self.EOS_tag, shape=[self.beam_batch])
            min_batch = tf.constant(-100.0, shape=[self.beam_batch])
            zero_batch = tf.constant(0.0, shape=[self.beam_batch])

            #pad_batch_max = tf.constant(self.PAD_tag, shape=[self.beam_batch_max])
            eos_batch_max = tf.constant(self.EOS_tag, shape=[self.beam_batch_max])

            #self.init_state = tf.reshape(tf.tile(self.init_state, [1, self.beam_size]), shape = [self.beam_batch, self.hidden_units])
            #if self.use_strong_feeding:
            #    final_input = tf.concat([init_state, tf.constant(0.0, shape=[self.batch_size, self.embedding_size]), self.rating_batch_vectors, self.review_dense_outputs], axis=1)
            #else:
            final_input = state

            #with tf.variable_scope('gen_review', reuse=reuse) as scope:
            logits = tf.layers.dense(final_input, self.vocab_size, kernel_initializer=self.initializer, bias_initializer=self.initializer, name='review_output_layer')
            self.preds = tf.nn.softmax(logits) - neg_words_batch
            values, indices = tf.nn.top_k(self.preds, self.args.beam_size)

            init_ans = tf.reshape(indices, shape=[self.beam_batch, 1])
            init_loss = tf.log(tf.reshape(values, shape=[self.beam_batch]))
            init_tag = tf.cast(tf.equal(tf.reshape(indices, shape=[self.beam_batch]), eos_batch), tf.int32)
            init_end_tag = tf.reduce_sum(init_tag, axis=None)
            sum_tag = tf.constant(self.beam_batch, dtype=tf.int32)
            max_length = tf.constant(self.args.gmax, dtype=tf.int32)

            init_len = tf.constant(1, shape=[self.beam_batch])

            init_lm_inputs = tf.reshape(indices, shape=[self.beam_batch])
            init_state = tf.reshape(tf.tile(state, [1, self.args.beam_size]), shape = [self.beam_batch, self.args.rnn_dim])

            #rating_beam_batch_vectors = tf.reshape(tf.tile(tf.expand_dims(self.rating_batch_vectors, 1), [1, self.args.beam_size, 1]), shape=[-1, self.num_rating])
            #review_beam_batch_dense_outputs = tf.reshape(tf.tile(tf.expand_dims(self.review_dense_outputs, 1), [1, self.beam_size, 1]), shape=[-1, self.hidden_units])

            def condition(end_tag, tag, answer, lens, *args):
                return tf.logical_and(end_tag < sum_tag, tf.shape(answer)[1]<= max_length)

            def forward_one_step(end_tag, tag, answer, lens, loss, lm_inputs, state):
                self.tip_inputs_embedded = tf.nn.embedding_lookup(
                        params=self.embeddings, ids=lm_inputs)

                #if self.use_strong_feeding:
                #    self.rnnlm_outputs, new_state = self.rnnlm_cell(tf.concat([self.tip_inputs_embedded, rating_beam_batch_vectors, review_beam_batch_dense_outputs], axis=1), state)
                #else:
                self.rnnlm_outputs, new_state = self.rnn_cell(self.tip_inputs_embedded, state)

                #new_tag = 1 - tag
                loss_old_pad = tf.where(tf.cast(tag, tf.bool), zero_batch, min_batch)
                loss_old = loss + loss_old_pad
                tag_old = tag
                lens_old = lens
                state_old = state
                answer_old = tf.concat([answer, tf.reshape(pad_batch, shape=[self.beam_batch, 1])], axis=1)

                loss_new_pad = tf.where(tf.cast(tag, tf.bool), min_batch, zero_batch)
                loss_new = loss + loss_new_pad
                loss_new = tf.reshape(tf.tile(tf.reshape(loss_new, shape=[self.beam_batch, 1]), [1, self.args.beam_size]), shape = [self.beam_batch_max])

                #output_layer
                #if self.use_strong_feeding:
                #    final_input = tf.concat([self.rnnlm_outputs, self.tip_inputs_embedded, rating_beam_batch_vectors, review_beam_batch_dense_outputs], axis=1)
                #else:
                final_input = self.rnnlm_outputs
                #with tf.variable_scope('gen_review', reuse=reuse) as scope:
                logits = tf.layers.dense(final_input, self.vocab_size, kernel_initializer=self.initializer, bias_initializer=self.initializer, name='review_output_layer', reuse=True)
                #masking loss
                self.preds = tf.nn.softmax(logits) - neg_words_beam_batch
                values, indices = tf.nn.top_k(self.preds, self.args.beam_size)

                values = tf.reshape(values, shape=[self.beam_batch_max])
                loss_new = loss_new + tf.log(values)

                answer_new = tf.reshape(tf.tile(answer, [1, self.args.beam_size]), shape=[self.beam_batch_max, -1])
                indices = tf.reshape(indices, shape=[self.beam_batch_max])
                answer_new = tf.concat([answer_new, tf.reshape(indices, shape=[self.beam_batch_max, 1])], axis=1)

                state_new = tf.reshape(tf.tile(new_state, [1, self.args.beam_size]), shape=[self.beam_batch_max, -1])
                tag_new = tf.cast(tf.equal(indices, eos_batch_max), tf.int32)
                lens_new = tf.reshape(tf.tile(tf.reshape(lens, shape=[self.beam_batch, 1]), [1, self.args.beam_size]), shape=[self.beam_batch_max])
                lens_new = lens_new + 1

                #merge
                merge_tag = tf.concat([tf.reshape(tag_old, shape=[self.args.batch_size, -1]), tf.reshape(tag_new, shape=[self.args.batch_size, -1])], axis=1)
                merge_lens = tf.concat([tf.reshape(lens_old, shape=[self.args.batch_size, -1]), tf.reshape(lens_new, shape=[self.args.batch_size, -1])], axis=1)
                merge_state = tf.concat([tf.reshape(state_old, shape=[self.args.batch_size, self.args.beam_size, -1]), tf.reshape(state_new, shape=[self.args.batch_size, self.args.beam_size * self.args.beam_size, -1])], axis=1)
                merge_loss = tf.concat([tf.reshape(loss_old, shape=[self.args.batch_size, -1]), tf.reshape(loss_new, shape=[self.args.batch_size, -1])], axis=1)
                #average_loss = tf.div(merge_loss, merge_lens)
                merge_answer =  tf.concat([tf.reshape(answer_old, shape=[self.args.batch_size, self.args.beam_size, -1]), tf.reshape(answer_new, shape=[self.args.batch_size, self.args.beam_size * self.args.beam_size, -1])], axis=1)
                merge_inputs = tf.concat([tf.reshape(lm_inputs, shape=[self.args.batch_size, -1]), tf.reshape(indices, shape=[self.args.batch_size, -1])], axis=1)

                merge_values, merge_indices = tf.nn.top_k(merge_loss, self.args.beam_size)

                #new_loss = tf.reshape(merge_values, shape=[self.beam_batch])
                merge_indices = tf.reshape(merge_indices, shape=[self.beam_batch, 1])
                range_batch = tf.reshape(tf.tile(tf.reshape(tf.range(self.args.batch_size), shape=[self.args.batch_size, 1]), [1, self.args.beam_size]), shape=[self.beam_batch, 1])
                index = tf.concat([range_batch, merge_indices], axis=1)

                new_loss = tf.reshape(tf.gather_nd(merge_loss, index), shape=[self.beam_batch])
                new_tag = tf.reshape(tf.gather_nd(merge_tag, index), shape=[self.beam_batch])
                new_lens = tf.reshape(tf.gather_nd(merge_lens, index), shape=[self.beam_batch])
                new_state = tf.reshape(tf.gather_nd(merge_state, index), shape=[self.beam_batch, -1])
                new_answer = tf.reshape(tf.gather_nd(merge_answer, index), shape=[self.beam_batch, -1])
                new_inputs = tf.reshape(tf.gather_nd(merge_inputs, index), shape=[self.beam_batch])

                sum_end = tf.reduce_sum(new_tag, axis=None)

                return sum_end, new_tag, new_answer, new_lens, new_loss, new_inputs, new_state

            sum_end, tag, answer, lens, loss, lm_inputs, state = tf.while_loop(condition, forward_one_step, [init_end_tag, init_tag, init_ans, init_len, init_loss, init_lm_inputs, init_state],  shape_invariants=[init_end_tag.get_shape(), init_tag.get_shape(), tf.TensorShape([self.beam_batch, None]), init_len.get_shape(), init_loss.get_shape(), init_lm_inputs.get_shape(), init_state.get_shape()])

        return answer


    def _gen_review(self, q1_output, q2_output, r_input, reuse=None):
        print ("Gen Output")
        dim = q1_output.get_shape().as_list()[1]
        batch_dim = q1_output.get_shape().as_list()[0]
        with tf.variable_scope('gen_review', reuse=reuse) as scope:
            #cal state
            #self.review_user_mapping = tf.get_variable(name='review_user_mapping',
            #                                          shape=[dim, self.args.rnn_dim],
            #                                          initializer=self.initializer)#, dtype=self.dtype)

            #self.review_item_mapping = tf.get_variable(name='review_item_mapping',
            #                                          shape=[dim, self.args.rnn_dim],
            #                                          initializer=self.initializer)#, dtype=self.dtype)

            #self.review_bias = tf.get_variable(name='review_bias',
            #                                          shape=[self.args.rnn_dim],
            #                                          initializer=self.initializer)#, dtype=self.dtype)

            
            #self.rnn_cell = self.build_single_cell()

            #cal state
            if r_input is not None:
                r_embed = tf.nn.embedding_lookup(self.review_rating_embeddings, r_input)
                state = tf.nn.tanh(r_embed + tf.matmul(q1_output, self.review_user_mapping) + tf.matmul(q2_output, self.review_item_mapping) + self.review_bias)
            else:
                state = tf.nn.tanh(tf.matmul(q1_output, self.review_user_mapping) + tf.matmul(q2_output, self.review_item_mapping) + self.review_bias)

            review_inputs_transpose = tf.transpose(self.gen_outputs, perm=[1,0])
            #print (review_inputs_transpose.get_shape())
            #print (q1_output.get_shape())
            #print (state.get_shape())
            max_review_length = tf.reduce_max(self.gen_len)
            masks = tf.transpose(tf.sequence_mask(self.gen_len, maxlen=max_review_length, dtype=tf.float32), perm=[1,0])
            #masks = tf.transpose(tf.sequence_mask(self.gen_len, maxlen=max_review_length), perm=[1,0])

            #predict first word
            self.dropout_out = tf.nn.dropout(state, self.dropout)
            #if self.use_strong_feeding:
            #    final_input = tf.concat([self.dropout_out, tf.constant(0.0, shape=[self.batch_size, self.embedding_size]), self.rating_batch_vectors, self.review_dense_outputs], axis=1)
            #else:
            final_input = self.dropout_out

            logits = tf.layers.dense(final_input, self.vocab_size, kernel_initializer=self.initializer, bias_initializer=self.initializer, name='review_output_layer', reuse=True)
            self.preds = tf.nn.softmax(logits)

            #if self.args.concept == 1:
            key_loss = self._cal_key_loss(self.preds)

            init_sum_key_loss = key_loss

            argm = tf.argmax(self.preds, axis=1, output_type = tf.int32)
            castm = tf.cast(argm, tf.int32)
            correct = tf.equal(castm,tf.gather(review_inputs_transpose, 1))

            init_accuracy = tf.reduce_sum(tf.cast(correct, tf.float32) * tf.gather(masks, 1))

            init_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.gather(review_inputs_transpose, 1), logits = logits) * tf.gather(masks, 1))

            init_sum = tf.reduce_sum(tf.gather(masks,1))#tf.constant(0, dtype=tf.int32)#batch_dim#tf.cast(batch_dim, dtype=tf.float32) #tf.constant(self.batch_size, dtype=self.dtype)
            init_iteration = tf.constant(1, dtype = tf.int32)


            def condition(i, *args):
                return i < max_review_length - 1


            #def forward_one_step(i, sum, loss, accuracy, state):
            def forward_one_step(i, sum, loss, sum_key_loss, accuracy, state):

                self.review_inputs_embedded = tf.nn.embedding_lookup(
                    params=self.embeddings, ids=tf.gather(review_inputs_transpose, i))

                #if self.use_strong_feeding:
                #    self.rnnlm_outputs, new_state = self.rnnlm_cell(tf.concat([self.tip_inputs_embedded, self.rating_batch_vectors, self.review_dense_outputs], axis=1), state)
                #else:
                self.rnnlm_outputs, new_state = self.rnn_cell(self.review_inputs_embedded, state)

                #dropout before output layer
                self.dropout_out = tf.nn.dropout(self.rnnlm_outputs, self.dropout)
                #if self.use_strong_feeding:
                #    final_input = tf.concat([self.dropout_out, self.tip_inputs_embedded, self.rating_batch_vectors, self.review_dense_outputs], axis=1)
                #else:
                final_input = self.dropout_out

                #output_layer
                logits = tf.layers.dense(final_input, self.vocab_size, kernel_initializer=self.initializer, bias_initializer=self.initializer, name='review_output_layer', reuse=True)
                #masking loss
                self.preds = tf.nn.softmax(logits)
                argm = tf.argmax(self.preds, axis=1, output_type = tf.int32)
                castm = tf.cast(argm, tf.int32)

                #if self.args.concept == 1:
                key_loss = self._cal_key_loss(self.preds)
                sum_key_loss += key_loss

                correct = tf.equal(castm,tf.gather(review_inputs_transpose, i+1))
                accuracy = accuracy + tf.reduce_sum(tf.cast(correct, tf.float32) * tf.gather(masks, i+1))
                sum = sum + tf.reduce_sum(masks[i+1])

                loss = loss + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.gather(review_inputs_transpose, i+1), logits = logits) * tf.gather(masks, i+1))

                #return i+1, sum, loss, accuracy, new_state

                #iteration, sum, loss, accuracy, state = tf.while_loop(condition, forward_one_step, [init_iteration, init_sum, init_loss, init_accuracy, state])
                return i+1, sum, loss, sum_key_loss, accuracy, new_state

            iteration, sum, loss, sum_key_loss, accuracy, state = tf.while_loop(condition, forward_one_step, [init_iteration, init_sum, init_loss, init_sum_key_loss, init_accuracy, state])

            #sum += batch_dim
            review_loss = loss# / batch_dim# / sum
            review_acc = accuracy / sum

        return review_loss, review_acc, sum_key_loss


    def prepare_hierarchical_input(self):
        """ Supports hierarchical data input
        Converts word level -> sentence level
        """
        # q1_inputs, self.qmax = clip_sentence(self.q1_inputs, self.q1_len)
        # q2_inputs, self.a1max = clip_sentence(self.q2_inputs, self.q2_len)
        # q3_inputs, self.a2max = clip_sentence(self.q3_inputs, self.q3_len)

        # Build word-level masks
        self.q1_mask = tf.cast(self.q1_inputs, tf.bool)
        self.q2_mask = tf.cast(self.q2_inputs, tf.bool)
        self.q3_mask = tf.cast(self.q3_inputs, tf.bool)

        def make_hmasks(inputs, smax):
            # Hierarchical Masks
            # Inputs are bsz x (dmax * smax)
            inputs = tf.reshape(inputs,[-1, smax])
            masked_inputs = tf.cast(inputs, tf.bool)
            return masked_inputs

        # Build review-level masks
        self.q1_hmask = make_hmasks(self.q1_inputs, self.args.smax)
        self.q2_hmask = make_hmasks(self.q2_inputs, self.args.smax)
        self.q3_hmask = make_hmasks(self.q3_inputs, self.args.smax)

        with tf.device('/cpu:0'):
            q1_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                self.q1_inputs)
            q2_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                self.q2_inputs)
            q3_embed = tf.nn.embedding_lookup(self.embeddings,
                                                self.q3_inputs)

        print("=============================================")
        # This is found in nn.py in tylib
        print("Hierarchical Flattening")
        q1_embed, q1_len = hierarchical_flatten(q1_embed,
                                            self.q1_len,
                                            self.args.smax)
        q2_embed, q2_len = hierarchical_flatten(q2_embed,
                                            self.q2_len,
                                            self.args.smax)
        q3_embed, q3_len = hierarchical_flatten(q3_embed,
                                            self.q3_len,
                                            self.args.smax)
        print(q1_len)

        self.o1_embed = q1_embed
        self.o2_embed = q2_embed
        self.o3_embed = q3_embed
        self.o1_len = q1_len
        self.o2_len = q2_len
        self.o3_len = q3_len

        if self.args.masking == 1:
            _, q1_embed = self.learn_single_repr(q1_embed, q1_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=None, pool=True,
                                            name='sent', mask=self.q1_hmask)
            _, q2_embed = self.learn_single_repr(q2_embed, q2_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=True, pool=True,
                                            name='sent', mask=self.q2_hmask)
            _, q3_embed = self.learn_single_repr(q3_embed, q3_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=True, pool=True,
                                            name='sent', mask=self.q3_hmask)
        else:
            _, q1_embed = self.learn_single_repr(q1_embed, q1_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=None, pool=True,
                                            name='sent', mask=None)
            _, q2_embed = self.learn_single_repr(q2_embed, q2_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=True, pool=True,
                                            name='sent', mask=None)
            _, q3_embed = self.learn_single_repr(q3_embed, q3_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=True, pool=True,
                                            name='sent', mask=None)

        _dim = q1_embed.get_shape().as_list()[1]
        q1_embed = tf.reshape(q1_embed, [-1, self.args.dmax, _dim])
        q2_embed = tf.reshape(q2_embed, [-1, self.args.dmax, _dim])
        q3_embed = tf.reshape(q3_embed, [-1, self.args.dmax, _dim])
        self.q1_embed = q1_embed
        self.q2_embed = q2_embed
        self.q3_embed = q3_embed
        self.qmax = self.args.dmax
        self.a1max = self.args.dmax
        self.a2max = self.args.dmax
        # Doesn't support any of these yet
        self.c1_cnn, self.c2_cnn, self.c3_cnn = None, None, None
        self.p1_pos, self.p2_pos, self.p3_pos = None, None, None
        if('TNET' in self.args.rnn_type):
            t_inputs, _ = clip_sentence(self.trans_inputs, self.trans_len)
            self.trans_embed = tf.nn.embedding_lookup(self.embeddings,
                                                        t_inputs)
        print("=================================================")

    def prepare_inputs(self):
        """ Prepares Input
        """
        q1_inputs, self.qmax = clip_sentence(self.q1_inputs, self.q1_len)
        q2_inputs, self.a1max = clip_sentence(self.q2_inputs, self.q2_len)
        q3_inputs, self.a2max = clip_sentence(self.q3_inputs, self.q3_len)

        self.q1_mask = tf.cast(q1_inputs, tf.bool)
        self.q2_mask = tf.cast(q2_inputs, tf.bool)
        self.q3_mask = tf.cast(q3_inputs, tf.bool)

        with tf.device('/cpu:0'):
            q1_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                    q1_inputs)
            q2_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                    q2_inputs)
            q3_embed = tf.nn.embedding_lookup(self.embeddings,
                                                    q3_inputs)

        if(self.args.all_dropout):
            # By default, this is disabled
            q1_embed = tf.nn.dropout(q1_embed, self.emb_dropout)
            q2_embed = tf.nn.dropout(q2_embed, self.emb_dropout)
            q3_embed = tf.nn.dropout(q3_embed, self.emb_dropout)

        # Ignore these. :)
        self.c1_cnn, self.c2_cnn, self.c3_cnn = None, None, None
        self.p1_pos, self.p2_pos, self.p3_pos = None, None, None

        if('TNET' in self.args.rnn_type):
            t_inputs, _ = clip_sentence(self.trans_inputs, self.trans_len)
            self.trans_embed = tf.nn.embedding_lookup(self.embeddings,
                                                        t_inputs)

        self.q1_embed = q1_embed
        self.q2_embed = q2_embed
        self.q3_embed = q3_embed

    def build_graph(self):
        ''' Builds Computational Graph
        '''
        if(self.mode=='HREC' and self.args.base_encoder!='Flat'):
            len_shape = [None, None]
        else:
            len_shape = [None]

        print("Building placeholders with shape={}".format(len_shape))

        with self.graph.as_default():
            self.is_train = tf.get_variable("is_train",
                                            shape=[],
                                            dtype=tf.bool,
                                            trainable=False)
            self.true = tf.constant(True, dtype=tf.bool)
            self.false = tf.constant(False, dtype=tf.bool)
            with tf.name_scope('q1_input'):
                self.q1_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.qmax],
                                                    name='q1_inputs')
            with tf.name_scope('q2_input'):
                self.q2_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='q2_inputs')
            with tf.name_scope('c1_input'):
                self.c1_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.qmax],
                                                    name='c1_inputs')
            with tf.name_scope('c2_input'):
                self.c2_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='c2_inputs')
            with tf.name_scope('q3_input'):
                # supports pairwise mode.
                self.q3_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='q3_inputs')

            with tf.name_scope('gen_output'):
                self.gen_outputs = tf.placeholder(tf.int32, shape=[None,
                                                    #self.args.gmax],
                                                    None],
                                                    name='gen_outputs')

            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf.float32,
                                                name='dropout')
                self.rnn_dropout = tf.placeholder(tf.float32,
                                                name='rnn_dropout')
                self.emb_dropout = tf.placeholder(tf.float32,
                                                name='emb_dropout')
            with tf.name_scope('q1_lengths'):
                self.q1_len = tf.placeholder(tf.int32, shape=len_shape)
            with tf.name_scope('q2_lengths'):
                self.q2_len = tf.placeholder(tf.int32, shape=len_shape)
            with tf.name_scope('c1_lengths'):
                self.c1_len = tf.placeholder(tf.int32, shape=len_shape)
            with tf.name_scope('c2_lengths'):
                self.c2_len = tf.placeholder(tf.int32, shape=len_shape)
            with tf.name_scope('q3_lengths'):
                self.q3_len = tf.placeholder(tf.int32, shape=len_shape)
            if self.args.implicit == 1:
                with tf.name_scope('user_id'):
                    self.user_id = tf.placeholder(tf.int32, shape=[None])
                with tf.name_scope('item_id'):
                    self.item_id = tf.placeholder(tf.int32, shape=[None])

            with tf.name_scope('gen_len'):
                self.gen_len = tf.placeholder(tf.int32, shape=[None])
            with tf.name_scope('learn_rate'):
                self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
            with tf.name_scope('batch_size'):
                self.batch_size = tf.placeholder(tf.int32, name='batch_size')
            if(self.args.pretrained==1):
                self.emb_placeholder = tf.placeholder(tf.float32,
                            [self.vocab_size, self.args.emb_size])

            with tf.name_scope("soft_labels"):
                # softmax cross entropy (not used here)
                data_type = tf.int32
                self.soft_labels = tf.placeholder(data_type,
                             shape=[None, self.args.num_class],
                             name='softmax_labels')

            with tf.name_scope("sig_labels"):
                # sigmoid cross entropy
                self.sig_labels = tf.placeholder(tf.float32,
                                                shape=[None],
                                                name='sigmoid_labels')
                self.sig_target = tf.expand_dims(self.sig_labels, 1)

            self.batch_size = tf.shape(self.q1_inputs)[0]

            with tf.variable_scope('embedding_layer'):
                if(self.args.pretrained==1):
                    self.embeddings = tf.Variable(tf.constant(
                                        0.0, shape=[self.vocab_size,
                                            self.args.emb_size]), \
                                        trainable=self.args.trainable,
                                         name="embeddings")
                    self.embeddings_init = self.embeddings.assign(
                                        self.emb_placeholder)
                else:
                    self.embeddings = tf.get_variable('embedding',
                                        [self.vocab_size,
                                        self.args.emb_size],
                                        initializer=self.initializer)

                if self.args.implicit == 1:
                    self.user_embeddings = tf.get_variable('user_embedding',
                                        [self.num_user,
                                        self.args.latent_size],
                                        initializer=self.initializer)

                    self.item_embeddings = tf.get_variable('item_embedding',
                                        [self.num_item,
                                        self.args.latent_size],
                                        initializer=self.initializer)

                    self.user_batch = tf.nn.embedding_lookup(self.user_embeddings,
                                                self.user_id)

                    self.item_batch = tf.nn.embedding_lookup(self.item_embeddings,
                                                self.item_id)

            self.i1_embed, self.i2_embed, self.i3_embed = None, None, None

            if('TNET' in self.args.rnn_type):
                self.trans_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.smax * 2],
                                                    name='trans_inputs')
                self.trans_len = tf.placeholder(tf.int32, shape=[None])

            if(self.mode=='HREC' and self.args.base_encoder!='Flat'):
                # Hierarchical mode
                self.prepare_hierarchical_input()
                q1_len = tf.cast(tf.count_nonzero(self.q1_len, axis=1),
                                    tf.int32)
                q2_len = tf.cast(tf.count_nonzero(self.q2_len, axis=1),
                                    tf.int32)
                q3_len = tf.cast(tf.count_nonzero(self.q3_len, axis=1),
                                    tf.int32)
            else:
                print("Flat Mode..")
                self.prepare_inputs()
                q1_len = self.q1_len
                q2_len = self.q2_len
                q3_len = self.q3_len
                self.o1_embed = None
                self.o2_embed = None
                self.o3_embed = None
                self.o1_len = None
                self.o2_len = None
                self.o3_len = None

            print (self.q1_embed.get_shape(), self.q2_embed.get_shape())
             
            #self.output_pos, _, _, _ = self._joint_representation(
            q1_output, q2_output, _, _ = self._joint_representation(
                                        self.q1_embed, self.q2_embed,
                                        q1_len, q2_len,
                                        self.qmax, self.a1max,
                                        score=1, reuse=None,
                                        #features=self.pos_features,
                                        features = None,
                                        extract_embed=True, side='POS',
                                        c1_embed=self.c1_cnn,
                                        c2_embed=self.c2_cnn,
                                        p1_embed=self.p1_pos,
                                        p2_embed=self.p2_pos,
                                        i1_embed=self.i1_embed,
                                        i2_embed=self.i2_embed,
                                        o1_embed=self.o1_embed,
                                        o2_embed=self.o2_embed,
                                        o1_len=self.o1_len,
                                        o2_len=self.o2_len,
                                        q1_mask=self.q1_mask,
                                        q2_mask=self.q2_mask
                                        )
            if('SOFT' not in self.args.rnn_type and 'RAW_MSE' not in self.args.rnn_type):
                """ This is only for pairwise ranking and not relevant to this repo!
                """
                print("Building Negative Graph...")
                self.output_neg,_,_, _ = self._joint_representation(
                                            self.q1_embed,
                                             self.q3_embed, q1_len,
                                             q3_len, self.qmax,
                                             self.a2max, score=1,
                                             reuse=True,
                                             features=self.neg_features,
                                             side='NEG',
                                             c1_embed=self.c1_cnn,
                                             c2_embed=self.c3_cnn,
                                             p1_embed=self.p1_pos,
                                             p2_embed=self.p3_pos,
                                             i1_embed=self.i1_embed,
                                             i2_embed=self.i3_embed,
                                             o1_embed=self.o1_embed,
                                             o2_embed=self.o3_embed,
                                             o1_len=self.o1_len,
                                             o2_len=self.o3_len,
                                             q1_mask=self.q1_mask,
                                             q2_mask=self.q3_mask
                                             )
            else:
                self.output_neg = None

            if(self.mode=='HREC'):
                # Use Rec Style output
                if('TNET' not in self.args.rnn_type):
                    self.output_pos = self._rec_output(q1_output, q2_output,
                                                       reuse=None,
                                                       side='POS')
            
            # Define loss and optimizer
            with tf.name_scope("train"):
                with tf.name_scope("cost_function"):
                    if("SOFT" in self.args.rnn_type):
                        target = self.soft_labels
                        if('POINT' in self.args.rnn_type):
                            target = tf.argmax(target, 1)
                            target = tf.expand_dims(target, 1)
                            target = tf.cast(target, tf.float32)
                            ce = tf.nn.sigmoid_cross_entropy_with_logits(
                                                logits=self.output_pos,
                                                labels=target)
                        else:
                            ce = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                    logits=self.output_pos,
                                                    labels=tf.stop_gradient(target))
                        self.cost = tf.reduce_mean(ce)
                    elif('RAW_MSE' in self.args.rnn_type):
                        sig = self.output_pos
                        target = tf.expand_dims(self.sig_labels, 1)
                        self.cost = tf.reduce_mean(
                                    tf.square(tf.subtract(target, sig)))
                    elif('LOG' in self.args.rnn_type):
                        # BPR loss for ranking
                        self.cost = tf.reduce_mean(
                                    -tf.log(tf.nn.sigmoid(
                                        self.output_pos-self.output_neg)))
                    else:
                        # Hinge loss for ranking
                        self.hinge_loss = tf.maximum(0.0,(
                                self.args.margin - self.output_pos \
                                + self.output_neg))

                        self.cost = tf.reduce_sum(self.hinge_loss)

                    self.cost = self.cost * self.args.rating_lambda

                    #with tf.name_scope('regularization'):
                    #    if(self.args.l2_reg>0):
                    #        vars = tf.trainable_variables()
                    #        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                    #                            if 'bias' not in v.name ])
                    #        lossL2 *= self.args.l2_reg
                    #        self.cost += lossL2

                    if self.args.feed_rating == 0:
                        r_input = None
                    elif self.args.feed_rating == 1:
                        #move from [1 ,5] -> [0, 4]
                        r_input = tf.cast(self.sig_labels, dtype=tf.int32) - 1
                    else:
                        r_input = tf.clip_by_value(tf.cast(tf.reshape(self.output_pos, [-1]), dtype=tf.int32), 1, 5) - 1

                    with tf.name_scope('generation_results'):
                        self.gen_results = self._beam_search_infer(q1_output, q2_output, r_input)                  

                    with tf.name_scope('generation_loss'):
                        #self.gen_loss, self.gen_acc = self._gen_review(q1_output, q2_output)
                        self.gen_loss, self.gen_acc, self.key_word_loss = self._gen_review(q1_output, q2_output, r_input)
                        if self.args.word_gumbel == 1:
                            if (self.args.key_word_lambda != 0.0) and (self.args.concept == 1):
                                self.gen_loss += self.args.key_word_lambda * self.key_word_loss
                        self.cost += self.args.gen_lambda * self.gen_loss

                    self.task_cost = self.cost

                    with tf.name_scope('regularization'):
                        if(self.args.l2_reg>0):
                            vars = tf.trainable_variables()
                            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                                                if 'bias' not in v.name ])
                            lossL2 *= self.args.l2_reg
                            self.cost += lossL2

                    tf.summary.scalar("cost_function", self.cost)
                global_step = tf.Variable(0, trainable=False)

                if(self.args.dev_lr>0):
                    lr = self.learn_rate
                else:
                    if(self.args.decay_steps>0):
                        lr = tf.train.exponential_decay(self.args.learn_rate,
                                      global_step,
                                      self.args.decay_steps,
                                       self.args.decay_lr,
                                       staircase=self.args.decay_stairs)
                    elif(self.args.decay_lr>0 and self.args.decay_epoch>0):
                        decay_epoch = self.args.decay_epoch
                        lr = tf.train.exponential_decay(self.args.learn_rate,
                                      global_step,
                                      decay_epoch * self.args.batch_size,
                                       self.args.decay_lr, staircase=True)
                    else:
                        lr = self.args.learn_rate

                control_deps = []

                with tf.name_scope('optimizer'):
                    if(self.args.opt=='SGD'):
                        self.opt = tf.train.GradientDescentOptimizer(
                            learning_rate=lr)
                    elif(self.args.opt=='Adam'):
                        self.opt = tf.train.AdamOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adadelta'):
                        self.opt = tf.train.AdadeltaOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adagrad'):
                        self.opt = tf.train.AdagradOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='RMS'):
                        self.opt = tf.train.RMSPropOptimizer(
                                    learning_rate=lr)
                    elif(self.args.opt=='Moment'):
                        self.opt = tf.train.MomentumOptimizer(lr, 0.9)

                    # Use SGD at the end for better local minima
                    #self.opt2 = tf.train.GradientDescentOptimizer(
                    #        learning_rate=self.args.wiggle_lr)
                    tvars = tf.trainable_variables()
                    def _none_to_zero(grads, var_list):
                        return [grad if grad is not None else tf.zeros_like(var)
                              for var, grad in zip(var_list, grads)]
                    if(self.args.clip_norm>0):
                        grads, _ = tf.clip_by_global_norm(
                                        tf.gradients(self.cost, tvars),
                                        self.args.clip_norm)
                        with tf.name_scope('gradients'):
                            gradients = self.opt.compute_gradients(self.cost)
                            def ClipIfNotNone(grad):
                                if grad is None:
                                    return grad
                                grad = tf.clip_by_value(grad, -10, 10, name=None)
                                return tf.clip_by_norm(grad, self.args.clip_norm)
                            if(self.args.clip_norm>0):
                                clip_g = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                            else:
                                clip_g = [(grad,var) for grad,var in gradients]

                        # Control dependency for center loss
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.apply_gradients(clip_g,
                                                global_step=global_step)
                            #self.wiggle_op = self.opt2.apply_gradients(clip_g,
                            #                    global_step=global_step)
                    else:
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.minimize(self.cost)
                            #self.wiggle_op = self.opt2.minimize(self.cost)

                self.grads = _none_to_zero(tf.gradients(self.cost,tvars), tvars)
                # grads_hist = [tf.summary.histogram("grads_{}".format(i), k) for i, k in enumerate(self.grads) if k is not None]
                self.merged_summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
                # model_stats()

                print(self.output_pos)

                 # for Inference
                self.predict_op = self.output_pos
                #if('RAW_MSE' in self.args.rnn_type):
                #    self.predict_op = tf.clip_by_value(self.predict_op, 1, 5)
                #if('SOFT' in self.args.rnn_type):
                #    if('POINT' in self.args.rnn_type):
                #        predict_neg = 1 - self.predict_op
                #        self.predict_op = tf.concat([predict_neg,
                #                         self.predict_op], 1)
                #    else:
                #        self.predict_op = tf.nn.softmax(self.output_pos)
                #    self.predictions = tf.argmax(self.predict_op, 1)
                #    self.correct_prediction = tf.equal(tf.argmax(self.predict_op, 1),
                #                                    tf.argmax(self.soft_labels, 1))
                #    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                #                                    "float"))


                #save model
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=0)
