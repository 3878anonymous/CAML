from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import csv
import argparse
from keras.preprocessing import sequence
from datetime import datetime
import numpy as np
import random
import codecs
np.random.seed(1337)  # for reproducibility
random.seed(1337)
import os
from tqdm import tqdm
from utilities import *
from tylib.exp.metrics import *
import time
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from collections import Counter
#import cPickle as pickle
import pickle
from keras.utils import np_utils
import visdom
import string
import re
import math
import operator
from utilities import *
from collections import defaultdict
import sys
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from tf_models.model_caml import Model
# from tf_models.rec_model import RecModel
from tylib.exp.experiment_caml import Experiment
from tylib.exp.exp_ops import *
from parser_CAML import *
from sklearn.metrics import mean_absolute_error
from nltk.translate.bleu_score import corpus_bleu


PAD = "<PAD>"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"


def batchify(data, i, bsz, max_sample):
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if(end>max_sample):
        end = max_sample
    new_data = data[start:end]
    for i in range(bsz):
        new_data.append(data[i])
    return new_data[:bsz]

class CFExperiment(Experiment):
    """ Main experiment class for collaborative filtering.

    Check tylib/exp/experiment.py for base class.
    """

    def __init__(self, inject_params=None):
        print("Starting Rec Experiment")
        super(CFExperiment, self).__init__()
        self.uuid = datetime.now().strftime("%d:%m:%H:%M:%S")
        self.parser = build_parser()
        self.no_text_mode = False

        self.args = self.parser.parse_args()

        self.max_val = 5
        self.min_val = 1

        self.show_metrics = ['MSE','RMSE','MAE', 'MSE_int','RMSE_int','MAE_int', 'Gen_loss', 'All_loss', 'Gen_loss', 'F1', 'ACC', 'Review_acc']
        #self.eval_primary = 'RMSE'
        self.eval_primary = 'All_loss'
        # For hierarchical setting
        self.args.qmax = self.args.smax * self.args.dmax
        self.args.amax = self.args.smax * self.args.dmax

        print("Setting up environment..")

        self.model_wrapper()

        #self.env = dictFromFileUnicode(data_path)
        self.model_name = self.args.rnn_type
        self._setup()

        self._load_sets()

        self.mdl = Model(self.vocab, self.args,
                            #char_vocab=len(self.char_index),
                            #pos_vocab=len(self.pos_index),
                            mode='HREC', num_item=self.num_items,
                            num_user=self.num_users)

        self._print_model_stats()
        self.hyp_str = self.model_name + '_' + self.uuid
        self._setup_tf(load_embeddings=not self.no_text_mode)

    def model_wrapper(self):
        """ Converts model name to consituent components.
        """
        original = self.args.rnn_type
        if(self.args.rnn_type=='DeepCoNN'):
            self.args.rnn_type = 'RAW_MSE_MAX_CNN_FM'
            self.args.base_encoder = 'Flat'
        elif(self.args.rnn_type=='TRANSNET'):
            self.args.rnn_type = 'RAW_MSE_MAX_CNN_FM_TNET'
            self.args.base_encoder = 'Flat'
        elif(self.args.rnn_type=='DATT'):
            self.args.rnn_type ='RAW_MSE_DUAL_DOT'
            self.args.base_encoder = 'Flat'
        elif(self.args.rnn_type=='CAML'):
            self.args.rnn_type = 'RAW_MSE_CAML_FN_FM'
            self.args.base_encoder = 'NBOW'
        elif(self.args.rnn_type=='CAML_mlp'):
            self.args.rnn_type = 'RAW_MSE_CAML_FN_MLP'
            self.args.base_encoder = 'NBOW'

        print("Conversion to {} | base:{}".format(
                                self.args.rnn_type,
                                self.args.base_encoder))


    def _combine_reviews(self, data, reviews = None):
        user = [x[0] for x in data]
        items = [x[1] for x in data]
        labels = [x[2] for x in data]

        #prep generation outputs
        if reviews != None:

            gen_outputs, gen_len = prep_data_list(reviews, self.args.gmax)

        output = []
        for i in range(len(user)):
            output.append([user[i], items[i], labels[i], gen_outputs[i], gen_len[i]])

        return output

    def _prepare_set(self, data):

        user = [x[0] for x in data]
        items = [x[1] for x in data]
        labels = [x[2] for x in data]

        # Raw user-item ids
        user_idx = user
        item_idx = items


        user_list = []
        item_list = []
        user_concept_list = []
        item_concept_list = []
        user_len = []
        item_len = []
        user_concept_len = []
        item_concept_len = []
        #user_dict = {}
        for i in range(len(user)):
            user_reviews = []
            item_reviews = []
            user_concepts = []
            item_concepts = []
            user_r_len = []
            user_c_len = []
            item_r_len = []
            item_c_len = []

            if self.args.data_prepare == 1:
                if items[i] in self.ui_review_dict[user[i]]:
                    user_reviews.append(self.ui_review_dict[user[i]][items[i]])
                    user_concepts.append(self.ui_concept_dict[user[i]][items[i]])
                    user_r_len.append(len(self.ui_review_dict[user[i]][items[i]]))
                    user_c_len.append(len(self.ui_concept_dict[user[i]][items[i]]))
                for x in self.ui_review_dict[user[i]]:
                    if not x==items[i]:
                        user_reviews.append(self.ui_review_dict[user[i]][x])
                        user_concepts.append(self.ui_concept_dict[user[i]][x])
                        user_r_len.append(len(self.ui_review_dict[user[i]][x]))
                        user_c_len.append(len(self.ui_concept_dict[user[i]][x]))
                        if len(user_reviews) == self.args.dmax:
                            break
                if user[i] in self.iu_review_dict[items[i]]:
                    item_reviews.append(self.iu_review_dict[items[i]][user[i]])
                    item_concepts.append(self.iu_concept_dict[items[i]][user[i]])
                    item_r_len.append(len(self.iu_review_dict[items[i]][user[i]]))
                    item_c_len.append(len(self.iu_concept_dict[items[i]][user[i]]))
                for x in self.iu_review_dict[items[i]]:
                    if not x==user[i]:
                        item_reviews.append(self.iu_review_dict[items[i]][x])
                        item_concepts.append(self.iu_concept_dict[items[i]][x])
                        item_r_len.append(len(self.iu_review_dict[items[i]][x]))
                        item_c_len.append(len(self.iu_concept_dict[items[i]][x]))
                        if len(item_reviews) == self.args.dmax:
                            break
                user_list.append(user_reviews)
                item_list.append(item_reviews)
                user_concept_list.append(user_concepts)
                item_concept_list.append(item_concepts)
                user_len.append(user_r_len)
                item_len.append(item_r_len)
                user_concept_len.append(user_c_len)
                item_concept_len.append(item_c_len)
            elif self.args.data_prepare == -1:
                tmp = len(self.ui_review_dict[user[i]])
                for x in self.ui_review_dict[user[i]]:
                    if (not x==items[i]):
                        user_reviews.append(self.ui_review_dict[user[i]][x])
                        user_concepts.append(self.ui_concept_dict[user[i]][x])
                        user_r_len.append(len(self.ui_review_dict[user[i]][x]))
                        user_c_len.append(len(self.ui_concept_dict[user[i]][x]))
                        if len(user_reviews) == self.args.dmax:
                            break
                tmp = len(self.iu_review_dict[items[i]])
                for x in self.iu_review_dict[items[i]]:
                    if (not x==user[i]):
                        item_reviews.append(self.iu_review_dict[items[i]][x])
                        item_concepts.append(self.iu_concept_dict[items[i]][x])
                        item_r_len.append(len(self.iu_review_dict[items[i]][x]))
                        item_c_len.append(len(self.iu_concept_dict[items[i]][x]))
                        if len(item_reviews) == self.args.dmax:
                            break
                user_list.append(user_reviews)
                item_list.append(item_reviews)
                user_concept_list.append(user_concepts)
                item_concept_list.append(item_concepts)
                user_len.append(user_r_len)
                item_len.append(item_r_len)
                user_concept_len.append(user_c_len)
                item_concept_len.append(item_c_len)
            else:
                for x in self.ui_review_dict[user[i]]:
                    user_reviews.append(self.ui_review_dict[user[i]][x])
                    user_concepts.append(self.ui_concept_dict[user[i]][x])
                    user_r_len.append(len(self.ui_review_dict[user[i]][x]))
                    user_c_len.append(len(self.ui_concept_dict[user[i]][x]))
                    if len(user_reviews) == self.args.dmax:
                        break
                user_list.append(user_reviews)
                user_concept_list.append(user_concepts)
                for x in self.iu_review_dict[items[i]]:
                    item_reviews.append(self.iu_review_dict[items[i]][x])
                    item_concepts.append(self.iu_concept_dict[items[i]][x])
                    item_r_len.append(len(self.iu_review_dict[items[i]][x]))
                    item_c_len.append(len(self.iu_concept_dict[items[i]][x]))
                    if len(item_reviews) == self.args.dmax:
                        break
                item_list.append(item_reviews)
                item_concept_list.append(item_concepts)
                user_len.append(user_r_len)
                item_len.append(item_r_len)
                user_concept_len.append(user_c_len)
                item_concept_len.append(item_c_len)

        if(self.args.base_encoder!='Flat'):
            # CAML uses hierarchical inputs

            user_concept, user_concept_len = prep_hierarchical_data_list_new(user_concept_list, user_concept_len,
                                                                                              self.args.smax,
                                                                                              self.args.dmax)
            items_concept, item_concept_len = prep_hierarchical_data_list_new(item_concept_list, item_concept_len,
                                                                                              self.args.smax,
                                                                                              self.args.dmax)

            user, user_len = prep_hierarchical_data_list_new(user_list, user_len,
                                                                  self.args.smax,
                                                                  self.args.dmax)
            items, item_len = prep_hierarchical_data_list_new(item_list, item_len,
                                                                   self.args.smax,
                                                                   self.args.dmax)

        else:
            print("Preparing [Flat Mode]")
            # Flat mode are for DeepCoNN or D-ATT models
            user, user_len = prep_flat_data_list(user_list,
                                                self.args.smax,
                                                self.args.dmax,
                                                add_delimiter=2
                                                )
            items, item_len = prep_flat_data_list(item_list,
                                                self.args.smax,
                                                self.args.dmax,
                                                add_delimiter=2)


        output = [user, user_len, items, item_len]
        self.mdl.register_index_map(0, 'q1_inputs')
        self.mdl.register_index_map(1, 'q1_len')
        self.mdl.register_index_map(2, 'q2_inputs')
        self.mdl.register_index_map(3, 'q2_len')

        output.append(user_concept)
        output.append(user_concept_len)
        output.append(items_concept)
        output.append(item_concept_len)
        self.mdl.register_index_map(4, 'c1_inputs')
        self.mdl.register_index_map(5, 'c1_len')
        self.mdl.register_index_map(6, 'c2_inputs')
        self.mdl.register_index_map(7, 'c2_len')

        idx = 7
        if self.args.implicit == 1:
            output.append(user_idx)
            output.append(item_idx)
            idx += 1
            self.mdl.register_index_map(idx, 'user_id')
            idx += 1
            self.mdl.register_index_map(idx, 'item_id')

        gen_outputs = [x[3] for x in data]
        gen_len = [x[4] for x in data]

        output.append(gen_outputs)
        output.append(gen_len)
        idx += 1
        self.mdl.register_index_map(idx, 'gen_outputs')
        idx += 1
        self.mdl.register_index_map(idx, 'gen_len')

        output.append(labels)

        output = list(zip(*output))
        return output

    def load_dataset(self, data_dir, dataset_type):
        output = []
        lines_user_id = codecs.open('%s/%s_userid.txt' % (data_dir, dataset_type), 'rb', 'utf-8').readlines()
        lines_item_id = codecs.open('%s/%s_itemid.txt' % (data_dir, dataset_type), 'rb', 'utf-8').readlines()
        lines_rating = codecs.open('%s/%s_rating.txt' % (data_dir, dataset_type), 'rb', 'utf-8').readlines()
        lines_review = codecs.open('%s/%s_review_1.txt' % (data_dir, dataset_type), 'rb', 'utf-8').readlines()

        reviews = []

        concept_dict = {}
        for key in self.word_index:
            words = key.split(' ')
            l = len(words)
            for i in range(l - 1):
                concept_dict[" ".join(words[:l - i])] = 1

        for i in range(len(lines_rating)):
            output.append([int(lines_user_id[i].strip()), int(lines_item_id[i].strip()), int(lines_rating[i].strip())])

            linedata = []
            line = lines_review[i].strip()
            line = line.split('\t')

            linedata.append(self.word_index[EOS])
            l = len(line)
            pos = l
            while 1:
                pos = pos - 1
                if pos < 0:
                    break

                match_string = line[pos]
                new_pos = pos
                for j in range(pos):
                    if (" ".join(line[pos - j - 1: pos + 1]) in concept_dict):
                        if (" ".join(line[pos - j - 1: pos + 1]) in self.word_index):
                            match_string = " ".join(line[pos - j - 1: pos + 1])
                            new_pos = pos - j - 1
                        continue
                    else:
                        break
                if match_string in self.word_index:
                    linedata.append(self.word_index[match_string])
                else:
                    linedata.append(self.word_index[UNK])
                pos = new_pos

            linedata.append(self.word_index[SOS])
            linedata = linedata[::-1]

            reviews.append(linedata)

        return output, reviews

    def load_vocab(self, data_dir):
        lines_vocab = codecs.open('%s/vocabulary.txt' % data_dir, 'rb', 'utf-8').readlines()

        vocab = {}
        for i,word in enumerate(lines_vocab):
            vocab[word.strip()] = i + 4#4
        vocab[PAD] = 0
        vocab[UNK] = 1
        vocab[SOS] = 2
        vocab[EOS] = 3
        
        return vocab

    def load_review_data(self, data_dir, data_type):
        lines_user_id = codecs.open('%s/train_userid.txt' % data_dir, 'rb', 'utf-8').readlines()
        lines_item_id = codecs.open('%s/train_itemid.txt' % data_dir, 'rb', 'utf-8').readlines()
        lines_review = codecs.open('%s/train_%s.txt' % (data_dir, data_type), 'rb', 'utf-8').readlines()

        ui_dict = {}
        iu_dict = {}

        #reviews = []
        stop_concept = self.stop_concept

        for i in range(len(lines_review)):
            user = int(lines_user_id[i].strip())
            item = int(lines_item_id[i].strip())

            linedata = []
            line = lines_review[i].strip()
            if not (len(line) == 0):
                line = line.split('\t')
                for j in range(len(line)):
                    if line[j] in stop_concept:
                        if data_type == "concepts":
                            continue
                    if line[j] in self.word_index:
                        linedata.append(self.word_index[line[j]])
                    else:
                        linedata.append(self.word_index[UNK])
                if len(linedata)>self.args.smax:
                   linedata = linedata[:self.args.smax]

            if user not in ui_dict:
                ui_dict[user] = {}
            if item not in iu_dict:
                iu_dict[item] = {}
            ui_dict[user][item] = linedata
            iu_dict[item][user] = linedata

        length1 = [len(ui_dict[x]) for x in ui_dict]
        length2 = [len(iu_dict[x]) for x in iu_dict]
        length3 = []
        for x in ui_dict:
         length3 += [len(ui_dict[x][y]) for y in ui_dict[x]]
        show_stats('{}:user num review'.format(data_type), length1)
        show_stats('{}:item num review'.format(data_type), length2)
        show_stats('{}:review num word'.format(data_type), length3)

        return ui_dict, iu_dict#, reviews

    def _load_sets(self):
        # Load train, test and dev sets
        data_link = self.args.data_link
      
        if(self.no_text_mode==False):
            self.word_index = self.load_vocab(data_link)
            self.index_word = {k:v for v, k in self.word_index.items()}
            self.vocab = len(self.word_index)

            self.stop_concept = {}
            frequent_words = 100
            for i in range(frequent_words + 4):
                self.stop_concept[self.index_word[i]] = 1
            print("vocab={}".format(self.vocab))
            self.word2df = None


        self.train_rating_set, self.train_reviews = self.load_dataset(data_link, 'train')
        self.dev_rating_set, self.dev_reviews = self.load_dataset(data_link, 'valid')

        if(self.args.dev==0):
            self.train_rating_set += self.dev_rating_set
        self.test_rating_set, self.test_reviews = self.load_dataset(data_link, 'test')

        if(self.no_text_mode==False):

            #load_reviews
            self.ui_review_dict, self.iu_review_dict = self.load_review_data(data_link, "review")
            self.ui_concept_dict, self.iu_concept_dict = self.load_review_data(data_link, "concepts")
            self.num_users = len(self.ui_review_dict)
            self.num_items = len(self.iu_review_dict)


        self.write_to_file("Train={} Dev={} Test={}".format(
                                len(self.train_rating_set),
                                len(self.dev_rating_set),
                                len(self.test_rating_set)))

    def evaluate(self, data, bsz, epoch, name="", set_type=""):

        acc = 0
        num_batches = int(len(data) / bsz)
        all_preds = []
        raw_preds = []
        ff_feats = []
        all_qout = []
        review_losses = []
        losses = []
        review_acc = 0
        dev_user_entropies = []
        dev_item_entropies = []

        predict_op = self.mdl.predict_op
        actual_labels = [x[-1] for x in data]
        for i in tqdm(range(num_batches+1)):
            batch = batchify(data, i, bsz, max_sample=len(data))
            if(len(batch)==0):
                continue
            feed_dict = self.mdl.get_feed_dict(batch, mode='testing')

            loss, preds, gen_loss, gen_acc, word_att1, word_att2  = self.sess.run([self.mdl.cost,
                            predict_op, self.mdl.gen_loss, self.mdl.gen_acc, self.mdl.word_att1, self.mdl.word_att2], feed_dict)

            for k in range(len(batch)):
                ent_user = 0.0
                ent_item = 0.0
                for j in range(self.args.num_heads):
                    probs = np.array(word_att1[j][k], dtype=np.float32) + 1E-10
                    ent_user += np.sum(probs * np.log(probs))
                    probs = np.array(word_att2[j][k], dtype=np.float32) + 1E-10
                    ent_item += np.sum(probs * np.log(probs))

                dev_user_entropies.append(ent_user/self.args.num_heads)
                dev_item_entropies.append(ent_item/self.args.num_heads)

            if(i==0 and self.args.write_qual==1):
                a1, a2 = self.sess.run([self.mdl.att1, self.mdl.att2], feed_dict)
                afm = self.sess.run([self.mdl.afm], feed_dict)
                afm2 = self.sess.run([self.mdl.afm2], feed_dict)
                # wa1, wa2 = self.sess.run([self.mdl.att1, self.mdl.att2], feed_dict)
                save_qual_data('./review_viz/{}'.format(self.args.dataset),
                                self.args.rnn_type,
                                a1, a2, afm, afm2, batch, self.index_word,
                                args=self.args)
            all_preds += [x[0] for x in preds]
            review_acc += (gen_acc * len(batch))
            review_losses.append(gen_loss)
            losses.append(loss)

        if('SIG_MSE' in self.args.rnn_type):
            def rescale(x):
                return (x * (self.max_val - self.min_val)) + self.min_val
            all_preds = [rescale(x) for x in all_preds]
            actual_labels = [rescale(x) for x in actual_labels]

        _stat_al = [math.ceil(x) for x in actual_labels]
        _stat_pred = [math.ceil(x) for x in all_preds]
        print(Counter(_stat_pred))
        print(Counter(_stat_al))

        def clip_labels(x):
            if(x>5):
                return 5
            elif(x<1):
                return 1
            else:
                return x

        all_preds = [clip_labels(x) for x in all_preds]
        acc_preds = [round(x) for x in all_preds]
        acc = accuracy_score(actual_labels, acc_preds)
        mse = mean_squared_error(actual_labels, all_preds)
        actual_labels = [int(x) for x in actual_labels]
        all_preds = [int(x) for x in all_preds]
        f1 = f1_score(actual_labels, all_preds, average='macro')
        mae = mean_absolute_error(actual_labels, all_preds)
        self._register_eval_score(epoch, set_type, 'MSE', mse)
        self._register_eval_score(epoch, set_type, 'MAE', mae)
        self._register_eval_score(epoch, set_type, 'RMSE', mse ** 0.5)
        self._register_eval_score(epoch, set_type, 'ACC', acc)
        self._register_eval_score(epoch, set_type, 'F1', f1)
        self._register_eval_score(epoch, set_type, 'GEN_loss', np.mean(review_losses))
        self._register_eval_score(epoch, set_type, 'All_loss', np.mean(losses))
        self._register_eval_score(epoch, set_type, 'ACC', review_acc)

        self.write_to_file("[{}] word entropy of user={} | | word entropy of item={}".format(
                                set_type,
                                np.mean(dev_user_entropies),
                                np.mean(dev_item_entropies)))

        #return mse, all_preds
        return np.mean(losses), all_preds

    def infer(self):
        scores = []

        #data = self._prepare_set(self.test_rating_set, self.test_reviews)
        data = self._combine_reviews(self.test_rating_set, self.test_reviews)
        num_batches = int(len(data) / self.args.batch_size)

        mkdir_p(self.args.gen_dir)
        mkdir_p(self.args.gen_true_dir)

        self.mdl.saver.restore(self.sess, self.args.model)

        data_len = len(data)

        gen_sentences = []
        ref_sentences = []

        for i in tqdm(range(0, num_batches+1)):
            batch = batchify(data, i, self.args.batch_size,
                                max_sample=data_len)

            if(len(batch)==0):
                    continue

            batch = self._prepare_set(batch)
            feed_dict = self.mdl.get_feed_dict(batch, mode='infer')
            #train_op = self.mdl.train_op
            run_options = tf.RunOptions(timeout_in_ms=10000)

            gen_results  = self.sess.run([self.mdl.gen_results],
                                                         feed_dict)

            gen_results = gen_results[0]

            for j in range(self.args.batch_size):
                if (self.args.batch_size * i + j)< data_len:

                    f = open(self.args.gen_dir + '/gen_review.'+str(self.args.batch_size * i + j)+'.txt', 'w+')
                    #for t in xrange(args.beamsize):
                    new_sentence = []
                    for k in range(len(gen_results[j*self.args.beam_size])):
                        if (gen_results[j*self.args.beam_size][k]==self.word_index[EOS]):
                            break
                        if k!=0:
                            f.write(' ')
                        f.write(self.index_word[gen_results[j*self.args.beam_size][k]])
                        tmp = self.index_word[gen_results[j*self.args.beam_size][k]].split(' ')
                        for l in range(len(tmp)):
                            new_sentence.append(tmp[l])
                        #new_sentence.append(self.index_word[gen_results[j*self.args.beam_size][k]])
                        #f.write(' '+str(ppls[j*args.beamsize+t]))
                        #f.write(' '+str(tags[j*args.beamsize+t]))
                        #f.write(' '+str(lens[j*args.beamsize+t]))
                        #f.write('\n')
                    f.close()
                    gen_sentences.append(new_sentence)
                    #f.write('\n')

                    #write truth tips
                    #truth_tip_batch = feed_dict[model.tip_inputs]
                    new_sentence = []
                    true_review = self.test_reviews[self.args.batch_size * i + j]
                    f1 = open(self.args.gen_true_dir + '/true_review.A.'+str(self.args.batch_size * i + j)+'.txt', 'w+')
                    for k in range(len(true_review)):
                        if (true_review[k]==self.word_index[EOS]):
                            break
                        if k==0:
                            continue
                        if k!=1:
                            f1.write(' ')
                        f1.write(self.index_word[true_review[k]])
                        tmp = self.index_word[true_review[k]].split(' ')
                        for l in range(len(tmp)):
                            new_sentence.append(tmp[l])
                    #f1.write('\n')
                    f1.close()
                    ref_sentences.append([new_sentence])

        print ('Infer finished!')
        score = corpus_bleu(ref_sentences, gen_sentences)
        print ('bleu score: {}'.format(score))
    

    def train(self):
        """ Main training loop
        """
        scores = []
        best_score = -1
        best_dev = -1
        best_epoch = -1
        counter = 0
        min_loss = 1e+7
        epoch_scores = {}
        self.eval_list = []
        data = self._combine_reviews(self.train_rating_set, self.train_reviews)
        self.test_set = self._combine_reviews(self.test_rating_set, self.test_reviews)
        self.dev_set = self._combine_reviews(self.dev_rating_set, self.dev_reviews)
        #data = self._prepare_set(self.train_rating_set, self.train_reviews)
        #self.test_set = self._prepare_set(self.test_rating_set, self.test_reviews)
        #self.dev_set = self._prepare_set(self.dev_rating_set, self.dev_reviews)
        #print("Training Interactions={}".format(len(data)))
        #self.sess.run(tf.assign(self.mdl.is_train,self.mdl.true))

        self.mdl.saver.save(self.sess, '%s/model.ckpt' % (self.out_dir), global_step = 0)

        print("Training Interactions={}".format(len(data)))
        self.sess.run(tf.assign(self.mdl.is_train,self.mdl.true))
        for epoch in range(1, self.args.epochs+1):

            all_att_dict = {}
            pos_val, neg_val = [],[]
            t0 = time.clock()
            self.write_to_file("=====================================")
            losses = []
            review_losses = []
            random.shuffle(data)
            num_batches = int(len(data) / self.args.batch_size)
            norms = []
            all_acc = 0
            review_acc = 0
            user_entropies = []
            item_entropies = []
            user_review_hits = []
            item_review_hits = []

            for i in tqdm(range(0, num_batches+1)):
                batch = batchify(data, i, self.args.batch_size,
                                max_sample=len(data))

                #print (len(batch), len(batch[0]))

                if(len(batch)==0):
                    continue

                #print (1)
                batch = self._prepare_set(batch)
                feed_dict = self.mdl.get_feed_dict(batch)
                train_op = self.mdl.train_op
                run_options = tf.RunOptions(timeout_in_ms=10000)

                #print (2)

                #_, loss,gen_loss, gen_acc  = self.sess.run([train_op,
                #                                         self.mdl.cost, self.mdl.gen_loss, self.mdl.gen_acc],
                #                                         feed_dict)

                _, loss, gen_loss, gen_acc, att1, att2, word_att1, word_att2  = self.sess.run([train_op,
                                        self.mdl.cost, self.mdl.gen_loss, self.mdl.gen_acc, self.mdl.att1, self.mdl.att2, self.mdl.word_att1, self.mdl.word_att2],
                                        feed_dict)

                for k in range(len(batch)):
                    ent_user = 0.0
                    ent_item = 0.0
                    user_hit = 0.0
                    item_hit = 0.0
                    for j in range(self.args.num_heads):
                        probs = np.array(word_att1[j][k], dtype=np.float32) + 1E-10
                        ent_user += np.sum(probs * np.log(probs))
                        probs = np.array(word_att2[j][k], dtype=np.float32) + 1E-10
                        ent_item += np.sum(probs * np.log(probs))
                        
                        if self.args.data_prepare == 1:
                            if np.argmax(np.array(att1[j][k], dtype=np.float32)) == 0:
                                user_hit = 1.0
                            if np.argmax(np.array(att2[j][k], dtype=np.float32)) == 0:
                                item_hit = 1.0
                    
                    if self.args.data_prepare == 1:
                        user_review_hits.append(user_hit)
                        item_review_hits.append(item_hit)

                    user_entropies.append(ent_user/self.args.num_heads)
                    item_entropies.append(ent_item/self.args.num_heads)         

                if('TNET' in self.args.rnn_type):
                    # TransNet secondary review-loss
                    loss2 = self.sess.run([self.mdl.trans_loss], feed_dict)

                # For visualisation purposes only
                if(self.args.show_att==1):
                    a1, a2 = self.sess.run([self.mdl.att1, self.mdl.att2], feed_dict)
                    show_att(a1)
                if(self.args.show_affinity==1):
                    afm = self.sess.run([self.mdl.afm], feed_dict)
                    show_afm(afm)

                all_acc += (loss * len(batch))
                review_acc += (gen_acc * len(batch))
                if(self.args.tensorboard):
                    self.train_writer.add_summary(summary, counter)
                counter +=1

                losses.append(loss)
                review_losses.append(gen_loss)

            t1 = time.clock()
            self.write_to_file("[{}] [Epoch {}] [{}] loss={} gen_loss={} acc={} gen_acc={}".format(
                                self.args.dataset, epoch, self.model_name,
                                np.mean(losses), np.mean(review_losses), all_acc / len(data), review_acc / len(data)))

            if self.args.data_prepare == 1:
                self.write_to_file("user reviews hit = {} || item reviews hit = {}".format(
                                np.mean(user_review_hits),
                                np.mean(item_review_hits)))

            self.write_to_file("word entropy of user={} | | word entropy of item={}".format(
                                np.mean(user_entropies),
                                np.mean(item_entropies)))

            self.write_to_file("GPU={} | | d={}".format(
                                            self.args.gpu,
                                            self.args.emb_size))
            
            #if min_loss > np.mean(losses):
                #self.mdl.saver.save(sess, '%s/model_best.ckpt' % (self.mdl.out_dir))
                #min_loss = l/dev_batch_number

            if(epoch % self.args.eval==0):
                self.sess.run(tf.assign(self.mdl.is_train, self.mdl.false))
                loss, dev_preds = self.evaluate(self.dev_set,
                    self.args.batch_size, epoch, set_type='Dev')

                self.mdl.saver.save(self.sess, '%s/model.ckpt' % (self.out_dir), global_step = epoch)
             
                if min_loss > loss:
                    self.mdl.saver.save(self.sess, '%s/model_best.ckpt' % (self.out_dir))
                    min_loss = loss

                self._show_metrics(epoch, self.eval_dev,
                                    self.show_metrics,
                                        name='Dev')
                best_epoch1, cur_dev = self._select_test_by_dev(epoch,
                                                    self.eval_dev,
                                                    {},
                                                    no_test=True,
                                                    lower_is_better=True)

                _, test_preds = self.evaluate(self.test_set,
                    self.args.batch_size, epoch, set_type='Test')
                self._show_metrics(epoch, self.eval_test,
                                    self.show_metrics,
                                        name='Test')
                stop, max_e, best_epoch = self._select_test_by_dev(
                                                epoch,
                                                self.eval_dev,
                                                self.eval_test,
                                                lower_is_better=True)
                if(epoch-best_epoch>self.args.early_stop and self.args.early_stop>0):
                    print("Ended at early stop")
                    sys.exit(0)

if __name__ == '__main__':
    exp = CFExperiment(inject_params=None)
    #exp.train()
    exp.infer()
    print("End of code!")
