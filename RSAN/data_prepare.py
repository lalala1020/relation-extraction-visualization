#encoding=utf-8
import config
import json
import os
import numpy as np
import six
from six.moves import cPickle


import jieba
import jieba.posseg as psg
dic = 'data/百度/dict.txt'
jieba.load_userdict(dic)


def pickle_load(f):
    # six.PY3：即python版本是3.X
    # 打开kpl文件
    if six.PY3:
        # cPickle可以对任意类型的python对象进行序列化操作
        # cPickle.load（）：从字符串变量中载入python对象两个参数
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    # 生成pkl文件
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)



class DataPrepare(object):

    def __init__(self, opt):
        self.opt = opt
        vocab = np.load(opt.input_vocab)
        # vocab是word.npy文件
        self.word2id = {j: i for i, j in enumerate(vocab)}
        self.id2word = {i: j for i, j in enumerate(vocab)}
        # rel2id是rel2id.json文件
        self.rel2id = json.load(open(opt.input_rel2id, 'r'))
        # label2id是label2id.json文件
        self.label2id = json.load(open(opt.input_label2id, 'r'))
        # pos2id是pos2id.json文件
        self.pos2id = json.load(open(opt.input_pos2id, 'r'))
        # char2id是char2id.json文件
        self.char2id = json.load(open(opt.input_char2id, 'r'))
        # train_data是train.json,test_data是test.json,dev_data是dev.json
        self.train_data = self.read_json(opt.input_train)
        self.test_data = self.read_json(opt.input_test)
        self.dev_data = self.read_json(opt.input_dev)

    def prepare(self):
        print('loading data ...')
        train_pos_f, train_pos_l, train_neg_f, train_neg_l = self.process_train(self.train_data)
        # 将train_pos_f, train_pos_l, train_neg_f, train_neg_l的值保存为pkl文件
        with open(os.path.join(''+self.opt.root, 'train_pos_features.pkl'), 'wb') as f:
            pickle_dump(train_pos_f, f)
        with open(os.path.join(''+self.opt.root, 'train_pos_len.pkl'), 'wb') as f:
            pickle_dump(train_pos_l, f)
        with open(os.path.join('' + self.opt.root, 'train_neg_features.pkl'), 'wb') as f:
            pickle_dump(train_neg_f, f)
        with open(os.path.join('' + self.opt.root, 'train_neg_len.pkl'), 'wb') as f:
            pickle_dump(train_neg_l, f)

        print('finish')

        dev_f, dev_l = self.process_dev_test(self.dev_data)
        # 将dev_f, dev_l的内容保存到npy文件中
        np.save(os.path.join(''+self.opt.root, 'dev_features.npy'), dev_f, allow_pickle=True)
        np.save(os.path.join(''+self.opt.root, 'dev_len.npy'), dev_l, allow_pickle=True)

        test_f, test_l = self.process_dev_test(self.test_data)
        # 将test_f, test_l的内容保存到npy文件中
        np.save(os.path.join('' + self.opt.root, 'test_features.npy'), test_f, allow_pickle=True)
        np.save(os.path.join('' + self.opt.root, 'test_len.npy'), test_l, allow_pickle=True)

    # 将json文件存放在数组中
    def read_json(self, filename):
        data = []
        with open('' + filename, 'rb+') as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    def process_dev_test(self, dataset):

        features = []
        sen_len = []
        for i, data in enumerate(dataset):
            sent_text = data['sentText']
            sent_words, sent_ids, pos_ids, sent_chars, cur_len = self.process_sentence(sent_text)
            entities = data['entityMentions']
            raw_triples_ = data['relationMentions']
            # 去重
            triples_list = []
            for t in raw_triples_:
                triples_list.append((t['em1Text'], t['em2Text'], t['label']))
            triples_ = list(set(triples_list))
            triples_.sort(key=triples_list.index)

            triples = []
            for triple in triples_:
                head, tail, relation = triple
                try:
                    if triple[2] != 'None':

                        head_index = sent_text.index(head)
                        head_pos = range(head_index, (head_index + len(head)))

                        tail_index = sent_text.index(tail)
                        tail_pos = range(tail_index, (tail_index + len(tail)))

                        h_chunk = ('H', head_pos[0], head_pos[-1] + 1)
                        t_chunk = ('T', tail_pos[0], tail_pos[-1] + 1)
                        print((relation, self.rel2id[relation]))
                        triples.append((h_chunk, t_chunk, self.rel2id[relation]))
                except:
                    continue


            features.append([sent_text, sent_ids, pos_ids, sent_chars, triples])
            sen_len.append(cur_len)
            if (i + 1) * 1.0 % 10000 == 0:
                print('finish %f, %d/%d' % ((i + 1.0) / len(dataset), (i + 1), len(dataset)))

        return np.array(features), np.array(sen_len)

    def process_train(self, dataset):
        positive_features = []
        positive_lens = []
        negative_features = []
        negative_lens = []
        c = 0
        # 用enumerate()遍历dataset，返回i，data。i代表索引，data代表dataset中的数据。
        for i, data in enumerate(dataset):
            positive_feature = []
            positive_len = []
            negative_feature = []
            negative_len = []
            # sent_chars : (max_len, max_word_len)
            sent_text = data["sentText"]
            sent_words, sent_ids, pos_ids, sent_chars, cur_len = self.process_sentence(sent_text)

            entities = data['entityMentions']

            raw_triples_ = data['relationMentions']
            # 去重
            triples_list = []
            for t in raw_triples_:
                triples_list.append((t['em1Text'], t['em2Text'], t['label']))
            triples_ = list(set(triples_list))
            triples_.sort(key=triples_list.index)

            triples = []
            cur_relations_list = []
            cur_relations_list.append(0)
            for triple in triples_:
                cur_relations_list.append(self.rel2id[triple[2]])
                head, tail, relation = triple
                try:
                    if triple[2] != 'None':

                        head_index = sent_text.index(head)
                        head_pos = range(head_index, (head_index + len(head)))

                        tail_index = sent_text.index(tail)
                        tail_pos = range(tail_index, (tail_index + len(tail)))

                        h_chunk = ('H', head_pos[0], head_pos[-1] + 1)
                        t_chunk = ('T', tail_pos[0], tail_pos[-1] + 1)
                        triples.append((h_chunk, t_chunk, self.rel2id[relation]))
                except:
                    continue

            cur_relations = list(set(cur_relations_list))
            cur_relations.sort(key=cur_relations_list.index)

            if len(cur_relations) == 1 and cur_relations[0] == 0:
                continue
            c += 1
            none_label = ['O'] * cur_len + ['X'] * (self.opt.max_len - cur_len)
            all_labels = {} #['O'] * self.max_len

            for triple in triples_:
                head, tail, relation = triple
                rel_id = self.rel2id[relation]
                #cur_label = none_label.copy()
                cur_label = all_labels.get(rel_id, none_label.copy())
                if triple[2] != 'None':
                    try:
                        head_index = sent_text.index(head)
                    except:
                        print(sent_text)
                        print(head)
                    head_pos = range(head_index, (head_index + len(head)))
                    try:
                        if len(head_pos) == 1:
                            cur_label[head_pos[0]] = 'S-H'
                        elif len(head_pos) >= 2:
                            cur_label[head_pos[0]] = 'B-H'
                            cur_label[head_pos[-1]] = 'E-H'
                            for ii in range(1, len(head_pos)-1):
                                cur_label[head_pos[ii]] = 'I-H'
                    except:
                        continue
                    try:
                        tail_index = sent_text.index(tail)
                    except:
                        print(sent_text)
                        print(tail)
                    tail_pos = range(tail_index, (tail_index + len(tail)))
                    try:
                        # not overlap enntity
                        if len(tail_pos) == 1:
                            cur_label[tail_pos[0]] = 'S-T'
                        elif len(tail_pos) >= 2:
                            cur_label[tail_pos[0]] = 'B-T'
                            cur_label[tail_pos[-1]] = 'E-T'
                            for ii in range(1, len(tail_pos)-1):
                                cur_label[tail_pos[ii]] = 'I-T'

                    except:
                        continue
                    all_labels[rel_id] = cur_label
            # print(all_labels)
            for ii in all_labels.keys():
                cur_label_ids = [self.label2id[e] for e in all_labels[ii]]
                positive_feature.append([sent_ids, ii, cur_label_ids, pos_ids, sent_chars])
                 #positive_triple.append()
                positive_len.append(cur_len)

            none_label_ids = [self.label2id[e] for e in none_label]
            for r_id in range(self.opt.rel_num):
                if r_id not in cur_relations:
                    negative_feature.append([sent_ids, r_id, none_label_ids, pos_ids, sent_chars])
                    negative_len.append(cur_len)
            if (i + 1) * 1.0 % 10000 == 0:
                print('finish %f, %d/%d' % ((i + 1.0) / len(dataset), (i + 1), len(dataset)))
            positive_features.append(positive_feature)
            positive_lens.append(positive_len)
            negative_features.append(negative_feature)
            negative_lens.append(negative_len)
        print(c)
        return positive_features, positive_lens, negative_features, negative_lens


    def process_sentence(self, sent_text):

        sen_len = min(len(sent_text), self.opt.max_len)
        sent_text = sent_text[:sen_len]
        sent_pos_ = psg.lcut(sent_text)
        sent_words_ = list(sent_text)   # 分字，包括标点符号
        sent_pos = []
        for i in sent_pos_:
            str(i).replace('/', ',')
            i = tuple(i)
            sent_pos.append(i)  # 分词 带词性
        sent_words = []
        for j in sent_pos:
            sent_words.append(j[0])  # 分词

        sent_pos_ids = []
        for pos in sent_pos:
            p = self.pos2id.get(pos[1], 1)
            p_l = len(pos[0])
            for a in range(0, p_l):
                sent_pos_ids.append(p)
        sent_ids = []
        for w in sent_words:
            ww = self.word2id.get(w, 1)
            w_l = len(w)
            for a in range(0, w_l):
                sent_ids.append(ww)

        # sent_pos_ids = [self.pos2id.get(pos[1], 1) for pos in sent_pos][:sen_len]
        # sent_ids = [self.word2id.get(w, 1) for w in sent_words][:sen_len]

        sent_chars = []
        for w in sent_words:
            tokens = [self.char2id.get(token, 1) for token in list(w)]
            word_len = min(len(w), self.opt.max_word_len)
            for _ in range(self.opt.max_word_len - word_len):
                tokens.append(0)
            for o in range(0, len(w)):
                sent_chars.append(tokens[: self.opt.max_word_len])

            # sent_chars.append(tokens[: self.opt.max_word_len])

        for _ in range(sen_len, self.opt.max_len):
            sent_ids.append(0)
            sent_pos_ids.append(0)
            sent_chars.append([0] * self.opt.max_word_len)

        if len(sent_ids) != 120:
            print(sent_text)
            print(len(sent_ids))
        if len(sent_pos_ids) != 120:
            print(sent_text)
            print(len(sent_pos_ids))
        if len(sent_chars) != 120:
            print(sent_text)
            print(len(sent_chars))


        return sent_words_[:sen_len], sent_ids, sent_pos_ids, sent_chars, sen_len

opt = config.parse_opt()
Prepare = DataPrepare(opt)
Prepare.prepare()
