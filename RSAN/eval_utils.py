import torch
import numpy as np
from misc.utils import tag_mapping,attn_mapping
import sys
import config
import json

opt = config.parse_opt()
rel2id = json.load(open(opt.input_rel2id, 'r'))

def eval(correct_preds, total_preds, total_gt):
    '''
    Evaluation
    :parameter
    :parameter
    :return: P,R,F1
    '''
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_gt if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return p, r, f1

def trans_index_to_entity(text, triples):
    f_triples = list()
    for triple in triples:
        h, t, r = triple
        _, h1, h2 = h
        head = text[h1:h2]
        _, t1, t2 = t
        tail = text[t1:t2]
        rs = "NONE"
        for rel in rel2id.items():
            rel1, rel2 = rel
            if rel2 == r:
                rs = rel1
                break
        f_triples.append((head, tail, rs))
    return f_triples

def evaluate(model, loader, label2id, batch_size, rel_num, prefix):
    model.eval()
    loader.reset(prefix)
    n = 0
    predictions = []
    final_attn = []
    targets = []
    metrics = {}
    correct_preds = 0.
    total_preds = 0.
    total_gt = 0.
    if prefix == 'dev':
        val_num = loader.dev_len
    else:
        val_num = loader.test_len
    while True:
        with torch.no_grad():
            texts, sents, gts, poses, chars, sen_lens, wrapped = loader.get_batch_dev_test(batch_size, prefix)
            sents = sents.cuda()
            sen_lens = sen_lens.cuda()
            mask = torch.zeros(sents.size()).cuda()
            poses = poses.cuda()
            chars = chars.cuda()
            n = n + batch_size
            for i in range(sents.size(0)):
                mask[i][:sen_lens[i]] = 1
            sents = sents.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
            poses = poses.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
            chars = chars.repeat([1, rel_num - 1, 1]).view(batch_size * (rel_num - 1), opt.max_len, -1)
            mask = mask.repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1), -1)
            sen_lens = sen_lens.unsqueeze(1).repeat([1, rel_num - 1]).view(batch_size * (rel_num - 1))
            rel = torch.arange(1, rel_num).repeat(batch_size).cuda()
            if not opt.use_char:
                chars = None
            if not opt.use_pos:
                poses = None
            predict, attention_score = model(sents, sen_lens, rel, mask, poses, chars)   # (batch * rel_num-1) * max_sen_len * label_num
            predict = torch.softmax(predict, -1)


            for i in range(predict.size(0)):
                predict[i][:sen_lens[i], -1] = -1e9
                predict[i][sen_lens[i]:, -1] = 1e9
            decode_tags = np.array(predict.max(-1)[1].data.cpu())
            current_relation = [k for k in range(1, rel_num)]
            # print(decode_tags)


            for i in range(batch_size):
                triple = tag_mapping(decode_tags[i * (rel_num - 1):(i + 1) * (rel_num - 1)], current_relation, label2id)
                #att = attn_mapping(attention_score[i * (rel_num - 1):(i + 1) * (rel_num - 1)], gts[i])
                target = gts[i]
                text = texts[i]
                predictions.append(triple)
                targets.append(target)
                # print(text)
                # print(predictions)
                # print(targets)
                f_triple = trans_index_to_entity(text, triple)
                f_target = trans_index_to_entity(text, target)
                # print(f_triple)
                # print(f_target)

                if n - batch_size + i + 1 <= val_num:
                    '''
                    print('Sentence %d:' % (n - batch_size + i + 1))
                    print('predict:')
                    print(triple)
                    print('target:')
                    print(target)
                    '''
                    correct_preds += len(set(f_triple) & set(f_target))
                    total_preds += len(set(f_triple))
                    total_gt += len(set(f_target))
                    # correct_preds += len(set(triple) & set(target))
                    # total_preds += len(set(triple))
                    # total_gt += len(set(target))

            if n >= val_num:
                for i in range(n - val_num):
                    predictions.pop()
                    targets.pop()
                p, r, f1, = eval(correct_preds, total_preds, total_gt)
                metrics['P'] = p
                metrics['R'] = r
                metrics['F1'] = f1
                print('test precision {}, recall {}, f1 {}'.format(p, r, f1))
                break
    model.train()
    return predictions, targets, None, metrics
