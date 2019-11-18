
import numpy as np
from random import shuffle
import os
import sys
import time
import multiprocessing
import gc
import math
import psutil
from sklearn import preprocessing

from Utils.utils_siamese import div_list
from model.attr_param import attr_P

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


def test_data_iterator(raw_data, test_batch_size, current_steps, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_att, shuf=True):

    test_data = raw_data[current_steps * test_batch_size:(current_steps + 1) * test_batch_size]
    kb1_train_att_type = np.ones([test_batch_size, kb1_att_num_per_ent]) * -1
    kb2_train_att_type = np.ones([test_batch_size, kb2_att_num_per_ent]) * -1
    kb1_train_att_char = np.ones([test_batch_size, kb1_att_num_per_ent, char_num_per_att]) * -1
    kb2_train_att_char = np.ones([test_batch_size, kb2_att_num_per_ent, char_num_per_att]) * -1
    label = np.array([1] * test_batch_size)

    real_test_idx = []
    reverse_real_idx = {}
    for item_idx, item in enumerate(test_data):
        if len(item) == 0:
            continue
        else:
            reverse_real_idx[item_idx] = len(real_test_idx)
            real_test_idx.append(item_idx)

        for p_idx, prop in enumerate(item[0]):
            if p_idx >= kb1_att_num_per_ent:
                break
            kb1_train_att_type[item_idx][p_idx] = prop[0]
            kb1_train_att_char[item_idx][p_idx] = prop[1] + ([-1]*(char_num_per_att-len(prop[1])))
        # print("2: ", len(item[1]))
        for p_idx, prop in enumerate(item[1]):
            if p_idx >= kb2_att_num_per_ent:
                break
            kb2_train_att_type[item_idx][p_idx] = prop[0]
            kb2_train_att_char[item_idx][p_idx] = prop[1] + ([-1]*(char_num_per_att-len(prop[1])))

    return kb1_train_att_type[real_test_idx], kb1_train_att_char[real_test_idx], kb2_train_att_type[real_test_idx], kb2_train_att_char[real_test_idx], label[real_test_idx], real_test_idx, reverse_real_idx


def whole_test_data(test_data, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_att, shuf=True):

    test_data_size = len(test_data)
    kb1_train_att_type = np.ones([test_data_size, kb1_att_num_per_ent]) * -1
    kb2_train_att_type = np.ones([test_data_size, kb2_att_num_per_ent]) * -1
    kb1_train_att_char = np.ones([test_data_size, kb1_att_num_per_ent, char_num_per_att]) * -1
    kb2_train_att_char = np.ones([test_data_size, kb2_att_num_per_ent, char_num_per_att]) * -1
    label = np.array([1] * test_data_size)

    for item_idx, item in enumerate(test_data):
        for p_idx, prop in enumerate(item[0]):
            if p_idx >= kb1_att_num_per_ent:
                break
            kb1_train_att_type[item_idx][p_idx] = prop[0]
            kb1_train_att_char[item_idx][p_idx] = prop[1] + ([-1]*(char_num_per_att-len(prop[1])))
        # print("2: ", len(item[1]))
        for p_idx, prop in enumerate(item[1]):
            if p_idx >= kb2_att_num_per_ent:
                break
            kb2_train_att_type[item_idx][p_idx] = prop[0]
            kb2_train_att_char[item_idx][p_idx] = prop[1] + ([-1]*(char_num_per_att-len(prop[1])))

    # print("before shuffle: ", kb1_train_att_type.shape)
    if shuf:
        shuffle_idx = list(range(test_data_size))
        shuffle(shuffle_idx)
        # print(shuffle_idx)
        kb1_train_att_type = kb1_train_att_type[shuffle_idx]
        kb1_train_att_char = kb1_train_att_char[shuffle_idx]
        kb2_train_att_type = kb2_train_att_type[shuffle_idx]
        kb2_train_att_char = kb2_train_att_char[shuffle_idx]
        label = label[shuffle_idx]
    # print("after shuffle: ", kb1_train_att_type.shape)
    return kb1_train_att_type, kb1_train_att_char, kb2_train_att_type, kb2_train_att_char, label


def cal_rank_multi_embed(frags, dic, sub_embed, embed, top_k):
    mean = 0
    mrr = 0
    num = np.array([0 for k in top_k])
    mean1 = 0
    mrr1 = 0
    num1 = np.array([0 for k in top_k])
    sim_mat = np.matmul(sub_embed, embed.T)
    prec_set = set()
    aligned_e = None
    for i in range(len(frags)):
        ref = frags[i]

        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank

        if dic is not None and dic.get(ref, -1) > -1:
            e2 = dic.get(ref)
            sim_mat[i, e2] += 1.0
            rank = (-sim_mat[i, :]).argsort()
            aligned_e = rank[0]
            assert ref in rank
            rank_index = np.where(rank == ref)[0][0]
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1
            # del rank
        else:
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1

        prec_set.add((ref, aligned_e))

    del sim_mat
    gc.collect()
    return mean, mrr, num, mean1, mrr1, num1, prec_set


def eval_alignment_multi_embed(embed1, embed2, top_k, selected_pairs, mess=""):
    def pair2dic(pairs):
        if pairs is None or len(pairs) == 0:
            return None
        dic = dict()
        for i, j in pairs:
            if i not in dic.keys():
                dic[i] = j
        assert len(dic) == len(pairs)
        return dic

    t = time.time()
    dic = pair2dic(selected_pairs)
    ref_num = embed1.shape[0]
    t_num = np.array([0 for k in top_k])
    t_mean = 0
    t_mrr = 0
    t_num1 = np.array([0 for k in top_k])
    t_mean1 = 0
    t_mrr1 = 0
    t_prec_set = set()
    frags = div_list(np.array(range(ref_num)), attr_P.nums_threads)
    pool = multiprocessing.Pool(processes=len(frags))
    reses = list()
    for frag in frags:
        reses.append(pool.apply_async(cal_rank_multi_embed, (frag, dic, embed1[frag, :], embed2, top_k)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, mean1, mrr1, num1, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += num
        t_mean1 += mean1
        t_mrr1 += mrr1
        t_num1 += num1
        t_prec_set |= prec_set

    assert len(t_prec_set) == ref_num

    acc = t_num / ref_num
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    t_mean /= ref_num
    t_mrr /= ref_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc, t_mean, t_mrr,
                                                                                 time.time() - t))
    if selected_pairs is not None and len(selected_pairs) > 0:
        acc1 = t_num1 / ref_num
        for i in range(len(acc1)):
            acc1[i] = round(acc1[i], 4)
        t_mean1 /= ref_num
        t_mrr1 /= ref_num
        print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc1, t_mean1, t_mrr1,
                                                                                     time.time() - t))
    return t_prec_set, (top_k, acc1, t_mean1, t_mrr1)

def __split_batch(kb1t, kb1c, kb2t, kb2c, label, batch_size, current_step, real_idx):
    return kb1t[current_step*batch_size:(current_step+1)*batch_size], kb1c[current_step*batch_size:(current_step+1)*batch_size], kb2t[current_step*batch_size:(current_step+1)*batch_size], kb2c[current_step*batch_size:(current_step+1)*batch_size], label[current_step*batch_size:(current_step+1)*batch_size], real_idx[current_step*batch_size:(current_step+1)*batch_size]


def test(model, test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, real_shape, real_idx, selected_pairs=None):
    batch = 5000
    t1 = time.time()
    print("test size: ", test_kb1t.shape[0], real_shape)
    step = math.ceil(test_kb1t.shape[0] / batch)
    test_ent1 = np.zeros((real_shape, attr_P.rnn_hidden_size * 2 + attr_P.type_ebed_size))
    test_ent2 = np.zeros((real_shape, attr_P.rnn_hidden_size * 2 + attr_P.type_ebed_size))
    fetches = {"kb1_ent": model.KB1_ent, "kb2_ent": model.KB2_ent}
    for s_i in range(step):

        kb1t, kb1c, kb2t, kb2c, label, idx = __split_batch(test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, batch, s_i, real_idx)
        feed_dict = {model.ent1_att: kb1t,
                     model.ent1_char: kb1c,
                     model.ent2_att: kb2t,
                     model.ent2_char: kb2c}
        ents = model.session.run(fetches=fetches, feed_dict=feed_dict)
        ent1_scal = preprocessing.normalize(ents["kb1_ent"], norm='l2')
        ent2_scal = preprocessing.normalize(ents["kb2_ent"], norm='l2')
        test_ent1[idx] += ent1_scal
        test_ent2[idx] += ent2_scal

    prec_set, prec_result = eval_alignment_multi_embed(test_ent1, test_ent2, attr_P.ent_top_k, selected_pairs, mess="ent alignment")
    m1 = psutil.virtual_memory().used
    del test_ent1, test_ent2
    gc.collect()
    print("testing ent alignment costs: {:.3f} s\n".format(time.time() - t1))
    return prec_set, prec_result