import pickle as pkl
from collections import defaultdict
import numpy as np
from random import shuffle
import os, gc
import sys
import time
import math
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from model.attr_param import attr_P
from Utils.utils_siamese import read_pairs
att_folder = "E:/githubWorkSpace/KnowledgeAlignment/dataset/0_3/attr/"
data_folder = "E:/githubWorkSpace/KnowledgeAlignment/dataset/0_3/"
experiment_folder = "E:/githubWorkSpace/KnowledgeAlignment/dataset/"


def read_attr_triples(file_path, max_length):
    triples = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            params = line.strip().split('\t')
            if len(params) != 3:
                print("wrong line" + line)
            # assert len(params) == 3
            h = params[0]
            a = int(params[1])
            v = params[2][:max_length]
            triples.append([h, a, v])
    return triples


def get_char_dict(trip1, trip2, w=False):
    char_dict = defaultdict(int)
    for t in trip1:
        for c in t[2]:
            char_dict[c] = char_dict[c] + 1
    for t in trip2:
        for c in t[2]:
            char_dict[c] = char_dict[c] + 1
    print(char_dict)
    print(len(char_dict))
    l = sorted(char_dict.items(), key=lambda d: d[1], reverse=True)
    print(l)
    cd = {}
    for i in range(len(l)):
        cd[l[i][0]] = i
    print(cd)
    if w:
        with open(att_folder + "char_ids", 'w', encoding="utf-8")as f:
            for i in range(len(l)):
                f.write(l[i][0]+"\t"+str(i)+"\n")
    return cd


def _change_char_ids(trip, cd):
    new_trip = []
    for item in trip:
        new_c = []
        for c in item[2]:
            new_c.append(cd[c])
        new_trip.append([item[0], item[1], new_c])
    return new_trip


def _change_kb2_att_ids():
    triples = []
    with open(att_folder + "attr_ids_2", 'r', encoding='utf8') as f:
        with open(att_folder + "new_att_ids_2", 'w', encoding="utf-8")as wf:
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 2:
                    print("wrong line" + line)
                h = params[0]
                a = int(params[1])-352
                wf.write(h+"\t"+str(a)+"\n")

    with open(att_folder + "attr_type_2", 'r', encoding='utf8') as f:
        with open(att_folder + "new_att_type_2", 'w', encoding="utf-8")as wf:
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 2:
                    print("wrong line" + line)
                h = int(params[0])-352
                a = params[1]
                wf.write(str(h)+"\t"+a+"\n")

    with open(att_folder + "filtered_attr_triples_2", 'r', encoding='utf8') as f:
        with open(att_folder + "new_filtered_att_triples_2", 'w', encoding="utf-8")as wf:
            for line in f.readlines():
                params = line.strip().split('\t')
                if len(params) != 3:
                    print("wrong line" + line)
                h = params[0]
                a = int(params[1])-352
                v = params[2]
                wf.write(h+"\t"+str(a)+"\t"+v+"\n")


def read_attr_input(attr_folder, trip1='filtered_attr_triples_1', trip2='new_filtered_att_triples_2'):
    triples_list1 = read_attr_triples(attr_folder + trip1, 35)
    triples_list2 = read_attr_triples(attr_folder + trip2, 35)
    cd = get_char_dict(triples_list1, triples_list2)
    # the number of char is 51

    triples_list1 = _change_char_ids(triples_list1, cd)
    triples_list2 = _change_char_ids(triples_list2, cd)

    # ent_att_count1, ent_att_count2 = defaultdict(int), defaultdict(int)
    # count_fre1, count_fre2 = defaultdict(int), defaultdict(int)
    # for item in triples_list1:
    #     ent_att_count1[item[0]] += 1
    # for item in triples_list2:
    #     ent_att_count2[item[0]] += 1
    # for item in ent_att_count1.values():
    #     count_fre1[item] += 1
    # for item in ent_att_count2.values():
    #     count_fre2[item] += 1
    # _display_dict_dis(count_fre1)
    # KB1 att number per ent is 9
    # 长尾效应，有一些实体有大量属性，后续可以考虑预处理一下，将这些ent中重复的属性删除
    # _display_dict_dis(count_fre2)
    # KB2 att number per ent is 54
    # 长尾效应，有一些实体有大量属性，后续可以考虑预处理一下，将这些ent中重复的属性删除

    return triples_list1, triples_list2


def _get_tdata(att_folder):
    trip1, trip2 = read_attr_input(att_folder, trip1="clean_new_filtered_att_triples_1", trip2="clean_new_filtered_att_triples_2")
    t1_dict, t2_dict = {}, {}
    for item in trip1:
        if item[0] in t1_dict:
            t1_dict[item[0]].append((item[1], item[2]))
        else:
            t1_dict[item[0]] = [(item[1], item[2])]
    for item in trip2:
        if item[0] in t2_dict:
            t2_dict[item[0]].append((item[1], item[2]))
        else:
            t2_dict[item[0]] = [(item[1], item[2])]
    ref_ents = read_pairs(att_folder + "ref_ent_ids")
    sup_ents = read_pairs(att_folder + "sup_ent_ids")
    print(len(ref_ents))
    print(len(sup_ents))
    print(len(t1_dict), len(t2_dict))

    train_data, test_data = [], []
    neg_train_data = []
    for item_idx, item in enumerate(sup_ents):
        if item[0] in t1_dict and item[1] in t2_dict:
            train_data.append([t1_dict[item[0]], t2_dict[item[1]]])
            neg_kb2_item_ent = (item_idx + 1) % len(sup_ents)
            while(sup_ents[neg_kb2_item_ent][1] not in t2_dict):
                neg_kb2_item_ent = (neg_kb2_item_ent + 1) % len(sup_ents)
            neg_train_data.append([t1_dict[item[0]], t2_dict[sup_ents[neg_kb2_item_ent][1]]])
        else:
            print(item[0], item[1])
    for item in ref_ents:
        if item[0] in t1_dict and item[1] in t2_dict:
            test_data.append([t1_dict[item[0]], t2_dict[item[1]]])
        else:
            test_data.append([])
    print(len(train_data))
    print(len(test_data))
    print(train_data[:3])

    with open(att_folder + "cleaned_new_filtered_dbp_yago_att_sup_train_correct_test.pkl", "wb") as f:
        pkl.dump(train_data, f)
        pkl.dump(test_data, f)
        pkl.dump(neg_train_data, f)
    return train_data, test_data, neg_train_data


# step_sum = len(relate_data) // int(batch_size * tf_radio)
def data_iterator(raw_data, neg_raw_data, pos_batch_size, neg_batch_size, current_steps, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_att, shuf=True):
    # tf_radio == 0.5 假设pos==neg
    pos_data = raw_data[current_steps * pos_batch_size:(current_steps + 1) * pos_batch_size]
    neg_data = neg_raw_data[current_steps * neg_batch_size:(current_steps + 1) * neg_batch_size]
    kb1_train_att_type = np.ones([pos_batch_size+neg_batch_size, kb1_att_num_per_ent]) * -1
    kb2_train_att_type = np.ones([pos_batch_size+neg_batch_size, kb2_att_num_per_ent]) * -1
    kb1_train_att_char = np.ones([pos_batch_size+neg_batch_size, kb1_att_num_per_ent, char_num_per_att]) * -1
    kb2_train_att_char = np.ones([pos_batch_size+neg_batch_size, kb2_att_num_per_ent, char_num_per_att]) * -1
    label = np.array([1] * pos_batch_size + [0]*neg_batch_size)

    for item_idx, item in enumerate(pos_data):
        for p_idx, prop in enumerate(item[0]):
            # print("--------------")
            # print(prop[0])
            # print(prop[1])
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

    for item_idx, item in enumerate(neg_data):
        for p_idx, prop in enumerate(item[0]):
            if p_idx >= kb1_att_num_per_ent:
                break
            kb1_train_att_type[item_idx+pos_batch_size][p_idx] = prop[0]
            kb1_train_att_char[item_idx+pos_batch_size][p_idx] = prop[1] + ([-1] * (char_num_per_att - len(prop[1])))
        for p_idx, prop in enumerate(item[1]):
            if p_idx >= kb2_att_num_per_ent:
                break
            kb2_train_att_type[item_idx+pos_batch_size][p_idx] = prop[0]
            kb2_train_att_char[item_idx+pos_batch_size][p_idx] = prop[1] + ([-1] * (char_num_per_att - len(prop[1])))
    # print("before shuffle: ", kb1_train_att_type.shape)
    if shuf:
        shuffle_idx = list(range(pos_batch_size+neg_batch_size))
        shuffle(shuffle_idx)
        # print(shuffle_idx)
        kb1_train_att_type = kb1_train_att_type[shuffle_idx]
        kb1_train_att_char = kb1_train_att_char[shuffle_idx]
        kb2_train_att_type = kb2_train_att_type[shuffle_idx]
        kb2_train_att_char = kb2_train_att_char[shuffle_idx]
        label = label[shuffle_idx]
    # print("after shuffle: ", kb1_train_att_type.shape)
    return kb1_train_att_type, kb1_train_att_char, kb2_train_att_type, kb2_train_att_char, label


def train_k_epoch(model, k_epoch, batch_size, train_data, neg_train_data, kb1_att_num_per_ent, kb2_att_num_per_ent,char_num_per_att, shuf):
    pos_batch_size = batch_size // 2
    neg_batch_size = batch_size // 2
    train_step_num = len(train_data) // pos_batch_size
    fetches = {"summary": model.merged, "loss": model.loss, "train_op": model.optimizer, "accuracy": model.accuracy}
    for i in range(k_epoch):
        ep_loss = 0
        start = time.time()
        for step in range(train_step_num):
            kb1t, kb1c, kb2t, kb2c, label = data_iterator(train_data, neg_train_data, pos_batch_size, neg_batch_size, step, kb1_att_num_per_ent, kb2_att_num_per_ent,char_num_per_att, shuf)
            # print(kb1t.shape, kb1c.shape, kb2t.shape, kb2c.shape, label.shape)
            feed_dict = {model.ent1_att: kb1t,
                         model.ent1_char: kb1c,
                         model.ent2_att: kb2t,
                         model.ent2_char: kb2c,
                         model.label: label}
            vals = model.session.run(fetches=fetches, feed_dict=feed_dict)
            ep_loss += vals["loss"]
            # num = train_step_num // 5
            # if (step+1) % num == 0:
            #     print("triple_loss = {:.6f}, accuracy = {:.3f}".format(vals["loss"], round(vals["accuracy"], 3)))
            #     model.writer.add_summary(vals["summary"], (step+1))
        ep_loss /= train_step_num
        end = time.time()
        print(str(i + 1)+" epoch: ", "Average triple_loss = {:.6f}, time = {:.3f} s".format(ep_loss, round(end - start, 2)))
    model.writer.close()


def train_k_epoch_generate_neg(model, k_epoch, batch_size, kb1t, kb1c, kb2t, kb2c, label, neg_percent, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_att, shuf, neg_dic):
    pos_batch_size = batch_size // (1 + neg_percent)
    neg_batch_size = pos_batch_size * neg_percent
    train_step_num = math.ceil(kb1t.shape[0] / pos_batch_size)
    last_batch = kb1t.shape[0] % pos_batch_size
    fetches = {"summary": model.merged, "loss": model.loss, "train_op": model.optimizer, "accuracy": model.accuracy}
    for i in range(k_epoch):
        ep_loss = 0
        start = time.time()
        for step in range(train_step_num):
            if step == (train_step_num-1) and last_batch != 0:
                pos_batch_size = last_batch
            else:
                pos_batch_size = batch_size // (1 + neg_percent)
            k1t, k1c, k2t, k2c, la = __generate_pos_neg_data(kb1t, kb1c, kb2t, kb2c, label, pos_batch_size, step, neg_dic, neg_percent, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_att, shuf)

            # print(kb1t.shape, kb1c.shape, kb2t.shape, kb2c.shape, label.shape)
            feed_dict = {model.ent1_att: k1t,
                         model.ent1_char: k1c,
                         model.ent2_att: k2t,
                         model.ent2_char: k2c,
                         model.label: la}
            vals = model.session.run(fetches=fetches, feed_dict=feed_dict)
            ep_loss += vals["loss"]
            # num = train_step_num // 5
            # if (step+1) % num == 0:
            #     print("triple_loss = {:.6f}, accuracy = {:.3f}".format(vals["loss"], round(vals["accuracy"], 3)))
            #     model.writer.add_summary(vals["summary"], (step+1))
        ep_loss /= train_step_num
        end = time.time()
        print(str(i + 1)+" epoch: ", "Average triple_loss = {:.6f}, time = {:.3f} s".format(ep_loss, round(end - start, 2)))
    model.writer.close()


def generate_train_data(train_data, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_att):
    pos_size = len(train_data)
    kb1_train_att_type = np.ones([pos_size, kb1_att_num_per_ent]) * -1
    kb2_train_att_type = np.ones([pos_size, kb2_att_num_per_ent]) * -1
    kb1_train_att_char = np.ones([pos_size, kb1_att_num_per_ent, char_num_per_att]) * -1
    kb2_train_att_char = np.ones([pos_size, kb2_att_num_per_ent, char_num_per_att]) * -1
    label = np.array([1] * pos_size)

    for item_idx, item in enumerate(train_data):
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
    return kb1_train_att_type, kb1_train_att_char, kb2_train_att_type, kb2_train_att_char, label


def __generate_pos_neg_data(kb1t, kb1c, kb2t, kb2c, ll, pos_batch_size, current_step, neg_dic, neg_percent, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_att, shuf):
    neg_per_pos = int(neg_percent)
    neg_size = pos_batch_size * neg_per_pos
    kb1_train_att_type = np.ones([pos_batch_size + neg_size, kb1_att_num_per_ent]) * -1
    kb2_train_att_type = np.ones([pos_batch_size + neg_size, kb2_att_num_per_ent]) * -1
    kb1_train_att_char = np.ones([pos_batch_size + neg_size, kb1_att_num_per_ent, char_num_per_att]) * -1
    kb2_train_att_char = np.ones([pos_batch_size + neg_size, kb2_att_num_per_ent, char_num_per_att]) * -1
    label = np.array([1] * pos_batch_size + [0] * neg_size)

    kb1_train_att_type[:pos_batch_size] = kb1t[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]
    kb1_train_att_char[:pos_batch_size] = kb1c[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]
    kb2_train_att_type[:pos_batch_size] = kb2t[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]
    kb2_train_att_char[:pos_batch_size] = kb2c[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]

    if len(neg_dic) == 0:
        import random
        plus_neg = random.sample(list(range(1, kb1t.shape[0]-1)), neg_per_pos)
        for neg_iter, neg_item in enumerate(plus_neg):
            kb1_train_att_type[pos_batch_size * (neg_iter+1): pos_batch_size * (neg_iter+2)] = kb1t[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]
            kb1_train_att_char[pos_batch_size * (neg_iter+1): pos_batch_size * (neg_iter+2)] = kb1c[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]
            neg_ids = (np.array(list(range(current_step * pos_batch_size, (current_step + 1) * pos_batch_size))) + neg_item) % (kb1t.shape[0])
            kb2_train_att_type[pos_batch_size * (neg_iter+1): pos_batch_size * (neg_iter+2)] = kb2t[neg_ids]
            kb2_train_att_char[pos_batch_size * (neg_iter+1): pos_batch_size * (neg_iter+2)] = kb2c[neg_ids]
    else:
        # print(kb1_train_att_type.shape, pos_batch_size * 1, pos_batch_size * 2)
        # print(kb1t.shape, current_step * pos_batch_size, (current_step + 1) * pos_batch_size)
        for neg_iter, neg_item in enumerate(neg_dic):
            # print("tenst;;;", neg_iter, neg_item)
            kb1_train_att_type[pos_batch_size * (neg_iter + 1): pos_batch_size * (neg_iter + 2)] = kb1t[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]
            kb1_train_att_char[pos_batch_size * (neg_iter + 1): pos_batch_size * (neg_iter + 2)] = kb1c[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]
            neg_ids = np.array(neg_item)
            # print(neg_ids.shape)
            kb2_train_att_type[pos_batch_size * (neg_iter + 1): pos_batch_size * (neg_iter + 2)] = kb2t[neg_ids[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]]
            kb2_train_att_char[pos_batch_size * (neg_iter + 1): pos_batch_size * (neg_iter + 2)] = kb2c[neg_ids[current_step * pos_batch_size:(current_step + 1) * pos_batch_size]]

    if shuf:
        shuffle_idx = list(range(pos_batch_size + neg_size))
        shuffle(shuffle_idx)
        # print(shuffle_idx)
        kb1_train_att_type = kb1_train_att_type[shuffle_idx]
        kb1_train_att_char = kb1_train_att_char[shuffle_idx]
        kb2_train_att_type = kb2_train_att_type[shuffle_idx]
        kb2_train_att_char = kb2_train_att_char[shuffle_idx]
        label = label[shuffle_idx]
    return kb1_train_att_type, kb1_train_att_char, kb2_train_att_type, kb2_train_att_char, label


def __compute_matrix_euclidean_similarity(A, B):
    steps = 3
    A_b = math.ceil(A.shape[0] / steps)
    real_shape = A.shape[0]
    sim = np.zeros((real_shape, real_shape))
    for step_i in range(steps):
        t1 = time.time()
        SA = A[step_i * A_b:(step_i+1) * A_b]
        sa = np.sum(np.square(SA), axis=1, keepdims=True)
        sb = np.transpose(np.sum(np.square(B), axis=1, keepdims=True))
        SqED = sa + sb - (2 * np.dot(SA, np.transpose(B)))
        SqED[SqED < 0] = 10000.0
        SqED = 1.0 / (1.0 + np.sqrt(SqED))
        sim[step_i * A_b : (step_i+1) * A_b] = SqED
        print("computing sim step ", step_i+1, "cost", int(time.time()-t1), "s")
    del SqED
    gc.collect()
    return sim


def generate_near_neg(model, kb1t, kb1c, kb2t, kb2c, label, neg_percent):
    batch = 256
    train_size = kb1t.shape[0]
    print("train size:", train_size)
    step = math.ceil(train_size / batch)
    # train_ent1 = np.zeros((train_size, attr_P.rnn_hidden_size * 2 + attr_P.type_ebed_size))
    # train_ent2 = np.zeros((train_size, attr_P.rnn_hidden_size * 2 + attr_P.type_ebed_size))
    train_ent1 = np.zeros((train_size, attr_P.char_embed_size + attr_P.type_ebed_size))
    train_ent2 = np.zeros((train_size, attr_P.char_embed_size + attr_P.type_ebed_size))

    fetches = {"kb1_ent": model.KB1_ent, "kb2_ent": model.KB2_ent}
    start = time.time()
    neg_per_pos = int(neg_percent)
    for s_i in range(step):
        k1t = kb1t[s_i * batch:(s_i + 1) * batch]
        k1c = kb1c[s_i * batch:(s_i + 1) * batch]
        k2t = kb2t[s_i * batch:(s_i + 1) * batch]
        k2c = kb2c[s_i * batch:(s_i + 1) * batch]

        # print(kb1t.shape, kb1c.shape, kb2t.shape, kb2c.shape, label.shape)
        feed_dict = {model.ent1_att: k1t,
                     model.ent1_char: k1c,
                     model.ent2_att: k2t,
                     model.ent2_char: k2c}
        ents = model.session.run(fetches=fetches, feed_dict=feed_dict)
        train_ent1[s_i * batch:(s_i+1) * batch] += ents["kb1_ent"]
        train_ent2[s_i * batch:(s_i+1) * batch] += ents["kb2_ent"]
    print("compute training data the euclidean_similarity....")
    vals = __compute_matrix_euclidean_similarity(train_ent1, train_ent2)
    print("sim matrix shape: ", vals.shape[0], vals.shape[1])
    print(vals[:5][:5])
    neg_ids = []
    for i in range(vals.shape[0]):
        sort_index = np.argpartition(-vals[i, :], neg_per_pos + 1)
        neg_ids.append([i_t for i_t in sort_index[0:neg_per_pos + 1] if i_t != i])
        neg_ids[-1] = neg_ids[-1][:neg_per_pos]

    end = time.time()
    # print("generating the negative train shape: ", len(neg_ids), len(neg_ids[0]), neg_ids[0][0])
    neg_ids = np.array(neg_ids).reshape((1, -1))
    # print(neg_ids[0][0])

    ### test neg ids ####
    with open("/home1/yk/KnowledgeAlignment/dataset/0_3/generate_neg_ids.csv", 'w', encoding="utf-8") as f:
        for item in neg_ids:
            print(item)
            f.write(str(item) + "\n")

    print("generating the negative train shape: ", neg_ids.shape, int(end-start), "s")

    return neg_ids


# read_attr_input(att_folder)
# _change_kb2_att_ids()
# _get_tdata(att_folder="E:/githubWorkSpace/KnowledgeAlignmentDataset/dbp_yago/")
# test_clean_attr2_rules()
# test_clean_attr1_rules()
# check_att_num_length(att_folder="E:/githubWorkSpace/KnowledgeAlignmentDataset/dbp_yago/", file_name="clean_new_filtered_att_triples_2")
# dbpedia data
# att1 : attr_num==8-9 attr_value_length==30-35
# att2 : attr_num==14 attr_value_length==35

# yago data
# att1 : attr_num==9 attr_value_length==35-40
# att2 : attr_num==20 attr_value_length==30-35

