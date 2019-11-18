import time
import itertools
import gc
import os, sys
import math
import numpy as np
from graph_tool.all import Graph, max_cardinality_matching
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from model.attr_param import attr_P

def __generate_attrs(alignment_pair, test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, reverse_real_idx):
    new_data_size = len(alignment_pair)
    kb1_att_num_per_ent = test_kb1t.shape[1]
    kb2_att_num_per_ent = test_kb2t.shape[1]
    char_num_per_att = test_kb1c.shape[2]
    kb1_train_att_type = np.ones([new_data_size, kb1_att_num_per_ent]) * -1
    kb2_train_att_type = np.ones([new_data_size, kb2_att_num_per_ent]) * -1
    kb1_train_att_char = np.ones([new_data_size, kb1_att_num_per_ent, char_num_per_att]) * -1
    kb2_train_att_char = np.ones([new_data_size, kb2_att_num_per_ent, char_num_per_att]) * -1
    label = np.array([1] * new_data_size)
    pair_idx = 0

    for pair in list(alignment_pair):
        if pair[0] in reverse_real_idx and pair[1] in reverse_real_idx:

            ents1_item = reverse_real_idx[pair[0]]
            ents2_item = reverse_real_idx[pair[1]]

            kb1_train_att_type[pair_idx] = test_kb1t[ents1_item]
            kb1_train_att_char[pair_idx] = test_kb1c[ents1_item]

            kb2_train_att_type[pair_idx] = test_kb2t[ents2_item]
            kb2_train_att_char[pair_idx] = test_kb2c[ents2_item]
            pair_idx += 1
    return kb1_train_att_type[:pair_idx], kb1_train_att_char[:pair_idx], kb2_train_att_type[:pair_idx], kb2_train_att_char[:pair_idx], label[:pair_idx]

def __generate_batch(kb1_train_att_type, kb1_train_att_char, kb2_train_att_type, kb2_train_att_char, label, batch_size, current_steps):
    kb1t = kb1_train_att_type[current_steps * batch_size:(current_steps + 1) * batch_size]
    kb1c = kb1_train_att_char[current_steps * batch_size:(current_steps + 1) * batch_size]
    kb2t = kb2_train_att_type[current_steps * batch_size:(current_steps + 1) * batch_size]
    kb2c = kb2_train_att_char[current_steps * batch_size:(current_steps + 1) * batch_size]
    la = label[current_steps * batch_size:(current_steps + 1) * batch_size]
    return kb1t, kb1c, kb2t, kb2c, la

def __check_alignment(aligned_pairs, all_n, context="", is_cal=True):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        print("{}, Empty aligned pairs".format(context))
        return
    num = 0
    for x, y in aligned_pairs:
        if x == y:
            num += 1
    print("{}, right alignment: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))
    if is_cal:
        precision = round(num / len(aligned_pairs), 6)
        recall = round(num / all_n, 6)
        if recall > 1.0:
            recall = round(num / all_n, 6)
        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = round(2 * precision * recall / (precision + recall), 6)
        print("precision={}, recall={}, f1={}".format(precision, recall, f1))

def __generate_alignment_pairs(sim_mat, sim_th, k, all_n):
    # search_nearest_k
    if k == 0:
        return None
    potential_aligned_pairs = set()
    test_size = sim_mat.shape[0]
    for i in range(test_size):
        rank = np.argpartition(-sim_mat[i, :], k)
        pairs = [j for j in itertools.product([i], rank[0:k]) if sim_mat[j[0]][j[1]] > sim_th]
        if len(pairs)>0:
            potential_aligned_pairs |= set(pairs)

    print("checking alignment....")
    __check_alignment(potential_aligned_pairs, all_n, context="after sim filtered")
    return potential_aligned_pairs

def __mwgm_graph_tool(pairs, sim_mat):
    if not isinstance(pairs, list):
        pairs = list(pairs)
    g = Graph()
    weight_map = g.new_edge_property("float")
    nodes_dict1 = dict()
    nodes_dict2 = dict()
    edges = list()
    for x, y in pairs:
        if x not in nodes_dict1.keys():
            n1 = g.add_vertex()
            nodes_dict1[x] = n1
        if y not in nodes_dict2.keys():
            n2 = g.add_vertex()
            nodes_dict2[y] = n2
        n1 = nodes_dict1.get(x)
        n2 = nodes_dict2.get(y)
        e = g.add_edge(n1, n2)
        edges.append(e)
        weight_map[g.edge(n1, n2)] = sim_mat[x, y]
    print("graph via graph_tool", g)
    res = max_cardinality_matching(g, heuristic=True, weight=weight_map, minimize=False)
    edge_index = np.where(res.get_array() == 1)[0].tolist()
    matched_pairs = set()
    for index in edge_index:
        matched_pairs.add(pairs[index])
    return matched_pairs

def __find_potential_alignment(sim_mat, sim_th, k, total_n):
    t = time.time()
    print("generating alignment...")
    potential_aligned_pairs = __generate_alignment_pairs(sim_mat, sim_th, k, total_n)
    if potential_aligned_pairs is None or len(potential_aligned_pairs) == 0:
        return None
    t1 = time.time()
    print("mwgm....")
    selected_pairs = __mwgm_graph_tool(potential_aligned_pairs, sim_mat)
    __check_alignment(selected_pairs, total_n, context="selected_pairs")
    del potential_aligned_pairs
    print("mwgm costs time: {:.3f} s".format(time.time() - t1))
    print("selecting potential alignment costs time: {:.3f} s".format(time.time() - t))
    return selected_pairs

def __update_labeled_alignment(labeled_alignment, curr_labeled_alignment, sim_mat, all_n):
    # all_alignment = labeled_alignment | curr_labeled_alignment
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment")
    labeled_alignment_dict = dict(labeled_alignment)
    n, n1 = 0, 0
    for i, j in curr_labeled_alignment:
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n1 += 1
        if i in labeled_alignment_dict.keys():
            jj = labeled_alignment_dict.get(i)
            old_sim = sim_mat[i, jj]
            new_sim = sim_mat[i, j]
            if new_sim > old_sim:
                if jj == i and j != i:
                    n += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n, "greedy update wrongly: ", n1)
    labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    __check_alignment(labeled_alignment, all_n, context="after editing labeled alignment (<-)")
    # selected_pairs = mwgm(all_alignment, sim_mat, mwgm_igraph)
    # check_alignment(selected_pairs, context="after updating labeled alignment with mwgm")
    return labeled_alignment

def __update_labeled_alignment_label(labeled_alignment, sim_mat, all_n):
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment label")
    labeled_alignment_dict = dict()
    updated_alignment = set()
    for i, j in labeled_alignment:
        ents_j = labeled_alignment_dict.get(j, set())
        ents_j.add(i)
        labeled_alignment_dict[j] = ents_j
    for j, ents_j in labeled_alignment_dict.items():
        if len(ents_j) == 1:
            for i in ents_j:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in ents_j:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    __check_alignment(updated_alignment, all_n, context="after editing labeled alignment (->)")
    return updated_alignment

def train_align_attr_epoch(model, batch_size, alignment_pair, k_epoch, test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, reverse_real_idx):
    if alignment_pair is None or len(alignment_pair) == 0:
        return
    kb1_train_att_type, kb1_train_att_char, kb2_train_att_type, kb2_train_att_char, label = __generate_attrs(alignment_pair, test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, reverse_real_idx)
    steps = math.ceil((kb1_train_att_type.shape[0] / batch_size))
    if steps == 0:
        steps = 1
    fetches = {"summary": model.merged, "loss": model.loss, "train_op": model.optimizer, "accuracy": model.accuracy}
    for i in range(k_epoch):
        t1 = time.time()
        average_loss = 0
        for i in range(steps):
            kb1t, kb1c, kb2t, kb2c, la = __generate_batch(kb1_train_att_type, kb1_train_att_char, kb2_train_att_type, kb2_train_att_char, label, batch_size, i)
            # print("------test input of algnment------")
            # print(kb1t.shape, kb1c.shape, kb2t.shape, kb2c.shape, la.shape)
            feed_dict = {model.ent1_att: kb1t,
                         model.ent1_char: kb1c,
                         model.ent2_att: kb2t,
                         model.ent2_char: kb2c,
                         model.label: la}
            vals = model.session.run(fetches=fetches, feed_dict=feed_dict)
            num = 1000
            if (i + 1) % num == 0:
                print("triple_loss = {:.6f}, accuracy = {:.3f}".format(vals["loss"], round(vals["accuracy"], 3)))
                model.writer.add_summary(vals["summary"], (i + 1))
            average_loss += vals["loss"]
        average_loss /= steps
        print("attr_alignment_loss = {:.3f}, time = {:.3f} s".format(average_loss, time.time() - t1))

def __split_batch(kb1t, kb1c, kb2t, kb2c, label, batch_size, current_step):
    return kb1t[current_step*batch_size:(current_step+1)*batch_size], kb1c[current_step*batch_size:(current_step+1)*batch_size], kb2t[current_step*batch_size:(current_step+1)*batch_size], kb2c[current_step*batch_size:(current_step+1)*batch_size], label[current_step*batch_size:(current_step+1)*batch_size]


def __compute_matrix_euclidean_similarity(SA, B, idx, real_shape):
    steps = 5
    SA_b = math.ceil(SA.shape[0] / steps)
    sim = np.zeros((real_shape, real_shape))
    for step_i in range(steps):
        t1 = time.time()
        A = SA[step_i*SA_b:(step_i+1)*SA_b]
        real_idx = idx[step_i*SA_b:(step_i+1)*SA_b]
        sa = np.sum(np.square(A), axis=1, keepdims=True)
        sb = np.transpose(np.sum(np.square(B), axis=1, keepdims=True))
        SqED = sa + sb - (2 * np.dot(A, np.transpose(B)))
        SqED[SqED < 0] = 100000000.0
        SqED = 1.0 / (1.0 + np.sqrt(SqED))
        for i_d, item in enumerate(real_idx):
            sim[item][idx] = SqED[i_d]
            # sim[item][item] -= 0.3
        print("computing sim step ", step_i+1, "cost", int(time.time()-t1), "s")
        del SqED
        gc.collect()
    return sim

def attr_bootstrapping(model, labeled_alignment, test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, ref_ent1, ref_ent2, real_idx, real_shape, thred=0.8, top_k=10):
    batch = 5000
    print("test size: ", test_kb1t.shape[0])
    step = math.ceil(test_kb1t.shape[0] / batch)
    # last_batch = test_kb1t.shape[0] % batch
    # test_ent1 = np.zeros((test_kb1t.shape[0], attr_P.rnn_hidden_size * 2 + attr_P.type_ebed_size))
    # test_ent2 = np.zeros((test_kb1t.shape[0], attr_P.rnn_hidden_size * 2 + attr_P.type_ebed_size))
    test_ent1 = np.zeros((test_kb1t.shape[0], attr_P.char_embed_size + attr_P.type_ebed_size))
    test_ent2 = np.zeros((test_kb1t.shape[0], attr_P.char_embed_size + attr_P.type_ebed_size))
    fetches = {"kb1_ent": model.KB1_ent, "kb2_ent": model.KB2_ent}
    for s_i in range(step):
        # if s_i == (step-1) and last_batch != 0:
        #     batch = last_batch
        # else:
        #     batch = 7000
        kb1t, kb1c, kb2t, kb2c, label = __split_batch(test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, batch, s_i)
        feed_dict = {model.ent1_att: kb1t,
                     model.ent1_char: kb1c,
                     model.ent2_att: kb2t,
                     model.ent2_char: kb2c}

        ents = model.session.run(fetches=fetches, feed_dict=feed_dict)
        test_ent1[s_i * batch:(s_i + 1) * batch] += ents["kb1_ent"]
        test_ent2[s_i * batch:(s_i + 1) * batch] += ents["kb2_ent"]
        # print("step ", s_i+1, "..", ents["kb1_ent"].shape, ents["kb2_ent"].shape)
    print("compute the euclidean_similarity....")
    ref_sim_mat = __compute_matrix_euclidean_similarity(test_ent1, test_ent2, real_idx, real_shape)
    print("sim matrix shape: ", ref_sim_mat.shape[0], ref_sim_mat.shape[1])
    print(ref_sim_mat[:10])
    test_size = ref_sim_mat.shape[0]
    curr_labeled_alignment = __find_potential_alignment(ref_sim_mat, thred, top_k, test_size)
    if curr_labeled_alignment is not None:
        labeled_alignment = __update_labeled_alignment(labeled_alignment, curr_labeled_alignment, ref_sim_mat, test_size)
        labeled_alignment = __update_labeled_alignment_label(labeled_alignment, ref_sim_mat, test_size)
        del curr_labeled_alignment
    # labeled_alignment = curr_labeled_alignment
    if labeled_alignment is not None:
        ents1 = [ref_ent1[pair[0]] for pair in labeled_alignment]
        ents2 = [ref_ent2[pair[1]] for pair in labeled_alignment]
    else:
        ents1, ents2 = None, None
    del ref_sim_mat
    gc.collect()
    print(ents1[:10], ents2[:10])
    return labeled_alignment, ents1, ents2

