import sys
import time
import os

import argparse
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from model.train_funcs_transe import get_model, train_tris_k_epo, train_alignment_1epo
from model.param_transe import P
from model.train_bp_transe import bootstrapping
import Utils.utils_transe as ut

from model.model_siamese import AttrAttentionModel
from model.attr_param import attr_P
from model.train_funcs_siamese import train_k_epoch, generate_train_data, generate_near_neg, train_k_epoch_generate_neg
from model.train_bp_siamese import train_align_attr_epoch, attr_bootstrapping
from model.test_funcs_siamese import test_data_iterator, test
from Utils.utils_siamese import read_raw_data

def train(triple_folder, attr_pkl, max_iteration=50):
    ori_triples1, ori_triples2, triples1, triples2, triple_model = get_model(triple_folder)
    hits1 = None
    labeled_align = set()
    ents1, ents2 = None, None
    related_mat = None
    if P.epsilon > 0:
        trunc_ent_num = round(len(ori_triples1.ent_list) * (1 - P.epsilon))
        assert trunc_ent_num > 0
        print("trunc ent num:", trunc_ent_num)
    else:
        trunc_ent_num = 0
        assert not trunc_ent_num > 0

    print("=====get triple data and model finished====")

    train_data, test_data, neg_train_data = read_raw_data(attr_pkl)
    attr_model = AttrAttentionModel(attr_P.char_embedding_num, attr_P.kb1_att_num_per_ent, attr_P.kb2_att_num_per_ent, attr_P.char_num_per_att, attr_P.char_embed_size, attr_P.type_ebed_size, attr_P.lr, attr_P.kb1_attr_type_embedding_num, attr_P.kb2_attr_type_embedding_num, attr_P.type_atten_size, attr_P.value_atten_size)
    test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, real_idx, reverse_real_idx = test_data_iterator(test_data, test_batch_size=len(test_data), current_steps=0, kb1_att_num_per_ent=attr_P.kb1_att_num_per_ent, kb2_att_num_per_ent=attr_P.kb2_att_num_per_ent, char_num_per_att=attr_P.char_num_per_att, shuf=False)

    kb1t, kb1c, kb2t, kb2c, label = generate_train_data(train_data, attr_P.kb1_att_num_per_ent, attr_P.kb2_att_num_per_ent, attr_P.char_num_per_att)
    neg_dic = []
    print("=====get attr data and model finished====")

    for t in range(1, max_iteration + 1):
        for in_t in range(2):
            print("iteration ", t, "in iteration: ", in_t)
            train_tris_k_epo(triple_model, triples1, triples2, 5, trunc_ent_num, None, None, is_test=False)
            train_alignment_1epo(triple_model, triples1, triples2, ents1, ents2, 1)
            train_tris_k_epo(triple_model, triples1, triples2, 5, trunc_ent_num, None, None, is_test=False)
            labeled_align, ents1, ents2 = bootstrapping(triple_model, related_mat, labeled_align)
            print('get size of algnment : ', len(labeled_align))
            train_alignment_1epo(triple_model, triples1, triples2, ents1, ents2, 1)


        print("iteration ", t, "in iteration: ", 0, "attribution")
        train_k_epoch_generate_neg(attr_model, k_epoch=10, batch_size=attr_P.batch, kb1t=kb1t, kb1c=kb1c, kb2t=kb2t, kb2c=kb2c, label=label, neg_percent=1, kb1_att_num_per_ent=attr_P.kb1_att_num_per_ent, kb2_att_num_per_ent=attr_P.kb2_att_num_per_ent,char_num_per_att=attr_P.char_num_per_att, shuf=True, neg_dic=neg_dic)
        neg_dic = generate_near_neg(attr_model, kb1t, kb1c, kb2t, kb2c, label, neg_percent=1)
        train_k_epoch_generate_neg(attr_model, k_epoch=10, batch_size=attr_P.batch, kb1t=kb1t, kb1c=kb1c, kb2t=kb2t, kb2c=kb2c, label=label, neg_percent=1, kb1_att_num_per_ent=attr_P.kb1_att_num_per_ent, kb2_att_num_per_ent=attr_P.kb2_att_num_per_ent, char_num_per_att=attr_P.char_num_per_att, shuf=True, neg_dic=neg_dic)
        labeled_align, ents1, ents2 = attr_bootstrapping(attr_model, labeled_align, test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, triple_model.ref_ent1, triple_model.ref_ent2, real_idx, len(test_data), thred=0.8, top_k=10)
        print('get size of alignment : ', len(labeled_align))
        train_align_attr_epoch(attr_model, batch_size=attr_P.batch, alignment_pair=labeled_align, k_epoch=1, test_kb1t=test_kb1t, test_kb1c=test_kb1c, test_kb2t=test_kb2t, test_kb2c=test_kb2c, test_label=test_label, reverse_real_idx=reverse_real_idx)

        train_tris_k_epo(triple_model, triples1, triples2, 5, trunc_ent_num, None, None, is_test=False)
        train_alignment_1epo(triple_model, triples1, triples2, ents1, ents2, 1)
        train_tris_k_epo(triple_model, triples1, triples2, 5, trunc_ent_num, None, None, is_test=False)
        labeled_align, ents1, ents2 = bootstrapping(triple_model, related_mat, labeled_align)
        print('get size of algnment : ', len(labeled_align))

        if t % 1 == 0:
            hits1, prec_res = triple_model.test(selected_pairs=labeled_align)
            # hits1, prec_res = test(attr_model, test_kb1t, test_kb1c, test_kb2t, test_kb2c, test_label, len(test_data), real_idx, labeled_align)
            ut.prec2file(triple_folder + "/dp-wd_COTSA_sa_ngram_origin_prec_result_trunc" + str(P.epsilon), prec_res)
    ut.pair2file(triple_folder + "/dp-wd_COTSA_sa_ngram_origin_resultset_trunc" + str(P.epsilon), hits1)
    triple_model.save(triple_folder, "dp-wd_COTSA_sa_ngram_origin_model_trunc" + str(P.epsilon))

if __name__ == '__main__':
    t = time.time()
    parser = argparse.ArgumentParser()

    # dbpedia-wiki
    parser.add_argument("-tp", "--triple_path", type=str, help="the path of triple data", default="/home1/yk/KnowledgeAlignment/dataset/0_3")
    parser.add_argument("-ap", "--attr_path", type=str, help="the path of attribution data", default="/home1/yk/KnowledgeAlignment/dataset/0_3/attr/cleaned_new_filtered_att_sup_train_correct_test.pkl")

    # debpedia-yago
    # parser.add_argument("-tp", "--triple_path", type=str, help="the path of triple data",default="/home1/yk/KnowledgeAlignment/dataset/dbp_yago")
    # parser.add_argument("-ap", "--attr_path", type=str, help="the path of attribution data",default="/home1/yk/KnowledgeAlignment/dataset/dbp_yago/cleaned_new_filtered_dbp_yago_att_sup_train_correct_test.pkl")

    # debpedia-yago
    # parser.add_argument("-tp", "--triple_path", type=str, help="the path of triple data", default="/home1/yk/entity_preprocess/dataset/RSNdata/dbp_yg_15k_V2/mapping/0_3/")
    # parser.add_argument("-ap", "--attr_path", type=str, help="the path of attribution data", default="/home1/yk/entity_preprocess/dataset/db-yg_V2/attrs/cleaned_new_filtered_dbp_yago_att_sup_train_correct_test.pkl")

    # dbpedia-yago-V1
    # parser.add_argument("-tp", "--triple_path", type=str, help="the path of triple data", default="/home1/yk/entity_preprocess/dataset/RSNdata/dbp_yg_15k_V1/mapping/0_3/")
    # parser.add_argument("-ap", "--attr_path", type=str, help="the path of attribution data", default="/home1/yk/entity_preprocess/dataset/db-yg_V1/attrs/cleaned_new_filtered_dbp_yago_att_sup_train_correct_test.pkl")

    # dbpedia-yago-V2
    # parser.add_argument("-tp", "--triple_path", type=str, help="the path of triple data", default="/home1/yk/entity_preprocess/dataset/RSNdata/dbp_yg_15k_V2/mapping/0_3/")
    # parser.add_argument("-ap", "--attr_path", type=str, help="the path of attribution data", default="/home1/yk/entity_preprocess/dataset/db-yg_V2/attrs/cleaned_new_filtered_dbp_yago_att_sup_train_correct_test.pkl")

    # dbpedia-wiki-V1
    # parser.add_argument("-tp", "--triple_path", type=str, help="the path of triple data", default="/home1/yk/entity_preprocess/dataset/RSNdata/dbp_wd_15k_V1/mapping/0_3")
    # parser.add_argument("-ap", "--attr_path", type=str, help="the path of attribution data", default="/home1/yk/entity_preprocess/dataset/db-wiki_V1/attrs/cleaned_new_filtered_dbp_wiki_att_sup_train_correct_test.pkl")

    # dbpedia-wiki-V2
    # parser.add_argument("-tp", "--triple_path", type=str, help="the path of triple data", default="/home1/yk/entity_preprocess/dataset/RSNdata/dbp_wd_15k_V2/mapping/0_3")
    # parser.add_argument("-ap", "--attr_path", type=str, help="the path of attribution data", default="/home1/yk/entity_preprocess/dataset/db-wiki_V2/attrs/cleaned_new_filtered_dbp_wiki_att_sup_train_correct_test.pkl")

    args = parser.parse_args()

    train(args.triple_path, args.attr_path, 20)
    print("total time = {:.3f} s".format(time.time() - t))
