class Params:
    def __init__(self):
        # dbpedia-wikidata
        self.char_embedding_num = 51
        self.kb1_att_num_per_ent = 9
        self.kb2_att_num_per_ent = 14
        self.char_num_per_att = 35
        self.char_embed_size = 16
        self.type_ebed_size = 32
        self.kb1_attr_type_embedding_num = 329
        self.kb2_attr_type_embedding_num = 601

        # depedia-yago
        # self.char_embedding_num = 51
        # self.kb1_att_num_per_ent = 9
        # self.kb2_att_num_per_ent = 20
        # self.char_num_per_att = 35
        # self.char_embed_size = 16
        # self.type_ebed_size = 32
        # self.kb1_attr_type_embedding_num = 320
        # self.kb2_attr_type_embedding_num = 35

        # depedia-wiki-V1
        # self.char_embedding_num = 51
        # self.kb1_att_num_per_ent = 10
        # self.kb2_att_num_per_ent = 20
        # self.char_num_per_att = 38
        # self.char_embed_size = 32
        # self.type_ebed_size = 64
        # self.kb1_attr_type_embedding_num = 343
        # self.kb2_attr_type_embedding_num = 613

        # depedia-wiki-V2
        # self.char_embedding_num = 51
        # self.kb1_att_num_per_ent = 10
        # self.kb2_att_num_per_ent = 20
        # self.char_num_per_att = 40
        # self.char_embed_size = 32
        # self.type_ebed_size = 64
        # self.kb1_attr_type_embedding_num = 236
        # self.kb2_attr_type_embedding_num = 435

        # dbp-yago-V1
        # self.char_embedding_num = 56
        # self.kb1_att_num_per_ent = 10
        # self.kb2_att_num_per_ent = 10
        # self.char_num_per_att = 30
        # self.char_embed_size = 32
        # self.type_ebed_size = 64
        # self.kb1_attr_type_embedding_num = 308
        # self.kb2_attr_type_embedding_num = 46

        # # dbp-yago-V2
        # self.char_embedding_num = 46
        # self.kb1_att_num_per_ent = 10
        # self.kb2_att_num_per_ent = 10
        # self.char_num_per_att = 30
        # self.char_embed_size = 32
        # self.type_ebed_size = 64
        # self.kb1_attr_type_embedding_num = 308
        # self.kb2_attr_type_embedding_num = 46

        self.lr = 0.001
        self.batch = 64

        self.type_atten_size = 64
        self.value_atten_size = 32
        self.rnn_hidden_size = 64
        self.ent_top_k = [1, 5, 10, 50]
        self.nums_threads = 10

    def print(self):
        print("Parameters used in this running are as follows:")
        for item in self.__dict__.items():
            print("%s: %s" % item)
        print("-----------------------------------------------")


attr_P = Params()
attr_P.print()
