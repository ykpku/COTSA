
import time
import math
import tensorflow as tf
import tensorflow.contrib.layers as layers

from model.attr_param import attr_P


def embed_init(mat_x, mat_y, name="embedding", is_l2=False):
    print(name, "embed_init with size of (", str(mat_x) + "," + str(mat_y) + ")")
    embeddings = tf.Variable(tf.truncated_normal([mat_x, mat_y], stddev=1.0 / math.sqrt(mat_y)))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2 else embeddings


class AttrAttentionModel:
    def __init__(self, char_embedding_num, kb1_att_num_per_ent, kb2_att_num_per_ent, char_num_per_value, char_embed_size, type_ebed_size, lr, kb1_attr_type_embedding_num, kb2_attr_type_embedding_num, type_atten_size, value_atten_size):
        self.char_embed_size = char_embed_size
        self.kb1_attr_type_embedding_num = kb1_attr_type_embedding_num
        self.kb2_attr_type_embedding_num = kb2_attr_type_embedding_num
        self.char_embedding_num = char_embedding_num
        self.type_ebed_size = type_ebed_size
        self.kb1_att_num_per_ent = kb1_att_num_per_ent
        self.kb2_att_num_per_ent = kb2_att_num_per_ent
        self.char_num_per_value = char_num_per_value
        self.lr = lr
        # self.batch = batch
        self.type_atten_size = type_atten_size
        self.value_atten_size = value_atten_size

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self._generate_variables()
        self._generate_attr_model_graph()
        self.merged = tf.summary.merge_all()
        # 将训练日志写入到logs文件夹下
        self.writer = tf.summary.FileWriter('logs/', self.session.graph)
        tf.global_variables_initializer().run(session=self.session)

    def _generate_variables(self):
        with tf.variable_scope('attr' + 'type_embedding'):
            self.char_embeddings = embed_init(self.char_embedding_num + 1, self.char_embed_size, "char_embeds", True)
            self.KB1_attr_type_embeddings = embed_init(self.kb1_attr_type_embedding_num + 1, self.type_ebed_size, "kb1_attr_type_embeds", True)
            self.KB2_attr_type_embeddings = embed_init(self.kb2_attr_type_embedding_num + 1, self.type_ebed_size, "kb2_attr_type_embeds", True)
            self.char_embeddings = tf.nn.l2_normalize(self.char_embeddings, 1)
            self.KB1_attr_type_embeddings = tf.nn.l2_normalize(self.KB1_attr_type_embeddings, 1)
            self.KB2_attr_type_embeddings = tf.nn.l2_normalize(self.KB2_attr_type_embeddings, 1)

    def _generate_attr_model_graph(self):

        def compute_distance(kb1_ent, kb2_ent):
            # oula distance
            d = tf.sqrt(tf.reduce_sum(tf.square(kb1_ent - kb2_ent), 1))
            d = tf.div(d, tf.add(tf.sqrt(tf.reduce_sum(tf.square(kb1_ent), 1)), tf.sqrt(tf.reduce_sum(tf.square(kb2_ent), 1))))
            # cosine distance
            # d = tf.reduce_sum(kb1_ent * kb2_ent, 1)
            return d

        def compute_matrix_euclidean_similarity(A, B):
            SqED = tf.math.add(tf.reduce_sum(tf.square(A), 1, keepdims=True), tf.transpose(tf.reduce_sum(tf.square(B), 1, keepdims=True))) - (2 * tf.matmul(A, tf.transpose(B)))
            return 1.0 / (1.0 + tf.sqrt(tf.where(SqED < 0, x=tf.zeros_like(SqED), y=SqED)))

        def generate_loss(d, label, batch_size, margin=1.0):
            # the constrastive loss in siamese network for pair data
            tmp = tf.multiply(tf.to_float(label), tf.square(d))
            tmp2 = tf.multiply((1. - tf.to_float(label)), tf.square(tf.maximum(margin - d, 0.)))
            return tf.reduce_sum(tmp + tmp2)/2.0/tf.to_float(batch_size)

        def generate_optimizer(loss):
            opt_vars = [v for v in tf.trainable_variables()]
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=opt_vars)
            return optimizer

        self.ent1_char = tf.placeholder(tf.int32, shape=[None, self.kb1_att_num_per_ent, self.char_num_per_value])
        self.ent2_char = tf.placeholder(tf.int32, shape=[None, self.kb2_att_num_per_ent, self.char_num_per_value])

        self.ent1_att = tf.placeholder(tf.int32, shape=[None, self.kb1_att_num_per_ent])
        self.ent2_att = tf.placeholder(tf.int32, shape=[None, self.kb2_att_num_per_ent])
        self.label = tf.placeholder(tf.int32, shape=[None])

        pec1 = tf.nn.embedding_lookup(self.char_embeddings, self.ent1_char)
        pec2 = tf.nn.embedding_lookup(self.char_embeddings, self.ent2_char)
        pea1 = tf.nn.embedding_lookup(self.KB1_attr_type_embeddings, self.ent1_att)
        pea2 = tf.nn.embedding_lookup(self.KB2_attr_type_embeddings, self.ent2_att)

        with tf.variable_scope('KB1_type_embedding') as scope:
            with tf.variable_scope('attention') as scope:
                pea1_out, pea1_atten = self._att_self_attention(pea1, self.type_atten_size, scope)
        with tf.variable_scope('KB2_type_embedding') as scope:
            with tf.variable_scope('attention') as scope:
                pea2_out, pea2_atten = self._att_self_attention(pea2, self.type_atten_size, scope)
        #---------using AAAI19 ngram to get the value of attribution---------------
        batch = tf.shape(self.ent1_att)[0]

        def calculate_ngram_weight(unstacked_tensor, batch_k):
            stacked_tensor = tf.stack(unstacked_tensor, 1)
            stacked_tensor = tf.reverse(stacked_tensor, [1])
            index = tf.constant(len(unstacked_tensor))
            expected_result = tf.zeros([batch_k, self.char_embed_size])

            def condition(index, summation):
                return tf.greater(index, 0)

            def body(index, summation):
                precessed = tf.slice(stacked_tensor, [0, index - 1, 0], [-1, -1, -1])
                summand = tf.reduce_mean(precessed, 1)
                return tf.subtract(index, 1), tf.add(summation, summand)

            result = tf.while_loop(condition, body, [index, expected_result])
            return result[1]

        self.seqlen = self.char_num_per_value

        batch_1 = batch * self.kb1_att_num_per_ent
        pec1_rs = tf.reshape(pec1, [batch_1, self.char_num_per_value, self.char_embed_size])
        pec1_rs_in_lstm = tf.unstack(pec1_rs, self.seqlen, 1)
        pec1_rs_in_lstm = calculate_ngram_weight(pec1_rs_in_lstm, batch_1)
        KB1_value_outputs = tf.reshape(pec1_rs_in_lstm, [batch, self.kb1_att_num_per_ent, self.char_embed_size])

        batch_2 = batch * self.kb2_att_num_per_ent
        pec2_rs = tf.reshape(pec2, [batch_2, self.char_num_per_value, self.char_embed_size])
        pec2_rs_in_lstm = tf.unstack(pec2_rs, self.seqlen, 1)
        pec2_rs_in_lstm = calculate_ngram_weight(pec2_rs_in_lstm, batch_2)
        KB2_value_outputs = tf.reshape(pec2_rs_in_lstm, [batch, self.kb2_att_num_per_ent, self.char_embed_size])

        # --------using GRU to get the value embedding of attention--------
        # self.seqlen = self.char_num_per_value
        # batch = tf.shape(self.ent1_att)[0]
        # batch_1 = batch * self.kb1_att_num_per_ent
        # print("----------batch size:--------", batch)
        # cell1_forward = tf.contrib.rnn.GRUCell(attr_P.rnn_hidden_size)
        # cell1_backward = tf.contrib.rnn.GRUCell(attr_P.rnn_hidden_size)
        # initial_state1_fw = cell1_forward.zero_state(batch_1, dtype=tf.float32)
        # initial_state2_bw = cell1_forward.zero_state(batch_1, dtype=tf.float32)
        # print("----------original input size:--------", pec1.get_shape())
        # pec1_rs = tf.reshape(pec1, [batch_1, self.char_num_per_value, self.char_embed_size])
        # print("----------input size:--------", pec1_rs.get_shape())
        # outputs1, final_state1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell1_forward, cell_bw=cell1_backward, inputs=pec1_rs,  initial_state_fw=initial_state1_fw, initial_state_bw=initial_state2_bw, dtype=tf.float32, scope="kb1_rnn")
        # rnn_outputs1 = tf.concat(outputs1, 2)
        # print("----------original out size:----------", rnn_outputs1.get_shape(), tf.stack((tf.range(batch_1), tf.ones(batch_1, dtype=tf.int32) * (self.seqlen - 1)), axis=1).get_shape())
        # pec1_out = tf.gather_nd(rnn_outputs1, tf.stack((tf.range(batch_1), tf.ones(batch_1, dtype=tf.int32) * (self.seqlen - 1)), axis=1))
        # print("----------out size:----------", pec1_out.get_shape())
        # KB1_value_outputs = tf.reshape(pec1_out, [batch, self.kb1_att_num_per_ent, attr_P.rnn_hidden_size * 2])
        #
        # batch_2 = batch * self.kb2_att_num_per_ent
        # print("----------batch size:--------", batch_2)
        # cell2_forward = tf.contrib.rnn.GRUCell(attr_P.rnn_hidden_size)
        # cell2_backward = tf.contrib.rnn.GRUCell(attr_P.rnn_hidden_size)
        # initial_state2_fw = cell2_forward.zero_state(batch_2, dtype=tf.float32)
        # initial_state2_bw = cell2_backward.zero_state(batch_2, dtype=tf.float32)
        # pec2_rs = tf.reshape(pec2, [batch_2, self.char_num_per_value, self.char_embed_size])
        # outputs2, final_state2 = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell2_forward, cell_bw=cell2_backward, inputs=pec2_rs, initial_state_fw=initial_state2_fw, initial_state_bw=initial_state2_bw, dtype=tf.float32, scope="kb2_rnn")
        # rnn_outputs2 = tf.concat(outputs2, 2)
        # pec2_out = tf.gather_nd(rnn_outputs2, tf.stack((tf.range(batch_2), tf.ones(batch_2, dtype=tf.int32) * (self.seqlen - 1)), axis=1))
        # KB2_value_outputs = tf.reshape(pec2_out, [batch, self.kb2_att_num_per_ent, attr_P.rnn_hidden_size * 2])
        # --------------------------------------------------------------------------------

        KB1_value_vec = tf.reduce_sum(tf.multiply(KB1_value_outputs, pea1_atten), axis=1)
        KB2_value_vec = tf.reduce_sum(tf.multiply(KB2_value_outputs, pea2_atten), axis=1)

        self.KB1_ent = tf.concat([KB1_value_vec, pea1_out], axis=1)
        self.KB2_ent = tf.concat([KB2_value_vec, pea2_out], axis=1)

        self.d = compute_distance(self.KB1_ent, self.KB2_ent)
        batch_size = tf.shape(self.KB1_ent)[0]
        self.loss = generate_loss(self.d, self.label, batch_size, 0.8)
        tf.summary.scalar('constractive_loss', self.loss)
        self.optimizer = generate_optimizer(self.loss)

        # auto threshold 0.5
        self.temp_sim = tf.subtract(tf.ones_like(self.d), tf.rint(self.d), name="temp_sim")
        correct_predictions = tf.equal(self.temp_sim, tf.cast(self.label, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)

        self.test_sim = compute_matrix_euclidean_similarity(self.KB1_ent, self.KB2_ent)

    def _att_self_attention(self, inputs, attention_size, scope):
        L2_REG = 1e-4
        with tf.variable_scope(scope or 'attention') as scope:
            attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[attention_size],  regularizer=layers.l2_regularizer(scale=L2_REG), dtype=tf.float32)
            input_projection = layers.fully_connected(inputs, attention_size, activation_fn=tf.tanh, weights_regularizer=layers.l2_regularizer(scale=L2_REG))
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            weighted_projection = tf.multiply(inputs, attention_weights)
            outputs = tf.reduce_sum(weighted_projection, axis=1)
        return outputs, attention_weights






