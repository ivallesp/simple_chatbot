from src.data_loaders import load_cornell_dialogs
from src.general_utilities import flatten
from src.text_tools import pad, remove_substrings
from src.general_utilities import batching, get_batcher
from src.tensorflow_utilities import start_tensorflow_session, get_summary_writer, TensorFlowSaver
from src.common_paths import get_tensorboard_logs_path, get_models_path, get_data_path


import os
import sys

from tqdm import tqdm
import numpy as np
import tensorflow as tf

BATCH_SIZE=256
project_id="chatbot"
version_id="v06"


# Data processing
dialogs = load_cornell_dialogs()
charset = list(set("".join(list(map(lambda x:x[0]+x[1], dialogs)))))
charset_size = len(charset)+2
max_length = max(map(len, flatten(dialogs)))
go_symbol = len(charset)
unk_symbol = len(charset)+1
character_to_code = dict(list(zip(charset+["$GO$", "$UNK$"], range(len(charset)+2))))
code_to_character = {k:v for (v,k) in character_to_code.items()}

process_dialog = lambda dialog: [tuple(pad(x = [character_to_code[ch] for ch in sentence], 
                                           max_length=max_length, 
                                           mode="right", 
                                           symbol=unk_symbol)) for sentence in dialog]

dialogs_codes = list(map(process_dialog, dialogs))





# In[9]:


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ChatBot:
    def __init__(self, max_length=None, charset_cardinality=None, mode="train", name="cb"):
        self.name = name
        self.mode = mode
        self.max_length = max_length
        self.charset_cardinality = charset_cardinality
        self.embedding_size = 256
        self.rnn_units = 2048
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.define_computation_graph()
        

        # Aliases
        self.ph = self.placeholders
        self.op = self.optimizers
        self.summ = self.summaries

    def define_computation_graph(self):
        # Reset graph
        tf.reset_default_graph()
        self.placeholders = NameSpacer(**self.define_placeholders())
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_placeholders(self):
        with tf.variable_scope("Placeholders"):
            question = tf.placeholder(tf.int32, shape=(None, self.max_length, 1), 
                                            name="question_ph")
            answer = tf.placeholder(tf.int32, shape=(None, self.max_length+1, 1),
                                   name="answer")
            
        return {"question": question, "answer": answer}

    def define_core_model(self):
        with tf.variable_scope("Core_Model"):
            char_embedding_matrix = tf.get_variable(name="char_embedding_matrix", 
                                                    shape=[self.charset_cardinality, self.embedding_size],
                                                    dtype=tf.float32)
            embedding_question = tf.nn.embedding_lookup(char_embedding_matrix, self.placeholders.question)
            embedding_question = embedding_question[:,:,0,:]
            
            embedding_question = tf.contrib.layers.layer_norm(embedding_question)
            
            encoder_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_units)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, embedding_question, dtype=tf.float32)
            
            encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=tf.contrib.layers.layer_norm(encoder_state.c),
                                                          h=tf.contrib.layers.layer_norm(encoder_state.h))
            decoder_input = tf.one_hot(self.placeholders.answer[:,:,0], self.charset_cardinality)

            seq_lengths=tf.cast(tf.ones(tf.shape(self.placeholders.answer)[0])*int(self.max_length), tf.int32)
            
                        
 
            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_units)
            proj_layer =  tf.layers.Dense(self.charset_cardinality)
            if self.mode=="train":
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input, 
                                                           sequence_length=seq_lengths,
                                                           name="decoder_helper")
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, 
                                                         proj_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                
            elif self.mode=="greedy":
                start_sym=tf.cast(tf.ones(tf.shape(self.placeholders.answer)[0])*int(go_symbol), tf.int32)
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(lambda x: tf.one_hot(x, self.charset_cardinality), 
                                                                            start_sym, unk_symbol)
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, 
                                                          proj_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
                
            elif self.mode=="beam":
                beam_width = 1000
                tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                embedding_function = lambda x: tf.one_hot(x, self.charset_cardinality)
                start_sym=tf.cast(tf.ones(tf.shape(self.placeholders.answer)[0])*int(go_symbol), tf.int32)
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_function, 
                                                                            start_sym, unk_symbol)

                decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, 
                                                               embedding=embedding_function, 
                                                               start_tokens=start_sym, 
                                                               end_token=unk_symbol, 
                                                               initial_state=tiled_encoder_state, 
                                                               beam_width=beam_width, 
                                                               output_layer=proj_layer)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
        
        return {"outputs": outputs}

    def define_losses(self):
        with tf.variable_scope("Losses"):
            if self.mode == "train":
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.placeholders.answer[:,1:, 0],
                                                                      logits=self.core_model.outputs.rnn_output)
            else: 
                loss = None
        return {"loss": loss}

    def define_optimizers(self):
        with tf.variable_scope("Optimizers"):
            if self.mode=="train":
                optimizer=self.optimizer.minimize(self.losses.loss)
            else:
                optimizer=None
        return {"op": optimizer}

    def define_summaries(self):
        with tf.variable_scope("Summaries"):
            if self.mode=="train":
                train_scalar_probes = {"loss": tf.squeeze(self.losses.loss)}

                train_performance = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                           for k, v in train_scalar_probes.items()]

                return {"train_performance": tf.summary.merge(train_performance)}
            else:
                return {}


# Train model
model = ChatBot(max_length=max_length, charset_cardinality=charset_size, mode="train")

sess=start_tensorflow_session()
saver = TensorFlowSaver(os.path.join(get_models_path(), "{}_{}".format(project_id, version_id)), max_to_keep=10)
i=0
sw = get_summary_writer(sess, logs_path=get_tensorboard_logs_path(),
                       project_id=project_id, version_id=version_id, remove_if_exists=True)



sess.run(tf.global_variables_initializer())


while True:
    batcher=get_batcher(dialogs_codes, BATCH_SIZE)
    saver.save(sess, i)
    for q, a in tqdm(batcher):
        _, s = sess.run([model.op.op, model.summ.train_performance], feed_dict={model.ph.answer: a, model.ph.question:q})
        sw.add_summary(s, i)
        i+=1
    


# Greedy
model = ChatBot(max_length=max_length, charset_cardinality=charset_size, mode="greedy")

sess=start_tensorflow_session()

saver = tf.train.Saver()
# Introduce here the name of the model to load
saver.restore(sess, save_path=os.path.join(get_models_path(), "{}_{}".format(project_id, version_id))+"-28918")
while 1:
    question = input("HUMAN:   ")
    if not question.endswith("."):
        question += "."
    question = [question]
    question_code = np.expand_dims(np.array(process_dialog(question)), 2)
    output = sess.run(model.core_model.outputs.rnn_output,
                     feed_dict={model.ph.question: question_code,
                                model.ph.answer: np.zeros(shape=(1,151,1))})
    answer = "".join(list(map(code_to_character.get, np.argmax(output[0], 1))))
    answer = remove_substrings(answer, "$UNK$")
    print("MACHINE: %s"%answer)

