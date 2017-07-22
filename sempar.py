#!/usr/bin/env python
##
## Copyright © 2017 Max Kuznetsov <syhpoon@gmail.com>
##
import numpy as np
import json
import tensorflow as tf
import nltk
import os
import argparse

from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.saved_model import builder

PAD = "#@PAD@#"
UNK = "#@UNK@#"
START = "#@START@#"
STOP = "#@STOP@#"

class Seq2SeqModel(object):
    def __init__(self,
                 data_x,
                 data_y,
                 enc_keep_prob,
                 dec_keep_prob,
                 batch_size,
                 x_length,
                 y_length,
                 max_length,
                 vocab_size_x,
                 vocab_size_y,
                 embedding_size,
                 cell_size,
                 enc_num_layers,
                 dec_num_layers,
                 vocab_to_int_y):

        self.data_x = data_x
        self.data_y = data_y
        self.enc_keep_prob = enc_keep_prob
        self.dec_keep_prob = dec_keep_prob
        self.batch_size = batch_size
        self.x_length = x_length
        self.y_length = y_length
        self.max_length = max_length
        self.vocab_size_x = vocab_size_x
        self.vocab_size_y = vocab_size_y
        self.embedding_size = embedding_size
        self.cell_size = cell_size
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.vocab_to_int_y = vocab_to_int_y

        self._init_encoder()
        self._init_decoder()

    def _init_encoder(self):
        self.enc_embeddings = tf.Variable(
              tf.random_uniform([self.vocab_size_x,
                                 self.embedding_size], -1.0, 1.0),
                                 dtype=tf.float32)

        enc_embedded = tf.nn.embedding_lookup(self.enc_embeddings, self.data_x)

        with tf.variable_scope("encoder"):
            fw_cell = rnn_cell(self.cell_size,
                               self.enc_num_layers, self.enc_keep_prob)

            bw_cell = rnn_cell(self.cell_size,
                               self.enc_num_layers, self.enc_keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                                    cell_fw=fw_cell,
                                    cell_bw=bw_cell,
                                    inputs=enc_embedded,
                                    swap_memory=True,
                                    sequence_length=self.x_length,
                                    dtype=tf.float32)

        # Concatenate outputs of the forward and backward RNNs
        self.enc_outputs = tf.concat(outputs, 2)
        self.enc_states = states

    def _init_decoder(self):
        data_y = process_decoding_input(self.data_y, self.vocab_to_int_y,
                                        self.batch_size)

        self.dec_embeddings = tf.Variable(
                   tf.random_uniform([self.vocab_size_y,
                                      self.embedding_size], -1.0, 1.0),
                                      dtype=tf.float32)

        dec_embedded = tf.nn.embedding_lookup(self.dec_embeddings, data_y)

        with tf.variable_scope("decoder"):
            dec_cell = rnn_cell(self.cell_size, self.dec_num_layers,
                                self.dec_keep_prob)

        out_layer = Dense(self.vocab_size_y,
                          kernel_initializer=tf.truncated_normal_initializer(
                          mean=0.0, stddev=0.1))

        att_mechanism = seq2seq.BahdanauAttention(
            self.cell_size, self.enc_outputs, self.x_length, normalize=False)

        dec_cell = seq2seq.DynamicAttentionWrapper(
            dec_cell, att_mechanism, attention_size=self.cell_size)

        init_state = seq2seq.DynamicAttentionWrapperState(
            cell_state=self.enc_states[0],
            attention=_zero_state_tensors(
                self.cell_size, self.batch_size, tf.float32))

        with tf.variable_scope("decoding"):
            train_helper = seq2seq.TrainingHelper(dec_embedded,
                                                  sequence_length=self.y_length,
                                                  time_major=False)

            train_decoder = seq2seq.BasicDecoder(dec_cell, train_helper,
                                                 init_state, out_layer)

            train_out, _ = seq2seq.dynamic_decode(
                train_decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.max_length,
                swap_memory=True)

            self.decoder_train = train_out.rnn_output

        with tf.variable_scope("decoding", reuse=True):
            start_tokens = tf.tile(
                tf.constant([self.vocab_to_int_y[START]], dtype=tf.int32),
                [self.batch_size])

            infer_helper = seq2seq.GreedyEmbeddingHelper(
                                 embedding=self.dec_embeddings,
                                 start_tokens=start_tokens,
                                 end_token=self.vocab_to_int_y[STOP])

            infer_decoder = seq2seq.BasicDecoder(dec_cell, infer_helper,
                                                 init_state, out_layer)

            infer_out, _ = seq2seq.dynamic_decode(infer_decoder,
                                                  output_time_major=False,
                                                  impute_finished=True,
                                                  maximum_iterations=self.max_length
                                                  )

            self.decoder_inference = infer_out.sample_id

        tf.identity(self.decoder_train, 'decoder_train')
        tf.identity(self.decoder_inference, 'decoder_inference')

def batch_data(source, target, batch_size, pad_code):
    """
    Batch source and target together
    """
    for batch_i in range(0, len(source)//batch_size):
        start_i = batch_i * batch_size
        source_batch = source[start_i:start_i + batch_size]
        target_batch = target[start_i:start_i + batch_size]

        source_batch = np.array(pad_sentence_batch(source_batch, pad_code))
        target_batch = np.array(pad_sentence_batch(target_batch, pad_code))

        source_lengths = [len(x) for x in source_batch]
        target_lengths = [len(x) for x in target_batch]

        yield (source_batch, target_batch, source_lengths, target_lengths)

def pad_sentence_batch(sentence_batch, pad_code):
    """
    Pad sentence with <PAD> id
    """
    max_sentence = max([len(sentence) for sentence in sentence_batch])

    return [sentence + [pad_code] * (max_sentence - len(sentence))
            for sentence in sentence_batch]

def pad(seq, max_size, pad_char):
    return seq + [pad_char] * (max_size - len(seq))

def parse_input(text):
    tok = nltk.WordPunctTokenizer()

    return [z.lower() for z in tok.tokenize(text)]

def preprocess(pairs):
    """
    Preprocess raw corpus pairs
    
    :param pairs: A list of pairs (x, y)
    :return: xs, ys, vocabulary_x, vocabulary_y, counts
    """

    xs = []
    ys = []
    vocab_x = set()
    vocab_y = set()
    counts = {}

    for (x, y) in pairs:
        x_words = parse_input(x)
        y_words = parse_input(y)

        for w in x_words:
            counts[w] = counts.get(w, 0) + 1

        for w in y_words:
            counts[w] = counts.get(w, 0) + 1

        xs.append(x_words)
        ys.append(y_words)

        vocab_x.update(x_words)
        vocab_y.update(y_words)

    return xs, ys, vocab_x, vocab_y, counts

def words2ids(text, vocab_to_int, eos=None):
    """
    Convert words into word ids
    
    :param text: List of words
    :param vocab_to_int:  
    :return: List of word ids
    """

    r = [vocab_to_int.get(x, vocab_to_int[UNK]) for x in text]

    if eos is not None:
        r.append(eos)

    return r

def load_pairs(xpath, ypath):
    fx = open(xpath)
    fy = open(ypath)

    data_x = fx.read().rstrip().split("\n")
    data_y = fy.read().rstrip().split("\n")

    fx.close()
    fy.close()

    assert(len(data_x) == len(data_y))

    return zip(data_x, data_y)

def create_lookup_tables(vocab, counts, cutoff_size=None, size_limit=None):
    """
    Create lookup tables for vocabulary
    :param vocab: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    vocab_to_int = {}
    int_to_vocab = {}

    _sorted = sorted(vocab, reverse=True, key=lambda x: counts[x])

    for i, word in enumerate([PAD, UNK, START, STOP] + _sorted):
        if size_limit is not None and i > size_limit:
            break

        vocab_to_int[word] = i
        int_to_vocab[i] = word

        if cutoff_size is not None and i > 3 and counts[word] < cutoff_size:
            break

    return vocab_to_int, int_to_vocab

def usage():
    print("Usage: sempar.py [train <corpus-file> | infer <text>]")

    exit(1)

def save_params(path, params):
    """
    Save parameters to file
    """

    with open(path, 'w') as f:
        json.dump(params, f)

def load_params(path):
    """
    Load parameters from file
    """

    with open(path, 'r') as f:
        return json.load(f)

def process_decoding_input(target_data, vocab_to_int_y, batch_size):
    """
    Preprocess target data for dencoding
    :param target_data: Target Placehoder
    :param vocab_to_int_y: Dictionary to go from the y words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    return tf.concat([tf.fill([batch_size, 1],
                              vocab_to_int_y[START]), ending], 1)

def rnn_cell(cell_size, num_layers, keep_prob=1.):
    """
    Create encoding layer
    :param cell_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :return: RNN state
    """

    cells = []

    for i in range(num_layers):
        cell = tf.contrib.rnn.LSTMCell(
                  cell_size, initializer=tf.random_uniform_initializer(
                -0.1, 0.1))

        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             output_keep_prob=1.,
                                             input_keep_prob=keep_prob)

        cells.append(cell)

    return tf.contrib.rnn.MultiRNNCell(cells)

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """

    max_seq = max(target.shape[1], logits.shape[1])

    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0, 0), (0, max_seq - target.shape[1])], 'constant')

    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

## TODO:
## Beam search

def train(args, parser, checkpoint, params_file, export_dir):
    if args.corpus == "":
        print("Corpus path is required")
        parser.print_help()
        return

    corpus_dir = args.corpus

    pairs = load_pairs("{}/sources.txt".format(corpus_dir),
                       "{}/targets.txt".format(corpus_dir))

    xs, ys, vocab_x, vocab_y, counts = preprocess(pairs)

    vocab_to_int_x, int_to_vocab_x = create_lookup_tables(
        vocab_x, counts, args.cutoff_size, args.dict_size)
    vocab_to_int_y, int_to_vocab_y = create_lookup_tables(
        vocab_y, counts, args.cutoff_size)

    vocab_size_x = len(vocab_to_int_x)
    vocab_size_y = len(vocab_to_int_y)

    xs_ids = [words2ids(x, vocab_to_int_x) for x in xs]
    ys_ids = [words2ids(y, vocab_to_int_y, vocab_to_int_y[STOP]) for y in ys]

    max_seq_length = min(max([len(z) for z in xs_ids]), 50)

    xs_ids = [x[:max_seq_length] for x in xs_ids]

    # Build model
    batch_size = args.batch_size
    cell_size = args.cell_size
    enc_num_layers = 2
    dec_num_layers = 2
    embed_size = args.embed_size
    learning_rate = 0.0003
    enc_keep_probability = 0.8
    dec_keep_probability = enc_keep_probability
    display_step = args.display_step
    global_step = 0

    test_len = min(int(len(xs_ids) * 0.2), 10000)

    pad_code = vocab_to_int_x[PAD]
    stop_code = vocab_to_int_x[STOP]

    train_xs = xs_ids[test_len:]
    train_ys = ys_ids[test_len:]
    test_xs = xs_ids[:test_len]
    test_ys = ys_ids[:test_len]

    print("Training model:\n"
          " number of train_pairs={}\n"
          " number of test pairs={}\n"
          " vocab_size_x={}, \n"
          " vocab_size_y={}".format(
          len(train_xs), len(test_xs),
          vocab_size_x, vocab_size_y))

    train_graph = tf.Graph()

    with train_graph.as_default():
        input_data = tf.placeholder(tf.int32, [None, None], name="input_data")
        targets = tf.placeholder(tf.int32, [None, None], name="targets")
        lr = tf.placeholder(tf.float32, name="learning_rate")
        enc_keep_prob = tf.placeholder(tf.float32, name="enc_keep_prob")
        dec_keep_prob = tf.placeholder(tf.float32, name="dec_keep_prob")

        x_length = tf.placeholder(tf.int32, (None,), name="x_length")
        y_length = tf.placeholder(tf.int32, (None,), name="y_length")
        max_y_length = tf.reduce_max(y_length)

        model = Seq2SeqModel(
            data_x=tf.reverse(input_data, [-1]),
            data_y=targets,
            enc_keep_prob=enc_keep_prob,
            dec_keep_prob=dec_keep_prob,
            batch_size=batch_size,
            x_length=x_length,
            y_length=y_length,
            max_length=max_y_length,
            vocab_size_x=vocab_size_x+4,
            vocab_size_y=vocab_size_y+4,
            embedding_size=embed_size,
            cell_size=cell_size,
            enc_num_layers=enc_num_layers,
            dec_num_layers=dec_num_layers,
            vocab_to_int_y=vocab_to_int_y)

        masks = tf.sequence_mask(y_length,
                                 max_y_length,
                                 dtype=tf.float32,
                                 name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                           model.decoder_train, targets, masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)

            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                                for grad, var in gradients if grad is not None]

            train_op = optimizer.apply_gradients(capped_gradients)

    losses = set()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(graph=train_graph, config=config) as ses:
        ses.run(tf.global_variables_initializer())

        for epoch_i in range(args.epochs):
            for batch_i, (x_batch, y_batch, x_lens, y_lens) in enumerate(
                  batch_data(train_xs, train_ys, batch_size, pad_code)):

                global_step += 1

                _, loss = ses.run(
                                [train_op, cost],
                                {input_data: x_batch,
                                 targets: y_batch,
                                 lr: learning_rate,
                                 x_length: x_lens,
                                 y_length: y_lens,
                                 enc_keep_prob: enc_keep_probability,
                                 dec_keep_prob: dec_keep_probability,
                                 })

                if not losses:
                    losses.add(loss)

                elif loss < min(losses):
                    print("New best loss: {:>6.5f}, "
                          "epoch={}, global_step={} "
                          "batch={}/{}".format(loss,
                                               epoch_i,
                                               global_step,
                                               batch_i,
                                               len(xs_ids) // batch_size
                                               ))

                    saver = tf.train.Saver()
                    saver.save(ses, checkpoint)

                    save_params(params_file,
                                {"vocab_to_int_x": vocab_to_int_x,
                                 "int_to_vocab_x": int_to_vocab_x,
                                 "int_to_vocab_y": int_to_vocab_y,
                                 "batch_size": batch_size,
                                 "pad_code": pad_code,
                                 "stop_code": stop_code,
                                 }
                                )

                    losses.add(loss)

                if global_step > 0 and global_step % display_step == 0:
                    print('Calculating intermediate results')

                    batch_train_logits = ses.run(model.decoder_inference,
                                                 {input_data: x_batch,
                                                  enc_keep_prob: 1.,
                                                  dec_keep_prob: 1.,
                                                  x_length: x_lens,
                                                  y_length: y_lens,
                                                 })

                    test_acc = calculate_test_acc(
                        ses,
                        model.decoder_inference,
                        test_xs,
                        test_ys,
                        batch_size,
                        pad_code,
                        input_data,
                        enc_keep_prob,
                        dec_keep_prob,
                        x_length,
                        y_length)

                    train_acc = get_accuracy(y_batch, batch_train_logits)

                    print('Epoch {:>3} Batch {:>4}/{} - '
                          'Avg Test Accuracy: {:>6.5f} '
                          'Train Accuracy: {:>6.5f}, Loss: {:>6.5f}'.format(
                          epoch_i, batch_i, len(xs_ids) // batch_size,
                          test_acc, train_acc, loss))

        b = builder.SavedModelBuilder(export_dir)
        b.add_meta_graph_and_variables(
            ses, [tf.saved_model.tag_constants.SERVING])

        b.save()

def calculate_test_acc(ses, decoder_inference, test_xs,
                       test_ys, batch_size, pad_code,
                       input_data, enc_keep_prob,
                       dec_keep_prob, x_length, y_length):
    """
    Calculate average accuracy on train data
    """

    accs = []

    for (x_batch, y_batch, x_lens, y_lens) in batch_data(
            test_xs, test_ys, batch_size, pad_code):

            logits = ses.run(decoder_inference,
                             {input_data: x_batch,
                              enc_keep_prob: 1.,
                              dec_keep_prob: 1.,
                              x_length: x_lens,
                              y_length: y_lens,
                            })

            accs.append(get_accuracy(y_batch, logits))

    return sum(accs) / len(accs)

def infer(args, parser, checkpoint, params_file, export_dir):
    if not args.text:
        parser.print_help()
        return

    input_data = args.text

    params = load_params(params_file)
    parsed = parse_input(input_data)

    ids = [i for i in words2ids(parsed, params['vocab_to_int_x'])]

    if args.mode == "infer-model":
        print("Infering from model '{}', text={}".format(
            export_dir, input_data))

        res = infer_from_model(ids, params['batch_size'], export_dir)
    else:
        print("Infering from checkpoint '{}', text={}".format(
            checkpoint, input_data))

        res = infer_from_checkpoint(ids, params['batch_size'])

    print(" ".join(
        [params['int_to_vocab_y'][str(i)]
         for i in res if i not in (params['pad_code'], params['stop_code'])]))

def infer_from_checkpoint(ids, batch_size):
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as ses:
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(ses, checkpoint)

        res = ses.run('decoder_inference:0',
                      {'input_data:0': [ids] * batch_size,
                       'enc_keep_prob:0': 1.,
                       'dec_keep_prob:0': 1.,
                       'x_length:0': [len(ids)] * batch_size,
                       'y_length:0': [30],
                       })[0]

        return res

def infer_from_model(ids, batch_size, export):
    with tf.Session(graph=tf.Graph()) as ses:
        tf.saved_model.loader.load(ses,
                                   [tf.saved_model.tag_constants.SERVING],
                                   export)

        res = ses.run('decoder_inference:0',
                      {'input_data:0': [ids] * batch_size,
                       'enc_keep_prob:0': 1.,
                       'dec_keep_prob:0': 1.,
                       'x_length:0': [len(ids)] * batch_size,
                       'y_length:0': [30],
                       })[0]

        return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic parser')
    parser.add_argument("mode", help="Mode", choices=["train", "infer-model",
                                                      "infer-checkpoint"])
    parser.add_argument('-s', '--save', default='save',
                        help='Save directory')
    parser.add_argument('-c', '--corpus', default='',
                        help='Corpus directory for training')
    parser.add_argument('-t', '--text', default='',
                        help='Text for infering mode')
    parser.add_argument('-f', '--cutoff_size', default=1, type=int,
                        help='Word count cutoff value')
    parser.add_argument('-d', '--dict_size', default=100000, type=int,
                        help='Maximum dict size')
    parser.add_argument('-e', '--epochs', default=2, type=int,
                        help='Number epochs')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('-l', '--cell_size', default=500, type=int,
                        help='Cell size')
    parser.add_argument('-m', '--embed_size', default=600, type=int,
                        help='Embedding size')
    parser.add_argument('-p', '--display_step', default=1000, type=int,
                        help='Display step')

    args = parser.parse_args()

    checkpoint = "{}/best_model.ckpt".format(args.save)
    params_file = "{}/params.json".format(args.save)
    export_dir = "{}/export".format(args.save)

    try:
        os.makedirs(args.save)
    except Exception:
        pass

    if args.mode == "train":
        if os.path.exists(export_dir):
            import shutil
            shutil.rmtree(export_dir)

        train(args, parser, checkpoint, params_file, export_dir)
    elif args.mode in ("infer-model", "infer-checkpoint"):
        infer(args, parser, checkpoint, params_file, export_dir)
