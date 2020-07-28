#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings computation class."""

import sys
import time

import numpy as np
import tensorflow as tf
import transformers


class WEmbeddings:
    """Class to keep multiple constructed word embedding computation models."""

    MODELS_MAP = {
        # Key: model name. Value: transformer model name, layer start, layer end.
        "bert-base-multilingual-uncased-last4": ("bert-base-multilingual-uncased", -4, None),
    }

    class _Model:
        """Keeps constructed tokenizer and transformers model graph."""
        def __init__(self, tokenizer, transformers_model, compute_embeddings):
            self.tokenizer = tokenizer
            self.transformers_model = transformers_model
            self.compute_embeddings = compute_embeddings

    def __init__(self):
        self._models = {}
        for model_name, (transformers_model, layer_start, layer_end) in self.MODELS_MAP.items():
            tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_model)

            transformers_model = transformers.TFAutoModel.from_pretrained(
                transformers_model,
                config=transformers.AutoConfig.from_pretrained(transformers_model, output_hidden_states=True),
            )

            def compute_embeddings(subwords, segments):
                _, _, subword_embeddings_layers = transformers_model((subwords, tf.cast(tf.not_equal(subwords, 0), tf.int32)))
                subword_embeddings = tf.math.reduce_mean(subword_embeddings_layers[layer_start:layer_end], axis=0)

                # Average subwords (word pieces) word embeddings for each token 
                def average_subwords(embeddings_and_segments):
                    subword_embeddings, segments = embeddings_and_segments
                    return tf.math.segment_mean(subword_embeddings, segments)
                word_embeddings = tf.map_fn(average_subwords, (subword_embeddings[:, 1:], segments), dtype=tf.float32)[:, :-1]
                return word_embeddings

            compute_embeddings = tf.function(compute_embeddings).get_concrete_function(
                tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            )

            self._models[model_name] = self._Model(tokenizer, transformers_model, compute_embeddings)


    def compute_embeddings(self, model, sentences):
        """Computes word embeddings.
        Arguments:
            model name: one of the keys of self._MODELS_MAP.
            sentences: 2D Python array with sentences with tokens (strings).
        Returns:
            embeddings as a Python list of 1D Numpy arrays
        """

        if model not in self._models:
            print("No such WEmbeddings model {}".format(model), file=sys.stderr, flush=True)

        model = self._models[model]

        start = time.time()

        subwords, segments = [], []
        for sentence in sentences:
            segments.append([])
            subwords.append([])
            for i, word in enumerate(sentence):
                word_subwords = model.tokenizer.encode(word, add_special_tokens=False)
                segments[-1].extend([i] * len(word_subwords))
                subwords[-1].extend(word_subwords)
            subwords[-1] = model.tokenizer.build_inputs_with_special_tokens(subwords[-1])

        max_sentence_len = max(len(sentence) for sentence in sentences)
        max_subwords = max(len(sentence) for sentence in subwords)

        print("Max sentence len", max_sentence_len, "max subwords", max_subwords, "batch subwords", len(sentences) * max_subwords, "in", time.time() - start, file=sys.stderr)

        np_subwords = np.zeros([len(subwords), max_subwords], np.int32)
        for i, subwords in enumerate(subwords):
            np_subwords[i, :len(subwords)] = subwords

        np_segments = np.full([len(subwords), max_subwords - 1], max_sentence_len, np.int32)
        for i, segments in enumerate(segments):
            np_segments[i, :len(segments)] = segments

        start = time.time()
        embeddings = model.compute_embeddings(tf.convert_to_tensor(np_subwords), tf.convert_to_tensor(np_segments)).numpy()
        print("BERT in", time.time() - start, file=sys.stderr)

        return [embeddings[i, :len(sentences[i])] for i in range(len(sentences))]
