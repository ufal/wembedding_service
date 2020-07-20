#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings computation."""

import sys

import numpy as np
import tensorflow as tf
import transformers


class WEmbeddings:
    def __init__(self):
        # TODO(Jana): have multiple models in self.models
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-multilingual-uncased", use_fast=True)
        self.model = transformers.TFAutoModel.from_pretrained("bert-base-multilingual-uncased")

    # TODO(Milan): Debug with tf.function to avoid repeated graph construction.
    #@staticmethod
    #@tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None, None], dtype=tf.int32)))
    def _compute_embeddings(self, subwords, segments):
        _, _, subword_embeddings_layers = self.model((subwords, tf.cast(tf.not_equal(subwords, 0), tf.int32)), output_hidden_states=tf.constant(True))
#         ret = self.model((subwords, tf.cast(tf.not_equal(subwords, 0), tf.int32)), output_hidden_states=tf.constant(True))
#         print([x.shape for x in ret])
        subword_embeddings = tf.math.reduce_mean(subword_embeddings_layers[-4:], axis=0)

        def average_subwords(embeddings_and_segments):
            subword_embeddings, segments = embeddings_and_segments
#            print(segments)
            return tf.math.segment_mean(subword_embeddings, segments)
        word_embeddings = tf.map_fn(average_subwords, (subword_embeddings[:, 1:], segments), dtype=tf.float32)[:, :-1]

        return word_embeddings


    def compute_embeddings(self, _model, sentences):
        batch = self.tokenizer(sentences, return_tensors="np", is_pretokenized=True)

        max_sentence_len = max(len(sentence) for sentence in sentences)
        segments = np.full([batch.data["input_ids"].shape[0], batch.data["input_ids"].shape[1] - 1], max_sentence_len, dtype=np.int32)
        for i, sentence in enumerate(sentences):
            for j in range(len(sentence)):
                start, end = batch.word_to_tokens(i, j)
                segments[i, start - 1:end - 1] = j

        return self._compute_embeddings(batch.data["input_ids"], segments)
