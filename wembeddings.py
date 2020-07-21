#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2018 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings computation class."""

import sys

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
            tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_model, use_fast=True)

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
            pickled embeddings
        """

        model = self._models[model]

        batch = model.tokenizer.batch_encode_plus(sentences, return_tensors="tf", is_pretokenized=True)

        max_sentence_len = max(len(sentence) for sentence in sentences)
        segments = np.full([batch.data["input_ids"].shape[0], batch.data["input_ids"].shape[1] - 1], max_sentence_len, dtype=np.int32)
        for i, sentence in enumerate(sentences):
            for j in range(len(sentence)):
                start, end = batch.word_to_tokens(i, j)
                segments[i, start - 1:end - 1] = j

        return model.compute_embeddings(batch.data["input_ids"], tf.convert_to_tensor(segments))
