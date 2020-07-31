#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings client.

WEmbeddings client computes word embeddings either locally or by sending
requests to a running WEmbeddings server.

The script accepts either an input file or reads data from standard input. It
prints out results to standard output.

Script commandline arguments:
-----------------------------
 
    --batch_size: Batch size (maximum number of sentences per batch).
                  Default: 64.
    --host: Either host:port pair or None for local processing. Default:
            None
    --infile: Either path to file or None for reading from stdio. Default:
              None
    --model: Model name (see wembeddings.py for options). Default:
             "bert-base-multilignual-uncased-last4"
    --threads: Number of threads. Default: 4

Example usage:
--------------

$ ./run_wembeddings_client.py --infile=examples/client_input_single_column.conll

Input format:
-------------

The expected format is vertical, CoNLL-like segmented and tokenized tokens. One
token per line, sentences delimited with newline. The file can have multiple
columns, separated by tabs just like the CoNLL format. Tokens are expected in
the first column and the other columns are simply ignored.

Single column input example (see examples/client_input_single_column.conll):

John
loves
Mary
.

Mary
loves
John
.

Multiple column input example (see examples/client_input_multiple_column.conll):

John	John	PROPN
loves	love	VERB
Mary	Mary	PROPN
.	.	PUNCT

Mary	Mary	PROPN
loves	love	VERB
John	John	PROPN
.	.	PROPN

"""

import json
import pickle
import requests
import sys

import tensorflow as tf

from wembeddings import wembeddings_client


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size (maximum number of sentences per batch)")
    parser.add_argument("--host", default=None, type=str, help="ip:port or None")
    parser.add_argument("--infile", default=sys.stdin, type=argparse.FileType("r"), help="path to file or None for stdio")
    parser.add_argument("--model", default="bert-base-multilingual-uncased-last4", type=str, help="Model name (see wembeddings.py for options)")
    parser.add_argument("--threads", default=4, type=int, help="Number of threads")
    args = parser.parse_args()

    # Impose the limit on the number of threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Read sentences
    sentences = []  # batches x sentences x words
    batch = []
    sentence = []  
    nsentences = 0
    for line in args.infile:
        line = line.rstrip()
        if line:
            word = line.split("\t")[0]
            sentence.append(word)
        else:
            if len(batch) == args.batch_size:
                sentences.append(batch)
                batch = []
            batch.append(sentence)
            nsentences += 1
            sentence = []
    # Leftover batch or sentence
    if batch or sentence:
        if len(batch) == args.batch_size:
            sentences.append(batch)
            batch = []
        if sentence:
            batch.append(sentence)
            nsentences += 1
        if batch:
            sentences.append(batch)

    args.infile.close()
    print("Read {} sentences in {} batches.".format(nsentences, len(sentences)), file=sys.stderr, flush=True)

    # Compute word embeddings
    client = wembeddings_client.WEmbeddingsClient(args.host)
    for batch in sentences:
        outputs = client.compute_embeddings(args.model, batch)
        for i, sentence_output in enumerate(outputs):
            for j, word_output in enumerate(sentence_output):
                print(batch[i][j]," ".join(str(round(e, 6)) for e in word_output))
            print()
