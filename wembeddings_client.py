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
requests to a running WEmbeddings server (see wembeddings_server.py).

The client can be used as a script or imported as a module.

The script accepts either an input file or reads data from standard input. It
prints out results to standard output.

Script commandline arguments:
-----------------------------
    
    --host: Either host:port pair or None for local processing. Default:
            None
    --infile: Either path to file or None for reading from stdio. Default:
              None
    --model: Model name (see wembeddings.py for options). Default:
             "bert-base-multilignual-uncased-last4"
    --batch_size: Batch size (maximum number of sentences per batch).
                  Default: 64.

Example usage:
--------------

$ ./wembeddings_client.py --infile=examples/client_input.conll

Input format:
-------------

The expected format is vertical, CoNLL-like segmented and tokenized tokens. One
token per line, sentences delimited with newline.

Input example (see also examples/client_input.conll):

John
loves
Mary
.

Mary
loves
John
.
"""

import json
import pickle
import requests
import sys


class WEmbeddingsClient:
    def __init__(self, host, batch_size):
        self._host = host
        self._batch_size = batch_size


    def _make_post_request(self, model, sentences):
        response = requests.post(self._host,
                                 json.JSONEncoder().encode({"model": model,
                                                           "sentences": sentences}))
        if response.ok:
            print("Successfully processed request, time elapsed: {}".format(response.elapsed), file=sys.stderr, flush=True)
            return pickle.loads(response.content)
        else:
            print("A server error occured: Response status code = {}".format(response.status_code), file=sys.stderr, flush=True)
            sys.exit(1)
        

    def compute_embeddings(self, model, sentences):
        if self._host:
            print("Sending requests to {}".format(self._host), file=sys.stderr, flush=True)

            outputs = []
            batch = []
            for sentence in sentences:
                if len(batch) == self._batch_size:
                    outputs.extend(self._make_post_request(model, batch))
                    batch = []
                batch.append(sentence)
            if batch:
                outputs.extend(self._make_post_request(model, batch))
            return outputs

        else:
            print("Computing word embeddings locally.", file=sys.stderr, flush=True)

            import wembeddings

            wembeddings = wembeddings.WEmbeddings()
            outputs = []
            batch = []
            for sentence in sentences:
                if len(batch) == self._batch_size:
                    outputs.extend(wembeddings.compute_embeddings(model, batch))
                    batch = []
                batch.append(sentence)
            if batch:
                outputs.extend(wembeddings.compute_embeddings(model, batch))
            return outputs


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None, type=str, help="ip:port or None")
    parser.add_argument("--infile", default=sys.stdin, type=argparse.FileType("r"), help="path to file or None for stdio")
    parser.add_argument("--model", default="bert-base-multilingual-uncased-last4", type=str, help="Model name (see wembeddings.py for options)")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size (maximum number of sentences per batch)")
    args = parser.parse_args()

    # Read sentences
    sentences = []  
    sentence = []  
    for line in args.infile:
        word = line.rstrip()
        if word:
            sentence.append(word)
        else:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)
    args.infile.close()
    print("Read {} sentences.".format(len(sentences)), file=sys.stderr, flush=True)

    # Compute word embeddings
    client = WEmbeddingsClient(args.host, args.batch_size)
    outputs = client.compute_embeddings(args.model, sentences)

    # Print outputs
    for sentence_output in outputs:
        for word_output in sentence_output:
            print(word_output)
        print()
