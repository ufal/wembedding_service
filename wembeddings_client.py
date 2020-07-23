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

This client computes word embeddings either locally by importing the
WEmbeddings class as a module or by sending requests to a running WEmbeddings
server (see wembeddings_server.py) indentified by a host:port pair.

The client accepts either an input file or reads data from standard input.

It prints out results to standard output.

Commandline arguments:
----------------------
    
    --host: Either host:port pair or None for local processing. Default:
            None
    --infile: Either path to file or None for reading from stdio. Default:
              None
    --model: Model name (see wembeddings.py for options). Default:
             "bert-base-multilignual-uncased-last4"

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


import sys


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None, type=str, help="host:port or None to process locally")
    parser.add_argument("--infile", default=sys.stdin, type=argparse.FileType("r"), help="path to file or None for stdio")
    parser.add_argument("--model", default="bert-base-multilingual-uncased-last4", type=str, help="Model name (see wembeddings.py for options)")
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

    # Compute word embeddings
    if args.host:
        print("Sending requests to {}".format(args.host), file=sys.stderr, flush=True)

        import json
        import pickle
        import requests

        response = requests.post(args.host,
                                 json.JSONEncoder().encode({"model": args.model,
                                                            "sentences": sentences}))
        if response.ok:
            print("Successfully processed request, time elapsed: {}".format(response.elapsed), file=sys.stderr, flush=True)
            outputs = pickle.loads(response.content)
        else:
            print("A server error occured: Response status code = {}".format(response.status_code), file=sys.stderr, flush=True)
            sys.exit(1)

    else:
        print("Computing word embeddings locally.", file=sys.stderr, flush=True)

        import wembeddings
       
        wembeddings = wembeddings.WEmbeddings()
        outputs = wembeddings.compute_embeddings(args.model, sentences)

    # Print outputs
    for sentence_output in outputs:
        for word_output in sentence_output:
            print(word_output)
        print()
