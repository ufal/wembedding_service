#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings client class.

WEmbeddings client computes word embeddings either locally or by sending
requests to a running WEmbeddings server (see wembeddings_server.py).
"""

import json
import pickle
import requests
import sys


class WEmbeddingsClient:
    def __init__(self, host):
        self._host = host


    def _make_post_request(self, model, sentences):
        response = requests.post(self._host,
                                 json.JSONEncoder().encode({"model": model,
                                                           "sentences": sentences}))
        if response.ok:
            print("Successfully processed request, time elapsed: {}".format(response.elapsed), file=sys.stderr, flush=True)
            return pickle.loads(response.content)
        else:
            raise RuntimeError("A WEmbeddings server error occured: Response status code = {}".format(response.status_code))
        

    def compute_embeddings(self, model, sentences):
        if self._host:
            return self._make_post_request(model, sentences)
        else:
            from . import wembeddings
            wembeddings = wembeddings.WEmbeddings()
            return wembeddings.compute_embeddings(model, sentences)
