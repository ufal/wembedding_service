#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings server.

Example setup:
$ venv/bin/python ./wembeddings_server.py

Example call:
$ curl --data-binary @examples/request.json localhost:8000/ | xxd
"""

import http.server
import json
import pickle
import socketserver
import sys

from wembeddings import wembeddings_server


PORT = 8000

       
if __name__ == "__main__":

    with wembeddings_server.WEmbeddingsServer(PORT) as server:
        print("Serving WEmbeddings at port {}".format(PORT), file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        server.server_close()
