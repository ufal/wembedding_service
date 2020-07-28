#!/usr/bin/env python3
# coding=utf-8
#
# Copyright 2020 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Word embeddings server class."""

import http.server
import json
import pickle
import socketserver
import sys

from . import wembeddings


PORT = 8000


class WEmbeddingsRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(request):
        length = int(request.headers.get("content-length", -1))
        data = json.loads(request.rfile.read(length))
        
        try:
            output = request.server.wembeddings.compute_embeddings(data["model"], data["sentences"])

            request.send_response(200)
            request.send_header("Content-type", "application/octet-stream")
            request.end_headers()

            pickle.dump(output, request.wfile, protocol=3)
        except:
            request.send_response(400)
            request.end_headers()
            raise


class WEmbeddingsServer(socketserver.TCPServer):
    allow_reuse_address = 1

    def __init__(self, port):
        super().__init__(("", port), WEmbeddingsRequestHandler)

        self.wembeddings = wembeddings.WEmbeddings()
