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
import socketserver
import threading

import numpy as np

class WEmbeddingsServer(socketserver.ThreadingTCPServer):

    class WEmbeddingsRequestHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(request):
            request.close_connection = True
            request.send_header("Connection", "close")

            try:
                length = int(request.headers.get("Content-length", -1))
                data = json.loads(request.rfile.read(length))
                with request.server._wembeddings_thread_mutex:
                    request.server._wembeddings_thread_input = data
                    request.server._wembeddings_thread_have_input.set()
                    request.server._wembeddings_thread_have_output.wait()
                    request.server._wembeddings_thread_have_output.clear()
                    sentences_embeddings = request.server._wembeddings_thread_output
            except:
                request.send_response(400)
                request.end_headers()
                raise

            request.send_response(200)
            request.send_header("Content-type", "application/octet-stream")
            request.end_headers()

            for sentence_embedding in sentences_embeddings:
                np.save(request.wfile, sentence_embedding.astype(np.float16), allow_pickle=False)

    allow_reuse_address = True
    daemon_threads = False

    def __init__(self, port, wembeddings_lambda):
        super().__init__(("", port), self.WEmbeddingsRequestHandler)

        self._wembeddings_thread_mutex = threading.Lock()
        self._wembeddings_thread_have_input = threading.Event()
        self._wembeddings_thread_have_output = threading.Event()
        self._wembeddings_thread = threading.Thread(target=self._wembeddings_thread_code, args=(wembeddings_lambda,), daemon=True)
        self._wembeddings_thread.start()

        # Wait for the worker thread to start
        self._wembeddings_thread_have_output.wait()
        self._wembeddings_thread_have_output.clear()

    def _wembeddings_thread_code(self, wembeddings_lambda):
        # Create the WEmbeddings object
        self._wembeddings = wembeddings_lambda()

        # Notify that the thread is ready
        self._wembeddings_thread_have_output.set()

        # Start the real work
        while True:
            self._wembeddings_thread_have_input.wait()
            self._wembeddings_thread_have_input.clear()
            self._wembeddings_thread_output = self._wembeddings.compute_embeddings(
                self._wembeddings_thread_input["model"], self._wembeddings_thread_input["sentences"])
            self._wembeddings_thread_have_output.set()
