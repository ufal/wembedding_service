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
import sys
import threading

import numpy as np

class WEmbeddingsServer(socketserver.ThreadingTCPServer):

    class WEmbeddingsRequestHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def respond_error(request, message):
            request.close_connection = True
            request.send_response(400)
            request.send_header("Connection", "close")
            request.send_header("Content-Type", "text/plain")
            request.end_headers()
            request.wfile.write(message.encode("utf-8"))

        def respond_ok(request, content_type):
            request.close_connection = True
            request.send_response(200)
            request.send_header("Connection", "close")
            request.send_header("Content-Type", content_type)
            request.end_headers()

        def do_POST(request):
            if request.headers.get("Transfer-Encoding", "identity").lower() != "identity":
                request.respond_error("Only 'identity' Transfer-Encoding of payload is supported for now.")
                return

            if "Content-Length" not in request.headers:
                request.respond_error("The Content-Length of payload is required.")
                return

            try:
                length = int(request.headers["Content-Length"])
                data = json.loads(request.rfile.read(length))
                with request.server._wembeddings_thread_mutex:
                    request.server._wembeddings_thread_input = data
                    request.server._wembeddings_thread_have_input.set()
                    request.server._wembeddings_thread_have_output.wait()
                    request.server._wembeddings_thread_have_output.clear()
                    sentences_embeddings = request.server._wembeddings_thread_output
            except:
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                request.respond_error("Malformed request.")
                return

            if sentences_embeddings is None:
                request.respond_error("An error occurred during wembeddings computation.")
            else:
                request.respond_ok("application/octet_stream")
                for sentence_embedding in sentences_embeddings:
                    np.lib.format.write_array(request.wfile, sentence_embedding.astype(request.server._dtype), allow_pickle=False)

    daemon_threads = False

    def __init__(self, port, dtype, wembeddings_lambda):
        self._dtype = dtype

        # Create the worker thread and required synchronization
        self._wembeddings_thread_mutex = threading.Lock()
        self._wembeddings_thread_have_input = threading.Event()
        self._wembeddings_thread_have_output = threading.Event()
        self._wembeddings_thread = threading.Thread(target=self._wembeddings_thread_code, args=(wembeddings_lambda,), daemon=True)
        self._wembeddings_thread.start()

        # Wait for the worker thread to start
        self._wembeddings_thread_have_output.wait()
        self._wembeddings_thread_have_output.clear()

        # Initialize the server
        super().__init__(("", port), self.WEmbeddingsRequestHandler)

    def _wembeddings_thread_code(self, wembeddings_lambda):
        # Create the WEmbeddings object
        self._wembeddings = wembeddings_lambda()

        # Notify that the thread is ready
        self._wembeddings_thread_have_output.set()

        # Start the real work
        while True:
            self._wembeddings_thread_have_input.wait()
            self._wembeddings_thread_have_input.clear()
            try:
                self._wembeddings_thread_output = self._wembeddings.compute_embeddings(
                    self._wembeddings_thread_input["model"], self._wembeddings_thread_input["sentences"])
            except:
                import traceback
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                self._wembeddings_thread_output = None
            self._wembeddings_thread_have_output.set()

    def server_bind(self):
        import socket
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        super().server_bind()
