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

import signal
import sys
import threading

import wembeddings.wembeddings as wembeddings
import wembeddings.wembeddings_server as wembeddings_server

if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Port to use")
    parser.add_argument("--model", default="bert-base-multilingual-uncased-last4", type=str, help="Model name (see wembeddings.py for options)")
    parser.add_argument("--threads", default=4, type=int, help="Threads to use")
    args = parser.parse_args()

    # Create the server and its own thread
    server = wembeddings_server.WEmbeddingsServer(
        args.port,
        lambda: wembeddings.WEmbeddings(models_map={args.model: wembeddings.WEmbeddings.MODELS_MAP[args.model]},
                                        threads=args.threads),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("Starting WEmbeddings server on port {}.".format(args.port), file=sys.stderr)
    print("To stop it gracefully, either send SIGINT (Ctrl+C) or SIGUSR1.", file=sys.stderr, flush=True)

    # Wait until the server should be closed
    signal.sigwait([signal.SIGINT, signal.SIGUSR1])
    print("Initiating shutdown of the WEmbeddings server.", file=sys.stderr, flush=True)
    server.shutdown()
    print("Stopped handling new requests, processing all current ones.", file=sys.stderr, flush=True)
    server.server_close()
    print("Finished shutdown of the WEmbeddings server.", file=sys.stderr, flush=True)
