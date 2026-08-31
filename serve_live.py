#!/usr/bin/env python3
"""Tiny static server for the Task F live viewer.

Serves this repo's directory so ``task_f_viewer.html`` can poll the live run data at
``figures/task_f/live/`` (browsers block ``fetch`` over ``file://``, hence a server).

Usage:
    uv run python3 serve_live.py            # serve on the first free port from 8010, open a tab
    uv run python3 serve_live.py --port 9000 --no-open

Run Task F with ``--f-live`` in another terminal to produce the live data.
"""
import argparse
import functools
import http.server
import os
import socket
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE = "task_f_viewer.html"


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serve from ROOT with caching disabled (so the poller always sees fresh data)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, *args):  # quiet: don't spam the console with every poll
        pass


def _find_port(start):
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:  # nothing listening → free
                return port
    raise RuntimeError(f"no free port found in [{start}, {start + 100})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=None, help="port (default: first free from 8010)")
    ap.add_argument("--no-open", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()

    port = args.port or _find_port(8010)
    url = f"http://127.0.0.1:{port}/{PAGE}"
    live = os.path.join(ROOT, "figures", "task_f", "live")
    if not os.path.isdir(live):
        print(f"note: {live}/ does not exist yet — run Task F with --f-live to create it.")
    print(f"Task F live viewer:  {url}")
    print("Serving", ROOT)
    print("Ctrl-C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")


if __name__ == "__main__":
    main()
