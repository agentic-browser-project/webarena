#!/usr/bin/env python3
"""Minimal threaded forward HTTP proxy to run ON hilbit2.

Handles absolute-URI HTTP requests (the WebArena sites are http://) and CONNECT.
Chromium reuses a small pool of keep-alive connections to this proxy, so only a few
SSH -L channels are needed (unlike ssh -D which opens one channel per site connection).
"""
import socket, threading, sys, select

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18900
BUFSIZE = 65536


def pipe(a, b):
    try:
        while True:
            r, _, _ = select.select([a, b], [], [], 60)
            if not r:
                break
            for s in r:
                data = s.recv(BUFSIZE)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except Exception:
        pass
    finally:
        for s in (a, b):
            try: s.close()
            except Exception: pass


def handle(client):
    try:
        client.settimeout(120)
        req = b""
        while b"\r\n\r\n" not in req:
            chunk = client.recv(BUFSIZE)
            if not chunk:
                client.close(); return
            req += chunk
            if len(req) > 1 << 20:
                break
        head, _, rest = req.partition(b"\r\n\r\n")
        lines = head.split(b"\r\n")
        method, target, _ = lines[0].split(b" ", 2)
        method = method.decode(); target = target.decode()

        if method == "CONNECT":
            host, port = target.rsplit(":", 1); port = int(port)
            upstream = socket.create_connection((host, port), timeout=30)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            pipe(client, upstream); return

        # absolute-URI: http://host:port/path
        if target.startswith("http://"):
            rest_t = target[len("http://"):]
            hostport, _, path = rest_t.partition("/")
            path = "/" + path
        else:
            # origin-form with Host header
            host_hdr = next((l for l in lines[1:] if l.lower().startswith(b"host:")), b"")
            hostport = host_hdr.split(b":", 1)[1].strip().decode() if host_hdr else ""
            path = target
        if ":" in hostport:
            host, port = hostport.rsplit(":", 1); port = int(port)
        else:
            host, port = hostport, 80
        upstream = socket.create_connection((host, port), timeout=30)
        # rebuild request line as origin-form, force Connection: close upstream simplicity off (keep-alive)
        new_head = b" ".join([method.encode(), path.encode(), b"HTTP/1.1"])
        hdrs = [l for l in lines[1:] if not l.lower().startswith(b"proxy-connection:")]
        upstream.sendall(new_head + b"\r\n" + b"\r\n".join(hdrs) + b"\r\n\r\n" + rest)
        pipe(client, upstream)
    except Exception:
        try: client.close()
        except Exception: pass


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((LISTEN_HOST, LISTEN_PORT)); srv.listen(512)
    print(f"httpproxy listening on {LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    while True:
        try:
            c, _ = srv.accept()
            threading.Thread(target=handle, args=(c,), daemon=True).start()
        except Exception:
            pass


if __name__ == "__main__":
    main()
