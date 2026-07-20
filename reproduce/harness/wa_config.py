"""WebArena site config — TWO physical servers (hilbit2 .158 + hilbit1 .153), many replicas.
Reached through the hilbit2 HTTP proxies (which can route to BOTH .158 and .153)."""
import os, json, copy
from urllib.parse import urlparse

H2 = os.environ.get("WA_HOST2", "158.130.4.158")   # hilbit2 (override: WA_HOST2)
H1 = os.environ.get("WA_HOST1", "158.130.4.153")   # hilbit1 (override: WA_HOST1)
HOST = H2              # back-compat default
PROXY = "socks5://127.0.0.1:1080"

def _u(host, port, path=""):
    return f"http://{host}:{port}{path}"

_WIKI = "/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"

# Replica URLs per site = hilbit2 replicas + hilbit1 replicas.
REPLICAS = {
    "shopping":       [_u(H2, p) for p in (7770, 7870, 7970)]
                    + [_u(H1, p) for p in (7770, 7870, 7970, 8070, 8170, 8270, 8370)],
    "shopping_admin": [_u(H2, p, "/admin") for p in (7780, 7880, 7980)]
                    + [_u(H1, p, "/admin") for p in (7780, 7880, 7980, 8090, 8180, 8280, 8380)],
    "reddit":         [_u(H2, p) for p in (9999, 10099, 10199)]
                    + [_u(H1, p) for p in (9999, 10099, 10199, 10299, 10399, 10499, 10599)],
    "gitlab":         [_u(H2, p) for p in (8023, 8123, 8223)]
                    + [_u(H1, p) for p in (8023, 8123, 8223, 8323, 8423, 8523)],  # .153:8623 unhealthy, dropped
    "wikipedia":      [_u(H2, 8888, _WIKI), _u(H1, 8888, _WIKI)],
    # Map: self-hosted WebArena OSM (:13000) — IDENTICAL to the main running experiment.
    # To instead run map on the REAL https://www.openstreetmap.org, set WA_MAP_URL — but FIRST read
    # README "Optional: map on real OpenStreetMap" (reference-answer mismatch + anti-scraping risks).
    "map":            ([os.environ["WA_MAP_URL"]] if os.environ.get("WA_MAP_URL")
                       else [_u(H2, 13000), _u(H1, 13000)]),
    "homepage":       [_u(H2, 4399), _u(H1, 4399)],
}

# Ports per replica (derived) — used by the scheduler for slot counting.
PORTS = {s: [urlparse(u).port for u in urls] for s, urls in REPLICAS.items()}

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
}

PLACEHOLDER = {
    "__SHOPPING__": "shopping", "__SHOPPING_ADMIN__": "shopping_admin",
    "__REDDIT__": "reddit", "__GITLAB__": "gitlab",
    "__WIKIPEDIA__": "wikipedia", "__MAP__": "map",
}

_HERE = os.path.dirname(os.path.abspath(__file__))
# login cookies: default <this dir>/auth, override with WA_AUTH_DIR
AUTH_DIR = os.environ.get("WA_AUTH_DIR", os.path.join(_HERE, "auth"))


def base_url(site, replica_idx=0):
    urls = REPLICAS[site]
    return urls[replica_idx % len(urls)]


def origin(site, replica_idx=0):
    pr = urlparse(base_url(site, replica_idx))
    return f"{pr.scheme}://{pr.hostname}:{pr.port}"


def all_origins():
    return sorted({f"{urlparse(u).hostname}:{urlparse(u).port}"
                   for urls in REPLICAS.values() for u in urls})


def all_hosts():
    return sorted({urlparse(u).hostname for urls in REPLICAS.values() for u in urls})


def substitute(text, site_to_replica):
    for ph, site in PLACEHOLDER.items():
        if ph in text:
            text = text.replace(ph, base_url(site, site_to_replica.get(site, 0)))
    return text


def resolve_task(raw_task, site_to_replica):
    t = copy.deepcopy(raw_task)
    def walk(o):
        if isinstance(o, str):
            return substitute(o, site_to_replica)
        if isinstance(o, list):
            return [walk(x) for x in o]
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        return o
    return walk(t)


def auth_path(site, replica_idx=0):
    """hilbit2 replicas keep the legacy name ({site}_{port}.json, already generated);
    hilbit1 replicas use a host-tagged name."""
    pr = urlparse(base_url(site, replica_idx))
    if pr.hostname == H2:
        return os.path.join(AUTH_DIR, f"{site}_{pr.port}.json")
    return os.path.join(AUTH_DIR, f"{site}_{pr.hostname}_{pr.port}.json")


def load_raw_tasks(path=None):
    # WebArena raw task definitions (external data); override with WA_RAW_TASKS
    path = path or os.environ.get("WA_RAW_TASKS", "/home/cc/webarena/config_files/test.raw.json")
    return json.load(open(path))
