"""Generate per-(site, replica) storage_state cookies by logging into the localhost sites.

Each replica is an independent container with its own server-side session, so we must
log into each one separately. Cookies ignore port, but sessions are per-container, so we
save one storage_state per replica port.
"""
import os, sys, json, time, argparse
from playwright.sync_api import sync_playwright
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wa_config as C

os.makedirs(C.AUTH_DIR, exist_ok=True)


def login_shopping(page, url, acc):
    page.goto(f"{url}/customer/account/login/", wait_until="domcontentloaded")
    page.get_by_label("Email", exact=True).fill(acc["username"])
    page.get_by_label("Password", exact=True).fill(acc["password"])
    page.get_by_role("button", name="Sign In").click()
    page.wait_for_load_state("domcontentloaded"); time.sleep(2)


def login_reddit(page, url, acc):
    page.goto(f"{url}/login", wait_until="domcontentloaded")
    page.get_by_label("Username").fill(acc["username"])
    page.get_by_label("Password").fill(acc["password"])
    page.get_by_role("button", name="Log in").click()
    page.wait_for_load_state("domcontentloaded"); time.sleep(2)


def login_shopping_admin(page, url, acc):
    # url already ends with /admin
    page.goto(url, wait_until="domcontentloaded")
    page.get_by_placeholder("user name").fill(acc["username"])
    page.get_by_placeholder("password").fill(acc["password"])
    page.get_by_role("button", name="Sign in").click()
    page.wait_for_load_state("domcontentloaded"); time.sleep(3)


def login_gitlab(page, url, acc):
    page.goto(f"{url}/users/sign_in", wait_until="domcontentloaded")
    page.get_by_test_id("username-field").fill(acc["username"])
    page.get_by_test_id("password-field").fill(acc["password"])
    page.get_by_test_id("sign-in-button").click()
    page.wait_for_load_state("domcontentloaded"); time.sleep(3)


VERIFY = {
    "shopping":       lambda url: f"{url}/customer/account/",
    "reddit":         lambda url: f"{url}/",
    "shopping_admin": lambda url: f"{url}/admin/dashboard/",
    "gitlab":         lambda url: f"{url}/-/profile",
}
LOGIN = {"shopping": login_shopping, "reddit": login_reddit,
         "shopping_admin": login_shopping_admin, "gitlab": login_gitlab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(LOGIN.keys()))
    ap.add_argument("--replicas", type=int, default=3)
    args = ap.parse_args()
    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"],
                                    proxy={"server": os.environ.get("AUTH_PROXY", "http://127.0.0.1:18900")})
        for site in args.sites:
            acc = C.ACCOUNTS[site]
            nrep = min(args.replicas, len(C.PORTS[site]))
            for ri in range(nrep):
                url = C.base_url(site, ri)
                out = C.auth_path(site, ri)
                tag = f"{site}[{C.PORTS[site][ri]}@{url.split('//')[1].split(':')[0]}]"
                if os.path.exists(out) and os.environ.get("AUTH_FORCE") != "1":
                    results[tag] = f"SKIP exists -> {out}"
                    continue
                try:
                    ctx = browser.new_context()
                    page = ctx.new_page()
                    page.set_default_navigation_timeout(120000)
                    page.set_default_timeout(60000)
                    LOGIN[site](page, url, acc)
                    ctx.storage_state(path=out)
                    ncookies = len(json.load(open(out)).get("cookies", []))
                    results[tag] = f"OK cookies={ncookies} -> {out}"
                    ctx.close()
                except Exception as e:
                    results[tag] = f"FAIL {type(e).__name__}: {str(e)[:160]}"
        browser.close()
    for k in sorted(results):
        print(k, results[k])


if __name__ == "__main__":
    main()
