"""Fyers API v3 authentication: one-time interactive login + daily refresh-token automation.

Fyers access_tokens expire daily (~6 AM IST, exchange-mandated). The refresh_token issued
alongside it is valid ~15 days and can mint a new access_token without a browser login,
via the pin-gated /validate-refresh-token endpoint. `refresh` is the command meant to run
on a daily schedule (e.g. Windows Task Scheduler); `login` is a one-time (or ~15-day) step.
"""

import argparse
import base64
import hashlib
import json
import os
from datetime import date
from urllib.parse import urlparse, parse_qs

import pyotp
import requests

BASE_URL = "https://api-t1.fyers.in/api/v3"
LOGIN_API_BASE = "https://api-t2.fyers.in/vagator/v2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, ".secrets", "fyers_config.json")
TOKEN_PATH = os.path.join(ROOT, ".cache", "fyers_tokens.json")


def _st_secrets() -> dict:
    """Streamlit Community Cloud delivers secrets via st.secrets, not env vars or files."""
    try:
        import streamlit as st
        if not hasattr(st, "secrets") or not st.secrets:
            return {}
        # 1. Try [fyers] table
        if "fyers" in st.secrets:
            return dict(st.secrets["fyers"])
        # 2. Try [FYERS] table
        if "FYERS" in st.secrets:
            return dict(st.secrets["FYERS"])
        # 3. Check flat top-level keys
        flat = {}
        for k in ("client_id", "secret_key", "redirect_uri", "pin", "fy_id", "totp_secret"):
            if k in st.secrets:
                flat[k] = str(st.secrets[k])
            elif k.upper() in st.secrets:
                flat[k] = str(st.secrets[k.upper()])
        return flat
    except Exception:
        return {}


LOGIN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://myaccount.fyers.in",
    "Referer": "https://myaccount.fyers.in/",
    "Accept": "application/json, text/plain, */*",
}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def _load_config() -> dict:
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    st_cfg = _st_secrets()
    resolved = {
        "client_id": str(cfg.get("client_id") or st_cfg.get("client_id") or os.environ.get("FYERS_CLIENT_ID") or "").strip(),
        "secret_key": str(cfg.get("secret_key") or st_cfg.get("secret_key") or os.environ.get("FYERS_SECRET_KEY") or "").strip(),
        "redirect_uri": str(cfg.get("redirect_uri") or st_cfg.get("redirect_uri") or os.environ.get("FYERS_REDIRECT_URI") or "").strip(),
        "pin": str(cfg.get("pin") or st_cfg.get("pin") or os.environ.get("FYERS_PIN") or "").strip(),
        "fy_id": str(cfg.get("fy_id") or st_cfg.get("fy_id") or os.environ.get("FYERS_FY_ID") or "").strip(),
        "totp_secret": str(cfg.get("totp_secret") or st_cfg.get("totp_secret") or os.environ.get("FYERS_TOTP_SECRET") or "").strip().replace(" ", ""),
    }
    missing = [k for k, v in resolved.items() if not v and k not in ("pin", "fy_id", "totp_secret")]
    if missing:
        raise RuntimeError(
            f"Missing Fyers config: {missing}. Set them in {CONFIG_PATH} "
            f"(see .secrets/fyers_config.example.json), via st.secrets, or as env vars."
        )
    return resolved


def _app_id_hash(client_id: str, secret_key: str) -> str:
    return hashlib.sha256(f"{client_id}:{secret_key}".encode()).hexdigest()


def _get_token_path() -> str:
    try:
        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        test_file = os.path.join(os.path.dirname(TOKEN_PATH), ".write_test")
        with open(test_file, "w") as f:
            f.write("1")
        os.remove(test_file)
        return TOKEN_PATH
    except Exception:
        import tempfile
        return os.path.join(tempfile.gettempdir(), "fyers_tokens.json")


def _load_tokens() -> dict:
    t_path = _get_token_path()
    if os.path.exists(t_path):
        try:
            with open(t_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_tokens(tokens: dict) -> None:
    t_path = _get_token_path()
    os.makedirs(os.path.dirname(t_path), exist_ok=True)
    with open(t_path, "w") as f:
        json.dump(tokens, f, indent=2)


def build_login_url() -> str:
    cfg = _load_config()
    return (
        f"{BASE_URL}/generate-authcode"
        f"?client_id={cfg['client_id']}"
        f"&redirect_uri={cfg['redirect_uri']}"
        f"&response_type=code&state=fyers_auth"
    )


def _extract_auth_code(raw: str) -> str:
    """Accepts either a bare auth_code or the full URL the browser redirected to."""
    if raw.startswith("http"):
        query = parse_qs(urlparse(raw).query)
        code = query.get("auth_code") or query.get("code")
        if not code:
            raise ValueError("No auth_code/code found in the pasted URL.")
        return code[0]
    return raw.strip()


def login_with_auth_code(raw_code_or_url: str) -> dict:
    cfg = _load_config()
    code = _extract_auth_code(raw_code_or_url)
    resp = requests.post(
        f"{BASE_URL}/validate-authcode",
        json={
            "grant_type": "authorization_code",
            "appIdHash": _app_id_hash(cfg["client_id"], cfg["secret_key"]),
            "code": code,
        },
        headers=DEFAULT_HEADERS,
        timeout=10,
    )
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"Fyers auth-code exchange failed: {data}")
    tokens = {
        "access_token": data["access_token"],
        "refresh_token": data["refresh_token"],
        "obtained_date": date.today().isoformat(),
    }
    _save_tokens(tokens)
    return tokens


def auto_login() -> dict:
    """
    Fully non-interactive login using Fyers' internal TOTP-based login endpoints
    (used because the public refresh-token API is disabled per SEBI regulation).
    Requires fy_id, totp_secret, and pin in config. Not part of Fyers' documented
    public API, so it may need adjustment if Fyers changes these endpoints.
    """
    cfg = _load_config()
    for key in ("fy_id", "totp_secret", "pin"):
        if not cfg[key]:
            raise RuntimeError(f"auto_login requires '{key}' in config.")

    # Step 1: request login OTP (issues a TOTP-flow request_key)
    resp = requests.post(
        f"{LOGIN_API_BASE}/send_login_otp",
        json={"fy_id": cfg["fy_id"], "app_id": "2"},
        headers=LOGIN_HEADERS,
        timeout=10,
    )
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"send_login_otp failed: {data}")
    request_key = data["request_key"]

    # Step 2: verify a freshly generated TOTP code against that request_key
    totp_code = pyotp.TOTP(cfg["totp_secret"]).now()
    resp = requests.post(
        f"{LOGIN_API_BASE}/verify_otp",
        json={"request_key": request_key, "otp": totp_code},
        headers=LOGIN_HEADERS,
        timeout=10,
    )
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"verify_otp failed: {data}")
    request_key = data["request_key"]

    # Step 3: verify PIN, returns a short-lived session bearer token
    resp = requests.post(
        f"{LOGIN_API_BASE}/verify_pin",
        json={"request_key": request_key, "identity_type": "pin", "identifier": cfg["pin"]},
        headers=LOGIN_HEADERS,
        timeout=10,
    )
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"verify_pin failed: {data}")
    session_token = data["data"]["access_token"]

    # Step 4: exchange the session token for an auth_code via the app's OAuth endpoint
    app_id = cfg["client_id"].split("-")[0]
    resp = requests.post(
        f"{BASE_URL}/token",
        json={
            "fyers_id": cfg["fy_id"],
            "app_id": app_id,
            "redirect_uri": cfg["redirect_uri"],
            "appType": "100",
            "code_challenge": "",
            "state": "fyers_auth",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True,
        },
        headers={"Authorization": f"Bearer {session_token}", **DEFAULT_HEADERS},
        timeout=10,
    )
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"token exchange failed: {data}")
    auth_code = _extract_auth_code(data["Url"])

    return login_with_auth_code(auth_code)


def refresh_access_token() -> str:
    cfg = _load_config()
    tokens = _load_tokens()
    if not tokens.get("refresh_token"):
        raise RuntimeError("No refresh_token on file - run `login` once first.")
    if not cfg["pin"]:
        raise RuntimeError("FYERS_PIN not set - required for the refresh-token flow.")
    resp = requests.post(
        f"{BASE_URL}/validate-refresh-token",
        json={
            "grant_type": "refresh_token",
            "appIdHash": _app_id_hash(cfg["client_id"], cfg["secret_key"]),
            "refresh_token": tokens["refresh_token"],
            "pin": cfg["pin"],
        },
        timeout=10,
    )
    data = resp.json()
    if data.get("s") != "ok":
        raise RuntimeError(f"Fyers refresh-token exchange failed: {data}")
    tokens["access_token"] = data["access_token"]
    tokens["obtained_date"] = date.today().isoformat()
    _save_tokens(tokens)
    return tokens["access_token"]


def get_access_token() -> str:
    """Returns today's valid access_token, auto-logging in (TOTP flow) if not minted yet today."""
    tokens = _load_tokens()
    if tokens.get("access_token") and tokens.get("obtained_date") == date.today().isoformat():
        return tokens["access_token"]
    auto_login()
    return _load_tokens()["access_token"]


def main():
    parser = argparse.ArgumentParser(description="Fyers API v3 auth helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("login-url", help="Print the one-time interactive login URL")

    p_login = sub.add_parser("login", help="Exchange an auth_code (or full redirect URL) for tokens")
    p_login.add_argument("auth_code")

    sub.add_parser("auto-login", help="Fully non-interactive TOTP-based login (run daily, e.g. via Task Scheduler)")

    sub.add_parser("token", help="Print a valid access_token, auto-logging in first if needed")

    args = parser.parse_args()

    if args.command == "login-url":
        print(build_login_url())
    elif args.command == "login":
        login_with_auth_code(args.auth_code)
        print(f"Logged in. Tokens saved to {TOKEN_PATH}")
    elif args.command == "auto-login":
        auto_login()
        print(f"Auto-login succeeded. New access_token obtained ({date.today().isoformat()}).")
    elif args.command == "token":
        print(get_access_token())


if __name__ == "__main__":
    main()
