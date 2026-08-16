"""Fyers API v3 authentication: one-time interactive login + daily refresh-token automation.

Fyers access_tokens expire daily (~6 AM IST, exchange-mandated). The refresh_token issued
alongside it is valid ~15 days and can mint a new access_token without a browser login,
via the pin-gated /validate-refresh-token endpoint. `refresh` is the command meant to run
on a daily schedule (e.g. Windows Task Scheduler); `login` is a one-time (or ~15-day) step.
"""

import argparse
import hashlib
import json
import os
from datetime import date
from urllib.parse import urlparse, parse_qs

import requests

BASE_URL = "https://api-t1.fyers.in/api/v3"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, ".secrets", "fyers_config.json")
TOKEN_PATH = os.path.join(ROOT, ".cache", "fyers_tokens.json")


def _load_config() -> dict:
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
    resolved = {
        "client_id": cfg.get("client_id") or os.environ.get("FYERS_CLIENT_ID"),
        "secret_key": cfg.get("secret_key") or os.environ.get("FYERS_SECRET_KEY"),
        "redirect_uri": cfg.get("redirect_uri") or os.environ.get("FYERS_REDIRECT_URI"),
        "pin": cfg.get("pin") or os.environ.get("FYERS_PIN"),
    }
    missing = [k for k, v in resolved.items() if not v and k != "pin"]
    if missing:
        raise RuntimeError(
            f"Missing Fyers config: {missing}. Set them in {CONFIG_PATH} "
            f"(see .secrets/fyers_config.example.json) or as env vars."
        )
    return resolved


def _app_id_hash(client_id: str, secret_key: str) -> str:
    return hashlib.sha256(f"{client_id}:{secret_key}".encode()).hexdigest()


def _load_tokens() -> dict:
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_tokens(tokens: dict) -> None:
    os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
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
    """Returns today's valid access_token, refreshing automatically if not minted yet today."""
    tokens = _load_tokens()
    if tokens.get("access_token") and tokens.get("obtained_date") == date.today().isoformat():
        return tokens["access_token"]
    return refresh_access_token()


def main():
    parser = argparse.ArgumentParser(description="Fyers API v3 auth helper")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("login-url", help="Print the one-time interactive login URL")

    p_login = sub.add_parser("login", help="Exchange an auth_code (or full redirect URL) for tokens")
    p_login.add_argument("auth_code")

    sub.add_parser("refresh", help="Mint a new access_token from the stored refresh_token (run daily)")

    sub.add_parser("token", help="Print a valid access_token, refreshing first if needed")

    args = parser.parse_args()

    if args.command == "login-url":
        print(build_login_url())
    elif args.command == "login":
        login_with_auth_code(args.auth_code)
        print(f"Logged in. Tokens saved to {TOKEN_PATH}")
    elif args.command == "refresh":
        refresh_access_token()
        print(f"Refreshed. New access_token obtained ({date.today().isoformat()}).")
    elif args.command == "token":
        print(get_access_token())


if __name__ == "__main__":
    main()
