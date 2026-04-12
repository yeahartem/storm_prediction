#!/usr/bin/env python3
"""
Download CMIP6 MRI-ESM2-0 data for wind prediction project.
Downloads historical (2000-2014) + ssp245 (2015-2064) for:
  sfcWindmax, pr, tasmax, tasmin, psl

Usage: python download_cmip6.py --out data/cmip6_world_orig
"""

import argparse
import os
import sys
import time
import requests
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry

ESGF_URL = "https://esgf-node.llnl.gov/esg-search/search"

# Files we need: (experiment, variable, time_start_filter)
TARGETS = [
    ("historical", "psl",        "2000"),
    ("ssp245",     "psl",        "2015"),
]


def make_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0,
                    status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://",  HTTPAdapter(max_retries=retries))
    return s


SKIP_HOSTS = ["diasjp.net"]   # nodes known to drop connections

def find_file(session, experiment, variable, time_filter):
    """Query ESGF for the target file; return all usable HTTP URLs ranked by preference."""
    params = {
        "project": "CMIP6",
        "source_id": "MRI-ESM2-0",
        "experiment_id": experiment,
        "variable": variable,
        "frequency": "day",
        "variant_label": "r1i1p1f1",
        "type": "File",
        "latest": "true",
        "limit": 50,
        "format": "application/solr+json",
    }
    r = session.get(ESGF_URL, params=params, timeout=30)
    r.raise_for_status()
    docs = r.json()["response"]["docs"]

    # Collect URLs from ALL docs (each mirror is a separate doc)
    matched_title = None
    matched_size = 0
    all_http_urls = []

    for doc in docs:
        title = doc.get("title", "")
        if time_filter not in title:
            continue
        if matched_title is None:
            matched_title = title
            matched_size = doc.get("size", 0)
        raw_urls = doc.get("url", [])
        for u in raw_urls:
            parts = u.split("|")
            if "http" in parts[-1].lower() and parts[0].startswith("http"):
                all_http_urls.append(parts[0])

    if not matched_title:
        return None, [], 0

    def rank(u):
        bad = any(h in u for h in SKIP_HOSTS)
        globus = "globus.org" in u
        return (bad, globus, not u.startswith("https"))

    all_http_urls.sort(key=rank)
    print(f"  Found {len(all_http_urls)} mirrors")
    return matched_title, all_http_urls, matched_size


def download_file(session, url, dest, expected_size):
    """Stream-download with progress bar. Skip if already complete."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    existing = dest.stat().st_size if dest.exists() else 0
    if expected_size and existing >= expected_size:
        print(f"  Already complete ({existing // 1024 // 1024} MB), skipping.")
        return True

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"

    print(f"  Downloading: {url}")
    print(f"  To: {dest}")

    try:
        with session.get(url, stream=True, timeout=(30, 600), headers=headers) as resp:
            if resp.status_code == 416:  # Range not satisfiable → already done
                print("  Already complete (server says so).")
                return True
            resp.raise_for_status()

            total = int(resp.headers.get("Content-Length", 0)) + existing
            downloaded = existing
            t0 = time.time()

            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = time.time() - t0 or 0.001
                        speed = (downloaded - existing) / elapsed / 1024 / 1024
                        pct = downloaded / total * 100 if total else 0
                        print(f"\r  {pct:5.1f}%  {downloaded//1024//1024} / {total//1024//1024} MB"
                              f"  {speed:.1f} MB/s   ", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"\n  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cmip6_world_orig",
                        help="Output directory for downloaded NC files")
    args = parser.parse_args()

    session = make_session()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir.resolve()}\n")

    results = []
    for experiment, variable, time_filter in TARGETS:
        print(f"[{experiment} / {variable}]")
        title, urls, size = find_file(session, experiment, variable, time_filter)
        if not title:
            print(f"  WARNING: file not found on ESGF!\n")
            continue
        dest = out_dir / title
        size_mb = size // 1024 // 1024
        print(f"  File: {title} ({size_mb} MB)")
        ok = False
        for url in urls:
            print(f"  Trying: {url}")
            ok = download_file(session, url, dest, size)
            if ok:
                break
            print(f"  Failed, trying next mirror...")
        results.append((title, ok))
        print()

    print("\n=== Summary ===")
    for title, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {title}")


if __name__ == "__main__":
    main()
