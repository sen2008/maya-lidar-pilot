#!/usr/bin/env python3
"""Push files to sen2008/maya-lidar-pilot via the GitHub git-database API.

Usage:
    gh_push.py -m "commit message" <path> [<path> ...]
Paths are relative to ~/workspace/lidar-pilot/.

Flow: get main ref -> get base tree -> create blobs -> create tree ->
create commit -> PATCH ref. Verifies the ref afterwards.

Safety: refuses files > 5 MB and any *.tif / *.laz / *.las under data/
(raw LiDAR must never be pushed).
"""
import argparse
import base64
import json
import os
import subprocess
import sys

HELPER = os.path.expanduser(
    "~/workspace/custom-connectors/custom.legacy-github/bin/github_api.py")
REPO = "sen2008/maya-lidar-pilot"
ROOT = os.path.expanduser("~/workspace/lidar-pilot")
MAX_BYTES = 5 * 1024 * 1024


def api(method, path, body=None):
    cmd = [sys.executable, HELPER, method, path]
    if body is not None:
        with open("/tmp/gh_push_body.json", "w") as fh:
            json.dump(body, fh)
        cmd += ["--body-file", "/tmp/gh_push_body.json"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout) if out.stdout.strip() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--message", required=True)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    paths = []
    for rel in args.files:
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            print(f"missing: {rel}", file=sys.stderr)
            return 1
        size = os.path.getsize(full)
        if size > MAX_BYTES:
            print(f"refusing {rel}: {size} bytes > 5 MB cap", file=sys.stderr)
            return 1
        if rel.startswith("data/") and rel.endswith((".tif", ".tiff", ".laz", ".las")):
            print(f"refusing {rel}: raw LiDAR data must not be pushed", file=sys.stderr)
            return 1
        paths.append((rel, full))

    ref = api("GET", f"/repos/{REPO}/git/refs/heads/main")
    commit_sha = ref["object"]["sha"]
    commit = api("GET", f"/repos/{REPO}/git/commits/{commit_sha}")
    base_tree = commit["tree"]["sha"]
    print(f"base commit {commit_sha[:8]} tree {base_tree[:8]}")

    entries = []
    for rel, full in paths:
        with open(full, "rb") as fh:
            raw = fh.read()
        try:
            text = raw.decode("utf-8")
            blob = api("POST", f"/repos/{REPO}/git/blobs",
                       {"content": text, "encoding": "utf-8"})
        except UnicodeDecodeError:
            blob = api("POST", f"/repos/{REPO}/git/blobs",
                       {"content": base64.b64encode(raw).decode(),
                        "encoding": "base64"})
        print(f"blob {rel} -> {blob['sha'][:8]}")
        entries.append({"path": rel, "mode": "100644",
                        "type": "blob", "sha": blob["sha"]})

    tree = api("POST", f"/repos/{REPO}/git/trees",
               {"base_tree": base_tree, "tree": entries})
    print(f"tree {tree['sha'][:8]}")
    new_commit = api("POST", f"/repos/{REPO}/git/commits",
                     {"message": args.message, "tree": tree["sha"],
                      "parents": [commit_sha]})
    print(f"commit {new_commit['sha'][:8]}")
    api("PATCH", f"/repos/{REPO}/git/refs/heads/main",
        {"sha": new_commit["sha"]})
    check = api("GET", f"/repos/{REPO}/git/refs/heads/main")
    assert check["object"]["sha"] == new_commit["sha"], "ref verify failed"
    print(f"main now at {new_commit['sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
