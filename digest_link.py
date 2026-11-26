"""Find this week's digest on Google Groups and save its link.

Runs after send_digest.py. send_digest.py sets the email's Message-ID itself and
saves it in digest_post.json. Google Groups serves every archived message at
https://groups.google.com/d/msgid/<group>/<Message-ID>, so we wait until that
page exists, then add "link" to digest_post.json for the Bluesky and feed steps.
"""
import json
import os
import time
from urllib.parse import quote

import requests

GROUP = "intelarchive"
GROUP_URL = f"https://groups.google.com/g/{GROUP}"
HANDOFF_FILE = "digest_post.json"
CHECK_ATTEMPTS = 6        # 6 checks x 5 minutes = up to 30 minutes
CHECK_INTERVAL = 5 * 60
TIMEOUT = 15


def find_link(message_id: str) -> str:
    url = f"https://groups.google.com/d/msgid/{GROUP}/{quote(message_id, safe='')}"
    for attempt in range(1, CHECK_ATTEMPTS + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, allow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (compatible; IntelArchiveBot/1.0)"})
            if r.status_code == 200:
                print(f"Digest archived: {r.url}")
                # Prefer the clean thread URL if Groups redirected to it
                return r.url if "/c/" in r.url else url
            print(f"Attempt {attempt}: Google Groups returned {r.status_code}")
        except requests.RequestException as e:
            print(f"Attempt {attempt}: {e}")
        if attempt < CHECK_ATTEMPTS:
            time.sleep(CHECK_INTERVAL)
    print("Could not confirm the archived message; using the group page instead.")
    return GROUP_URL


def main():
    if not os.path.exists(HANDOFF_FILE):
        print("No digest was sent this run.")
        return
    with open(HANDOFF_FILE) as f:
        info = json.load(f)
    info["link"] = find_link(info["message_id"])
    with open(HANDOFF_FILE, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Digest link: {info['link']}")


if __name__ == "__main__":
    main()
