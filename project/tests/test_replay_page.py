"""Replay page assembly: offline three.js + body meshes inlined after the main script."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from tennis_vision.shot_replay import write_replay_page

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "tennis_vision" / "vendor" / "three-0.159.0.min.js"


class ReplayPageTest(unittest.TestCase):
    def test_page_is_single_offline_file_with_legacy_script_first(self):
        data = {"report": {"note": "</script> inside data must stay escaped"}, "frames": [], "marker": "__THREE_JS__"}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "replay-data.json").write_text(json.dumps(data), encoding="utf-8")
            write_replay_page(out)
            html = (out / "replay.html").read_text(encoding="utf-8")
        scripts = re.findall(r"<script\b([^>]*)>([\s\S]*?)</script>", html)
        ids = [re.search(r'id="([^"]+)"', attrs).group(1) if 'id="' in attrs else None for attrs, _ in scripts]
        # JSON data, then the legacy UI script (what the fake-DOM tests exercise),
        # then vendored three.js, then the body-mesh adapter.
        self.assertEqual(ids, ["replay-data", None, "three-vendor", "body-meshes"])
        self.assertEqual(json.loads(scripts[0][1]), data)  # Placeholder text in data is untouched.
        self.assertIn("meshRenderer=null", scripts[1][1])
        self.assertIn("meshRenderer={enabled", scripts[3][1])
        self.assertNotIn("src=", "".join(attrs for attrs, _ in scripts))  # No external/CDN script.
        self.assertNotIn("__BODY_MESHES_JS__", html)

    def test_vendored_three_matches_recorded_hash(self):
        self.assertEqual(hashlib.sha256(VENDOR.read_bytes()).hexdigest(),
                         "7b1c5d75b28d9de15042e2b374f83566d8c7146697af8fdeb4558b0fb528a585")


if __name__ == "__main__":
    unittest.main()
