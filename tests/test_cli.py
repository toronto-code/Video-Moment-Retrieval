import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CLITests(unittest.TestCase):
    def test_demo_search_evaluate_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = [sys.executable, "-m", "video_moment_retrieval", "--data-dir", tmp,
                    "--env-file", str(root / "missing.env")]
            demo = subprocess.run([*base, "demo"], capture_output=True, text=True, check=True)
            self.assertTrue(json.loads(demo.stdout)["synthetic"])
            search = subprocess.run([*base, "--demo", "search", "Find moments being handcuffed",
                                     "--verify-budget", "11"], capture_output=True, text=True, check=True)
            self.assertEqual([r["record_id"] for r in json.loads(search.stdout)["results"]], ["cuffs_apply"])
            output = root / "evaluation.json"
            subprocess.run([*base, "--demo", "evaluate", str(root / "demo-labels.json"),
                            "--output", str(output)], capture_output=True, text=True, check=True)
            report = json.loads(output.read_text())
            self.assertEqual(len(report["policies"]), 7)
            self.assertEqual(report["query_count"], 6)

    def test_missing_key_fails_clearly_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env.pop("OPENROUTER_API_KEY", None)
            result = subprocess.run([sys.executable, "-m", "video_moment_retrieval", "--data-dir", tmp,
                "--env-file", str(Path(tmp)/"absent.env"), "search", "red shirt"],
                capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 2)
            self.assertIn("OPENROUTER_API_KEY", json.loads(result.stderr)["error"])
