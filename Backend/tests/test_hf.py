"""The Hugging Face layer that is pure logic: what one repository listing
answers, and how much of a model is on disk."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm import hf

GB = 1 << 30
INFO = {"downloads": 7, "likes": 3, "siblings": [
    {"rfilename": "model-00001-of-00002.safetensors", "size": 5 * GB,
     "lfs": {"sha256": "aaa"}, "blobId": "b1"},
    {"rfilename": "model-00002-of-00002.safetensors", "size": 2 * GB,
     "lfs": {"sha256": "bbb"}},
    {"rfilename": "tokenizer.json", "size": 2 << 20, "blobId": "c3"},
    {"rfilename": "Q4_K_M/model-00001-of-00002.gguf", "size": 4 * GB,
     "lfs": {"sha256": "4a1"}},
    {"rfilename": "Q4_K_M/model-00002-of-00002.gguf", "size": 1 * GB,
     "lfs": {"sha256": "4a2"}},
    {"rfilename": "Q8_0/model.gguf", "size": 8 * GB, "lfs": {"sha256": "8a0"}},
    {"rfilename": "mmproj-f16.gguf", "size": 1 * GB, "lfs": {"sha256": "ddd"}},
    {"rfilename": "mtp-Qwen.gguf", "size": 1 * GB, "lfs": {"sha256": "eee"}},
]}


class RepositoryListingTests(unittest.TestCase):
    def test_one_listing_answers_size_model_files_and_resumable_shards(self):
        with patch.object(hf, "_repo_info", return_value=INFO) as info:
            self.assertEqual(hf.repo_size("org/repo", None),
                             (5 + 2 + 4 + 1 + 8 + 1 + 1) * GB + (2 << 20))
            self.assertEqual(hf.fetch_stats("org/repo", None),
                             {"downloads": 7, "likes": 3,
                              "size_bytes": (5 + 2 + 4 + 1 + 8 + 1 + 1) * GB + (2 << 20)})
            self.assertEqual(
                [name for name, _bytes in hf.repo_gguf_files("org/repo", None)],
                ["Q4_K_M/model-00001-of-00002.gguf", "Q4_K_M/model-00002-of-00002.gguf",
                 "Q8_0/model.gguf"])
        self.assertEqual(info.call_count, 3)

    def test_ranged_fetch_takes_the_big_lfs_files_of_the_chosen_quant_only(self):
        everything = hf._ranged_files(INFO, None)
        self.assertEqual([name for name, _size, _sha in everything][:3], [
            "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
            "Q4_K_M/model-00001-of-00002.gguf"])
        self.assertEqual(hf._ranged_files(INFO, ["Q8_0/model.gguf"]),
                         [("Q8_0/model.gguf", 8 * GB, "8a0")])
        self.assertEqual(hf._ranged_files(INFO, ["tokenizer.json"]), [])

    def test_an_unreachable_hub_is_an_empty_answer_not_a_crash(self):
        with patch.object(hf, "_repo_info", return_value=None):
            self.assertEqual(hf.repo_size("org/repo", None), 0)
            self.assertIsNone(hf.repo_gguf_files("org/repo", None))
            self.assertIsNone(hf.missing_files(Path("hub"), "org/repo", None))
            with self.assertRaises(OSError):
                hf.fetch_stats("org/repo", None)

    def test_missing_files_are_what_the_listing_has_and_the_snapshot_lacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp)
            snap = hub / "models--org--repo" / "snapshots" / "rev" / "sub"
            snap.mkdir(parents=True)
            (snap / "model.gguf").write_bytes(b"GGUF")
            (hub / "models--org--repo" / "snapshots" / "rev" / "gone.gguf").symlink_to(
                hub / "nowhere")
            with patch.object(hf, "_repo_info", return_value=INFO):
                self.assertEqual(
                    hf.missing_files(hub, "org/repo", None),
                    ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "tokenizer.json", "Q4_K_M/model-00001-of-00002.gguf",
                     "Q4_K_M/model-00002-of-00002.gguf", "Q8_0/model.gguf",
                     "mmproj-f16.gguf", "mtp-Qwen.gguf"])


class GGUFTests(unittest.TestCase):
    def test_shards_group_into_quants_largest_first_with_complete_sets_first(self):
        groups = hf.gguf_groups([("Q4_K_M/m-Q4_K_M-00002-of-00002.gguf", 1),
                                 ("Q8_0/m-Q8_0.gguf", 8),
                                 ("Q4_K_M/m-Q4_K_M-00001-of-00002.gguf", 4)])
        self.assertEqual([(label, total) for label, total, _names in groups],
                         [("m-Q8_0.gguf", 8), ("m-Q4_K_M.gguf", 5)])
        self.assertEqual(groups[1][2], ["Q4_K_M/m-Q4_K_M-00001-of-00002.gguf",
                                        "Q4_K_M/m-Q4_K_M-00002-of-00002.gguf"])

    def test_a_shard_set_without_its_first_split_is_not_servable(self):
        split = ["d/model-00002-of-00003.gguf"]
        self.assertFalse(hf.gguf_set_complete(split))
        self.assertTrue(hf.gguf_set_complete(
            ["d/model-00001-of-00003.gguf", "d/model-00002-of-00003.gguf"]))
        self.assertTrue(hf.gguf_set_complete(["d/model.gguf"]))

    def test_side_files_are_not_the_model(self):
        for name in ("org/mmproj-f16.gguf", "imatrix.dat", "x/Qwen3-mtp-Q8.gguf",
                     "mtp-head.gguf"):
            with self.subTest(name=name):
                self.assertFalse(hf._is_model_gguf(name))
        self.assertTrue(hf._is_model_gguf("Q4_K_M/model-00001-of-00002.gguf"))


class DiskUsageTests(unittest.TestCase):
    def test_a_model_counts_its_own_and_shared_blobs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp)
            model = hub / "models--org--repo"
            own = model / "blobs" / "own"
            shared = hub / "blobs" / "ab" / "shared"
            snap = model / "snapshots" / "rev"
            own.parent.mkdir(parents=True)
            shared.parent.mkdir(parents=True)
            snap.mkdir(parents=True)
            own.write_bytes(b"x" * 100)
            shared.write_bytes(b"y" * 300)
            (snap / "config.json").write_text("{}")              # 2 bytes, regular
            (snap / "own.safetensors").symlink_to(own)           # inside: counted above
            (snap / "shared.safetensors").symlink_to(shared)     # the shared store
            (snap / "shared-copy.safetensors").symlink_to(shared)
            (snap / "broken").symlink_to(hub / "nowhere")        # not there
            self.assertEqual(hf.model_disk_bytes(hub, "org/repo"), 402)
        self.assertEqual(hf.model_disk_bytes(Path("hub"), "org/absent"), 0)


if __name__ == "__main__":
    unittest.main()
