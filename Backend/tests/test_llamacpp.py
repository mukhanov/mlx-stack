import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_llm import llamacpp


class LlamaDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="mlx-stack-llama-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        payload = b"test llama-server\n"
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tf:
            info = tarfile.TarInfo("llama-bTEST/llama-server")
            info.size = len(payload)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(payload))
        self.payload = payload
        self.archive = archive.getvalue()

    def response(self):
        response = io.BytesIO(self.archive)
        response.headers = {"content-length": str(len(self.archive))}
        return response

    def test_a_verified_archive_is_extracted_and_pinned(self):
        digest = hashlib.sha256(self.archive).hexdigest()
        with (patch.object(llamacpp, "_LLAMA_TAG", "bTEST"),
              patch.object(llamacpp, "_LLAMA_SHA256", digest),
              patch.object(llamacpp, "_fetch", lambda url, timeout: self.response())):
            server = Path(llamacpp._download_build(self.root))
        self.assertEqual(server.read_bytes(), self.payload)
        self.assertEqual(server.stat().st_mode & 0o111, 0o111)
        self.assertEqual((self.root / "build.txt").read_text(), "bTEST\n")

    def test_a_checksum_mismatch_is_not_extracted(self):
        with (patch.object(llamacpp, "_LLAMA_TAG", "bTEST"),
              patch.object(llamacpp, "_LLAMA_SHA256", "0" * 64),
              patch.object(llamacpp, "_fetch", lambda url, timeout: self.response())):
            with self.assertRaisesRegex(SystemExit, "sha256 mismatch"):
                llamacpp._download_build(self.root)
        self.assertFalse((self.root / "llama-bTEST/llama-server").exists())
        self.assertFalse((self.root / "llama-bTEST.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()
