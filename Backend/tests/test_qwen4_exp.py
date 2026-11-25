import unittest
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from local_llm.mlxwrap import _vendor_models

_vendor_models()
from mlx_lm.models.qwen4_exp import Model, NGramEmbedding, TextArgs


class _Indices(nn.Module):
    def __call__(self, indices):
        return indices


class Qwen4ExpTests(unittest.TestCase):
    def test_ngram_hash_rebuilds_each_ngram_like_reference(self):
        args = TextArgs(
            vocab_size=1000,
            eos_token_id=0,
            ngram_size=3,
            heads_per_ngram=1,
            ngram_vocab_size_base=19,
            make_ngram_vocab_size_divisible_by=1,
            split_ngram_parts=2,
        )
        table = NGramEmbedding(args, embed_dim=2)
        table.ngram_embedding = _Indices()

        actual = table(mx.array([[5, 6, 7]]), mx.array([[0, 0]]))
        mults = [int(v) for v in table._mults]
        sizes = [int(v) for v in table._sizes]
        offsets = [int(v) for v in table._offsets]
        expected = [
            ((7 * mults[0]) ^ (6 * mults[1])) % sizes[0] + offsets[0],
            ((7 * mults[0]) ^ (6 * mults[1]) ^ (5 * mults[2])) % sizes[1]
            + offsets[1],
        ]
        self.assertEqual(np.array(actual)[0, -1].tolist(), expected)

    def test_mlx_vlm_layout_folds_zero_centered_norms_once(self):
        model = SimpleNamespace(
            _FOLD_ONE=Model._FOLD_ONE,
            _NGRAM_CONSTANTS=Model._NGRAM_CONSTANTS,
        )
        norm = "language_model.model.layers.0.ple.norm_key.weight"
        marker = (
            "language_model.model.layers.0.ple.ple_embedding."
            "ngram_embedding.shards.0.weight"
        )
        out = Model.sanitize(
            model,
            {norm: mx.array([-0.25, 0.5]), marker: mx.array([[1.0]])},
        )
        self.assertEqual(out["model.layers.0.ple.norm_key.weight"].tolist(), [0.75, 1.5])

        converted = "model.layers.0.ple.norm_key.weight"
        out = Model.sanitize(model, {converted: mx.array([0.75, 1.5])})
        self.assertEqual(out[converted].tolist(), [0.75, 1.5])


if __name__ == "__main__":
    unittest.main()
