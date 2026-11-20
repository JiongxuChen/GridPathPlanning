"""CPU checks for the isolated learned-offset experimental head."""
from pathlib import Path
import sys

import torch
from transformers import T5Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "analysis/relational_set_v1"))
from gridpath.planning.protocol import feature as original_feature
from gridpath.types import Grid
from set_model import OffsetSetHead, SetStepVerifier, collate, feature


class MarkerTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=False):
        pieces = text.replace("<extra_id_0>", " @ ").replace(
            "<extra_id_1>", " # ").split()
        return {"input_ids": [10 if x == "@" else 11 if x == "#" else 12
                              for x in pieces]}

    def convert_tokens_to_ids(self, text):
        return {"<extra_id_0>": 10, "<extra_id_1>": 11}[text]


def test_raw_offsets_and_original_text_unchanged():
    tokenizer = MarkerTokenizer()
    grid = Grid(6, 6, frozenset({(2, 3), (4, 1)}), (2, 1), (2, 4))
    old = original_feature(tokenizer, grid, ("right", "right"), 2048)
    new = feature(tokenizer, grid, ("right", "right"), 2048)
    for key in old:
        assert new[key] == old[key]
    assert new["obstacle_offsets"] == [
        [[0, 1], [2, -1]], [[0, 0], [2, -2]], []]


def test_set_head_permutation_and_empty_mask():
    head = OffsetSetHead(16, 0).eval()
    hidden = torch.randn(1, 2, 16)
    offsets = torch.tensor([[[[0., 0.], [2., -1.], [-1., 3.]],
                             [[0., 0.], [0., 0.], [0., 0.]]]])
    mask = torch.tensor([[[True, True, True], [False, False, False]]])
    first = head(hidden, offsets, mask)
    permuted = head(hidden, offsets[:, :, [2, 0, 1]], mask[:, :, [2, 0, 1]])
    assert torch.allclose(first, permuted, atol=1e-6)
    assert torch.isfinite(first).all()
    changed = offsets.clone()
    changed[0, 0, 0] = torch.tensor([5., 5.])
    assert not torch.allclose(first[0, 0], head(hidden, changed, mask)[0, 0])


def test_set_model_collate_and_roundtrip(tmp_path):
    cfg = T5Config(vocab_size=40, d_model=16, d_kv=4, d_ff=32,
                   num_layers=1, num_heads=4, dropout_rate=0.)
    model = SetStepVerifier(cfg).eval()
    features = [
        {"input_ids": [2, 3, 4], "positions": [0, 2], "kinds": [0, 1],
         "labels": [1, 4], "obstacle_offsets": [[[0, 0], [1, -1]], []]},
        {"input_ids": [2, 3], "positions": [0, 1], "kinds": [0, 1],
         "labels": [0, 4], "obstacle_offsets": [[], []]},
    ]
    inputs, labels = collate(features, 0, torch.device("cpu"))
    logits = model(**inputs)
    assert logits.shape == (2, 2, 6) and labels.shape == (2, 2)
    assert torch.all(logits[:, 0, 4:] < -1000)
    assert torch.all(logits[:, 1, :4] < -1000)
    model.save_pretrained(tmp_path / "set", safe_serialization=True)
    loaded = SetStepVerifier.from_pretrained(tmp_path / "set").eval()
    assert torch.equal(loaded(**inputs), logits)
