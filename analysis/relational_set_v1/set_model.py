"""Learned permutation-invariant offset-set head for the original T5 step V."""
import math

import torch
from torch import nn

from gridpath.planning.model import StepVerifier, collate as base_collate
from gridpath.planning.protocol import TAGS, feature as base_feature, virtual_states


def feature(tokenizer, grid, candidate, limit, labels=None):
    result = base_feature(tokenizer, grid, candidate, limit, labels)
    states = virtual_states(grid.start, candidate)
    obstacles = sorted(grid.obstacles)
    result["obstacle_offsets"] = [
        [[r - after[0], c - after[1]] for r, c in obstacles]
        for after in states[1:]
    ] + [[]]
    assert len(result["obstacle_offsets"]) == len(result["positions"])
    return result


def collate(features, pad_id, device):
    inputs, labels = base_collate(features, pad_id, device)
    batch, queries = inputs["positions"].shape
    count = max(1, *(len(offsets) for row in features
                     for offsets in row["obstacle_offsets"]))
    offsets = torch.zeros((batch, queries, count, 2),
                          dtype=torch.float32, device=device)
    mask = torch.zeros((batch, queries, count), dtype=torch.bool, device=device)
    for i, row in enumerate(features):
        assert len(row["obstacle_offsets"]) == len(row["positions"])
        for j, pairs in enumerate(row["obstacle_offsets"]):
            if pairs:
                offsets[i, j, :len(pairs)] = torch.tensor(
                    pairs, dtype=torch.float32, device=device)
                mask[i, j, :len(pairs)] = True
    inputs["obstacle_offsets"] = offsets
    inputs["obstacle_mask"] = mask
    return inputs, labels


class OffsetSetHead(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU())
        self.query = nn.Linear(width, 64)
        self.project = nn.Linear(64, width)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width, len(TAGS))
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, offsets, mask):
        # Scaling is invertible and supplies no collision verdict.
        encoded = self.pair(offsets / 10.0)
        scores = (self.query(hidden).unsqueeze(-2) * encoded).sum(-1) / math.sqrt(64)
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=-1)
        context = (weights.unsqueeze(-1) * encoded * mask.unsqueeze(-1)).sum(-2)
        return self.output(self.dropout(hidden + self.project(context)))


class SetStepVerifier(StepVerifier):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = OffsetSetHead(config.d_model, config.dropout_rate)

    def forward(self, input_ids, attention_mask, positions, kinds,
                obstacle_offsets, obstacle_mask):
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        gathered = hidden.gather(
            1, positions.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, hidden.shape[-1]))
        logits = self.classifier(gathered, obstacle_offsets, obstacle_mask)
        classes = torch.arange(len(TAGS), device=logits.device)
        allowed = torch.where(kinds.unsqueeze(-1) == 1,
                              classes >= 4, classes < 4)
        return logits.masked_fill(~allowed, -1e4)
