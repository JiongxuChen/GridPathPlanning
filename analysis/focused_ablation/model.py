"""T5 step verifier ablation: supervise the decisive position only.

This file replaces planning/model.py in an isolated reproduction copy. The
architecture, inference, class weights and decisive loss coefficient match the
main experiment; only the legal-prefix loss term is removed.
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5EncoderModel, T5PreTrainedModel
from .protocol import TAGS


class StepVerifier(T5PreTrainedModel):
    _tied_weights_keys = [r'encoder.encoder.embed_tokens.weight']

    def __init__(self, config):
        config.num_labels = len(TAGS)
        config.id2label = dict(enumerate(TAGS))
        config.label2id = {tag: i for i, tag in enumerate(TAGS)}
        super().__init__(config)
        self.encoder = T5EncoderModel(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(config.d_model, len(TAGS))
        self.post_init()
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_backbone(cls, path):
        encoder = T5EncoderModel.from_pretrained(path, local_files_only=True)
        model = cls(encoder.config)
        model.encoder = encoder
        return model

    def forward(self, input_ids, attention_mask, positions, kinds):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        gathered = hidden.gather(1, positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        logits = self.classifier(self.dropout(gathered))
        classes = torch.arange(len(TAGS), device=logits.device)
        allowed = torch.where(kinds.unsqueeze(-1) == 1, classes >= 4, classes < 4)
        return logits.masked_fill(~allowed, -1e4)


def collate(features, pad_id, device):
    width = max(len(f['input_ids']) for f in features)
    queries = max(len(f['positions']) for f in features)
    inputs = torch.full((len(features), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(inputs)
    positions = torch.full((len(features), queries), -1, dtype=torch.long, device=device)
    kinds = torch.zeros_like(positions)
    labels = torch.full_like(positions, -100)
    for i, f in enumerate(features):
        n, q = len(f['input_ids']), len(f['positions'])
        inputs[i, :n] = torch.tensor(f['input_ids'], device=device)
        mask[i, :n] = 1
        positions[i, :q] = torch.tensor(f['positions'], device=device)
        kinds[i, :q] = torch.tensor(f['kinds'], device=device)
        if 'labels' in f:
            labels[i, :q] = torch.tensor(f['labels'], device=device)
    return {'input_ids': inputs, 'attention_mask': mask, 'positions': positions, 'kinds': kinds}, labels


def path_loss(logits, labels):
    """Only the decisive position contributes; retain its original coefficient."""
    losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100,
                             reduction='none').reshape(labels.shape)
    values = []
    for i, row in enumerate(labels):
        supervised = torch.nonzero(row != -100).flatten()
        if not len(supervised):
            raise ValueError('Training path has no supervision')
        last = supervised[-1]
        weight = torch.where(row[last] == 4, logits.new_tensor(0.4), logits.new_tensor(1.6))
        decision = weight * losses[i, last]
        values.append(.5 * decision if len(supervised) > 1 else decision)
    return torch.stack(values).mean()
