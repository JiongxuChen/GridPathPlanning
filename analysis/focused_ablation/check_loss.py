"""Quick gradient contract for the isolated decision-only variant."""
import torch
import torch.nn.functional as F
from gridpath.planning.model import path_loss


def check():
    logits = torch.zeros((2, 4, 6), requires_grad=True)
    labels = torch.tensor([[0, 1, -100, -100], [0, 0, 0, 4]])
    value = path_loss(logits, labels)
    value.backward()
    assert torch.all(logits.grad[0, 0] == 0)
    assert torch.all(logits.grad[1, :3] == 0)
    assert torch.all(logits.grad[0, 2:] == 0)

    for i, position, weight in ((0, 1, 1.6), (1, 3, .4)):
        expected = torch.zeros(6, dtype=logits.dtype, requires_grad=True)
        (.5 * weight * F.cross_entropy(expected[None], labels[i, position][None]) / 2).backward()
        assert torch.allclose(logits.grad[i, position], expected.grad)

    # With no legal prefix, the ablation retains the original full decision loss.
    lone = torch.zeros((1, 1, 6), requires_grad=True)
    assert torch.allclose(path_loss(lone, torch.tensor([[5]])),
                          1.6 * F.cross_entropy(lone[0], torch.tensor([5])))


if __name__ == '__main__':
    check()
    print('decision-only loss contract passed')
