import torch


def accuracy(output, target, dim=1):
    probs = torch.nn.functional.softmax(output, dim=dim)
    winners = probs.argmax(dim=dim)

    acc = (winners == target).sum().float() / target.size(0)

    return acc.item()