import torch

def acc(y , t ):
    y = torch.softmax(y , dim = 1).argmax(dim = 1).type(torch.long)
    return (y == t).sum().item() / len(y) * 100
