import torch

VOCAB_SIZE = 64
SEQ_LEN = 16


def get_batch(step, micro_step, batch_size=8, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(1000 * step + micro_step)
    x = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN), generator=g, device=device)
    shift = (x.sum(dim=1, keepdim=True) % 7) + 1
    y = (x + shift) % VOCAB_SIZE
    return x, y