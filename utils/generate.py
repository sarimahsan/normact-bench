import torch
import torch.nn.functional as F

def top_k_logits(logits, k):
    k = min(k, logits.size(-1))
    v, ix = torch.topk(logits, k)
    out = torch.full_like(logits, -float("inf"))
    out.scatter_(dim=-1, index=ix, src=v)
    return out

def top_p_logits(logits, p=0.9):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)

    mask = cum_probs > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = 0

    sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
    out = torch.full_like(logits, -float("inf"))
    out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return out

@torch.no_grad()
def generate(
    model,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=50,
    top_p=None,
    device="cpu"
):
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    model.eval()
    x = prompt.to(device)

    for _ in range(max_new_tokens):
        logits = model(x)          # (B, T, V)
        logits = logits[:, -1, :]  # last token only
        logits = logits / temperature

        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        if top_p is not None:
            logits = top_p_logits(logits, top_p)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    return x
