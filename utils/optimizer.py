import torch

class AdamW:
    """
    Decoupled Weight Decay Adam (AdamW) implemented from scratch.
    """
    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # optimizer state
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        self.last_stats = None

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = self.betas

        total_update_abs = 0.0
        total_update_sq = 0.0
        total_denom_mean = 0.0
        total_denom_var = 0.0
        count = 0

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Decoupled weight decay
            if self.weight_decay != 0:
                p.data.mul_(1 - self.lr * self.weight_decay)

            # First moment update
            self.m[i].mul_(b1).add_(g, alpha=1 - b1)

            # Second moment update
            self.v[i].mul_(b2).addcmul_(g, g, value=1 - b2)

            # Bias correction
            m_hat = self.m[i] / (1 - b1 ** self.t)
            v_hat = self.v[i] / (1 - b2 ** self.t)

            # Parameter update
            denom = v_hat.sqrt().add(self.eps)
            update = m_hat.div(denom).mul(self.lr)
            p.add_(update, alpha=-1.0)

            # Accumulate statistics
            total_update_abs += update.abs().mean().item()
            total_update_sq += update.pow(2).mean().item()
            total_denom_mean += denom.mean().item()
            total_denom_var += denom.var().item()
            count += 1

        if count > 0:
            self.last_stats = {
                "mean_update_magnitude": total_update_abs / count,
                "update_variance": (total_update_sq / count) - ((total_update_abs / count) ** 2),
                "mean_denom": total_denom_mean / count,
                "var_denom": total_denom_var / count
            }
        else:
            self.last_stats = None
