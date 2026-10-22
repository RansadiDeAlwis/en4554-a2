"""Task 2(a): SGD with momentum and Adam written as torch.optim.Optimizer
subclasses. Update rules follow the PyTorch documentation so the results can
be compared with torch.optim.SGD / torch.optim.Adam exactly."""
import math
import torch


# >>> REPORT: optimizers
class MySGD(torch.optim.Optimizer):
    """v_t = mu * v_{t-1} + g_t   (v_1 = g_1);   theta <- theta - lr * v_t
    (with nesterov: theta <- theta - lr * (g_t + mu * v_t)).
    weight_decay adds wd * theta to the gradient (L2 regularisation)."""
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        if lr <= 0:
            raise ValueError("lr must be positive")
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            wd, nesterov = group["weight_decay"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                if mu != 0:
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = g.clone()
                        state["momentum_buffer"] = buf
                    else:
                        buf.mul_(mu).add_(g)
                    g = g.add(buf, alpha=mu) if nesterov else buf
                p.add_(g, alpha=-lr)
        return loss


class MyAdam(torch.optim.Optimizer):
    """m_t = b1 m + (1-b1) g,   v_t = b2 v + (1-b2) g^2
    m_hat = m_t / (1-b1^t),     v_hat = v_t / (1-b2^t)      (bias correction)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    Implemented in the algebraically identical form PyTorch uses:
    theta <- theta - (lr/(1-b1^t)) * m_t / (sqrt(v_t)/sqrt(1-b2^t) + eps)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2) = group["lr"], group["betas"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1 = 1 - b1 ** t
                bc2 = 1 - b2 ** t
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss
# <<< REPORT
