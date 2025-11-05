import torch
from typing import Any, Dict, Tuple, Iterable, Optional, Callable
from torch import Tensor
import math

class MyAdamW(torch.optim.Optimizer):
    def __init__(self, 
                 params: Iterable[Tensor] | Iterable[Dict[str, Any]] | Iterable[Tuple[str, Tensor]], 
                 lr: float,
                 betas:tuple[float,float],
                 eps:float,
                 weight_decay:float
                 ) -> None:
        beta1, beta2 = betas
        if lr < 0 or weight_decay < 0 or eps < 0:
            raise ValueError(f"these params should be positive, \
                             now lr: {lr}, weight decay:{weight_decay},\
                             eps: {eps}")
        if (beta1 >= 1 or beta1 <= 0) or (beta2 >= 1 or beta2 <= 0):
            raise ValueError(f"beta1 and beta2 should be in (0,1), now: {beta1}, {beta2}")
        defaults = {
            "lr":lr,
            "beta1":beta1,
            "beta2":beta2,
            "eps":eps,
            "weight_decay":weight_decay
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None): 
        loss = None
        if closure is not None:
            with torch.enable_grad():
                closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                p:Tensor
                if p.grad is None:
                    continue
                g = p.grad  # 获得梯度
                state = self.state[p] # 获得其他信息
                # self.state是一个字典
                # 其中每个kv对是 参数-参数信息，参数信息也是一个字典
                # 参数信息可以随自己心意存储，例如"t"存储时间步

                #最初：m和v为0，t没有
                # 当伪代码中t=1的循环时，做的事：
                # 0. 计算梯度
                # 1. 更新m（用0作为初值）；更新v（用0作为初值）
                # 2. 用当前的t=1来计算当前的学习率
                # 3. 用当前学习率、当前m和当前v更新参数
                # 4. 用权重衰减更新参数
                # 5. t自增1

                if "t" not in state:
                    state["t"] = 1
                    state["first_moment"] = torch.zeros_like(p)
                    state["second_moment"] = torch.zeros_like(p)

                t = state["t"]
                m = state["first_moment"]
                v = state["second_moment"]
               
                m.mul_(beta1).add_(g, alpha = 1 - beta1)
                v.mul_(beta2).add_(g * g, alpha = 1 - beta2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

                
                p.add_(m / (torch.sqrt(v) + eps), alpha= -lr_t)
                p.mul_(1 - lr * weight_decay)
                
                state["t"] = t + 1
        return loss
    


def cosine_annealing_with_warm_up(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
)->float:
    t = it
    min = min_learning_rate
    max = max_learning_rate
    Tw = warmup_iters
    Tc = cosine_cycle_iters
    if t < Tw:
        lr = t * max / Tw
    elif t >= Tw and t <= Tc:
        theta = math.pi * (t-Tw) / (Tc-Tw)
        lr = min + 0.5 * (1 + math.cos(theta)) * (max-min)
    elif t > Tc:
        lr = min
    return lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float)->None:
    sum = 0
    for param in parameters:
        if param.grad != None:
            sum += torch.norm(param.grad)**2
    total_norm = math.sqrt(sum)
    if total_norm >= max_l2_norm:
        k = max_l2_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad != None:
                param.grad.mul_(k)