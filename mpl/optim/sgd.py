import torch
import torch.optim as optim


class SGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                # exclude 1) dense param with None grad and 2) dense placeholders for sparse params, and
                # 3) sparse param with None grad
                if hasattr(p, "is_placeholder") or (
                        p.grad is None and (not hasattr(p, "is_sparse_param") or p.dense.grad is None)):
                    # dense placeholder
                    continue
                # if p.grad is None:
                #     if not hasattr(p, "is_sparse_param"):
                #         # dense param with None grad
                #         continue
                #     elif p.dense.grad is None:
                #         # sparse param with None grad
                #         continue

                if hasattr(p, "is_sparse_param"):
                    d_p = p.dense.grad.masked_select(p.mask)
                    p = p._values()
                else:
                    d_p = p.grad

                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        return loss
