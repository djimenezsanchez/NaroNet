import torch


def euclidean_norm_squared(vec_list):
    return sum(torch.sum(v ** 2).item() for v in vec_list)


def modify_lr(optimizer, mul):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= mul


def get_lr(optimizer):
    lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    for lr in lrs:
        if lr != lrs[0]:
            return lrs
    return lrs[0]


DENOMINATOR_EPS = 1e-25


class LossgradOptimizer:

    def __init__(self, optimizer, net, criterion, c=1.05):
        self.optimizer = optimizer
        self.c = c
        self.net = net
        self.criterion = criterion

    def load_state_dict(self, state_dict):
        if 'c' in state_dict:
            self.c = state_dict['c']
            del state_dict['c']
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['c'] = self.c
        return state_dict

    def step(self, X, y, loss):
        with torch.no_grad():
            grad_norm_squared = euclidean_norm_squared((-p.grad for p
                                                        in self.net.parameters()))
            lred = grad_norm_squared * get_lr(self.optimizer)
            approx = loss.item() - lred
            self.optimizer.step()
            actual = self.criterion(self.net(X), y).item()
            rel_err = (actual - approx) / (lred + DENOMINATOR_EPS)
        if rel_err > 0.5:
            h_mul = 1 / self.c
        else:
            h_mul = self.c
        modify_lr(self.optimizer, h_mul)
        return rel_err, grad_norm_squared

    def get_lr(self):
        return get_lr(self.optimizer)

class LossgradOptimizer:

    def __init__(self, optimizer, net, criterion, c=1.05):
        self.optimizer = optimizer
        self.c = c
        self.net = net
        self.criterion = criterion

    def load_state_dict(self, state_dict):
        if 'c' in state_dict:
            self.c = state_dict['c']
            del state_dict['c']
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['c'] = self.c
        return state_dict

    def step(self, X, y, loss):
        with torch.no_grad():
            grad_norm_squared = euclidean_norm_squared((-p.grad for p
                                                        in self.net.parameters()))
            lred = grad_norm_squared * get_lr(self.optimizer)
            approx = loss.item() - lred
            self.optimizer.step()
            actual = self.criterion(self.net(X), y).item()
            rel_err = (actual - approx) / (lred + DENOMINATOR_EPS)
        if rel_err > 0.5:
            h_mul = 1 / self.c
        else:
            h_mul = self.c
        modify_lr(self.optimizer, h_mul)
        return rel_err, grad_norm_squared

    def get_lr(self):
        return get_lr(self.optimizer)