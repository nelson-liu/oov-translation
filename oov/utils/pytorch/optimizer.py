import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import logging

from .yellowfin import YFOptimizer

logger = logging.getLogger(__name__)


class Optimizer(object):
    def set_parameters(self, params):
        self.params = list(params)
        if self.method == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == "adagrad":
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == "adadelta":
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == "adam":
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        elif self.method == "yellowfin":
            self.optimizer = YFOptimizer(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_gradient_norm,
                 lr_decay=1, start_decay_at=None):
        self.last_ppl = None
        self.lr = lr
        self.max_gradient_norm = max_gradient_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

    def step(self):
        "Compute the norm of the graadients."
        if self.max_gradient_norm:
            clip_grad_norm(self.params, self.max_gradient_norm)
        self.optimizer.step()

    def update_learning_rate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """

        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            logger.info("Decaying learning rate "
                        "from {} to {}".format(self.lr,
                                               self.lr * self.lr_decay))
            self.lr = self.lr * self.lr_decay

        self.last_ppl = ppl
        self.optimizer.param_groups[0]["lr"] = self.lr
