import torch
import random
import quadprog
import numpy as np


class AGEM():

    def __init__(self, patterns_per_step, sample_size):

        self.patterns_per_step = int(patterns_per_step)
        self.sample_size = int(sample_size)

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None

    def project_gradients(self, model):
        if self.memory_x is not None:
            for (n1, p1), (n2, refg) in zip(model.named_parameters(),
                                    self.reference_gradients):

                assert n1 == n2, "Different model parameters in AGEM projection"
                assert (p1.grad is not None and refg is not None) \
                        or (p1.grad is None and refg is None)

                if refg is None:
                    continue

                dotg = torch.dot(p1.grad.view(-1), refg.view(-1))
                dotref = torch.dot(refg.view(-1), refg.view(-1))
                if dotg < 0:
                    p1.grad -= (dotg / dotref) * refg

    def sample_from_memory(self, sample_size):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """

        if self.memory_x is None or self.memory_y is None:
            raise ValueError('Empty memory for AGEM.')

        if self.memory_x.size(0) <= sample_size:
            return self.memory_x, self.memory_y
        else:
            idxs = random.sample(range(self.memory_x.size(0)), sample_size)
            return self.memory_x[idxs], self.memory_y[idxs]

    @torch.no_grad()
    def update_memory(self, dataloader):
        """
        Update replay memory with patterns from current step.
        """

        tot = 0
        for x, y in dataloader:
            if tot + x.size(0) <= self.patterns_per_step:
                if self.memory_x is None:
                    self.memory_x = x.clone()
                    self.memory_y = y.clone()
                else:
                    self.memory_x = torch.cat((self.memory_x, x), dim=0)
                    self.memory_y = torch.cat((self.memory_y, y), dim=0)
            else:
                diff = self.patterns_per_step - tot
                if self.memory_x is None:
                    self.memory_x = x[:diff].clone()
                    self.memory_y = y[:diff].clone()
                else:
                    self.memory_x = torch.cat((self.memory_x, x[:diff]), dim=0)
                    self.memory_y = torch.cat((self.memory_y, y[:diff]), dim=0)
                break
            tot += x.size(0)


class GEM():
    def __init__(self, patterns_per_step: int, memory_strength: float):
        """
        :param patterns_per_step: number of patterns per step in the memory
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_step = int(patterns_per_step)
        self.memory_strength = memory_strength

        self.memory_x, self.memory_y = {}, {}

        self.G = None

    @torch.no_grad()
    def project_gradients(self, model, task_id, device):
        """
        Project gradient based on reference gradients
        """

        if task_id > 0:
            g = torch.cat([p.grad.flatten()
                          for p in model.parameters()
                          if p.grad is not None], dim=0)

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in model.parameters():

                curr_pars = p.numel()

                if p.grad is None:
                    continue

                p.grad.copy_(v_star[num_pars:num_pars+curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"


    @torch.no_grad()
    def update_memory(self, dataloader, task_id):
        """
        Update replay memory with patterns from current step.
        """

        t = task_id

        tot = 0
        for x, y in dataloader:
            if tot + x.size(0) <= self.patterns_per_step:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_y[t] = y.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
            else:
                diff = self.patterns_per_step - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]),
                                                 dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]),
                                                 dim=0)
                break
            tot += x.size(0)

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        n_tasks = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        # the following line is taken from the original code
        # however, 0.5 * (P + P^T) == 0.5 * 2P == P so it does not make much sense.
        # I could simply add np.eye(n_tasks) * 1e-3 to the line above
        P = 0.5 * (P + P.transpose()) + np.eye(n_tasks) * 0.001
        q = np.dot(memories_np, gradient_np) * -1
        h = np.zeros(n_tasks) + self.memory_strength
        v = quadprog.solve_qp(P, q, np.eye(n_tasks), h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()
