import torch
import torch.autograd.profiler as profiler

class Trainer():

    def __init__(self, model, optimizer, criterion, device,
            eval_metric=None, clip_grad=0, penalties=None):
        """
        :param clip_grad: > 0 to clip gradient after backward. 0 not to clip.
        :param penalties: dictionary of penalties name->hyperparams. None to disable them.
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.eval_metric = eval_metric
        self.clip_grad = clip_grad
        self.penalties = penalties
        self.device = device

    def train(self, x, y, task_id=None, lengths=None):
        self.model.train()

        self.optimizer.zero_grad()

        if lengths is None:
            out = self.model(x)
        else:
            out = self.model(x, lengths)

        if task_id is not None:
            to_zero = list(set(range(self.model.layers['out'].weight.size(0))) - set([task_id*2, task_id*2+1]))
            out[:, to_zero] = 0.

        loss = self.criterion(out, y)
        loss += self.add_penalties()

        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)

        self.optimizer.step()
        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric


    def test(self, x, y, task_id=None, lengths=None):
        with torch.no_grad():
            self.model.eval()

            if lengths is None:
                out = self.model(x)
            else:
                out = self.model(x, lengths)

            if task_id is not None:
                to_zero = list(set(range(self.model.layers['out'].weight.size(0))) - set([task_id*2, task_id*2+1]))
                out[:, to_zero] = 0.

            loss = self.criterion(out, y)
            metric = self.eval_metric(out, y) if self.eval_metric else None

            return loss.item(), metric

    def train_ewc(self, x, y, ewc, task_id, multi_head=False, lengths=None):

        self.model.train()

        self.optimizer.zero_grad()

        if lengths is None:
            out = self.model(x)
        else:
            out = self.model(x, lengths)

        if multi_head:
            to_zero = list(set(range(self.model.layers['out'].weight.size(0))) - set([task_id*2, task_id*2+1]))
            out[:, to_zero] = 0.

        loss = self.criterion(out, y)
        loss += self.add_penalties()
        loss += ewc.penalty(task_id)
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric

    def train_gem(self, x, y, gem, task_id, lengths=None):
        self.model.train()
        if task_id > 0:
            G = []
            self.model.train()
            for t in range(task_id):
                self.optimizer.zero_grad()
                xref = gem.memory_x[t].to(self.device)
                yref = gem.memory_y[t].to(self.device)
                if lengths is None:
                    out = self.model(xref)
                else:
                    out = self.model(xref, lengths)
                loss = self.criterion(out, yref)
                loss.backward()

                G.append(torch.cat([p.grad.flatten()
                         for p in self.model.parameters()
                         if p.grad is not None], dim=0))

            gem.G = torch.stack(G)  # (steps, parameters)

        self.optimizer.zero_grad()

        if lengths is None:
            out = self.model(x)
        else:
            out = self.model(x, lengths)
        loss = self.criterion(out, y)
        loss += self.add_penalties()
        loss.backward()
        gem.project_gradients(self.model, task_id, self.device)

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        metric = self.eval_metric(out, y) if self.eval_metric else None
        return loss.item(), metric

    def train_agem(self, x, y, agem, task_id, multi_head=False, lengths=None):
        self.model.train()

        # compute reference gradients
        if agem.memory_x is not None:
            self.optimizer.zero_grad()
            xref, yref = agem.sample_from_memory(agem.sample_size)
            xref, yref = xref.to(self.device), yref.to(self.device)
            if lengths is None:
                out = self.model(xref)
            else:
                out = self.model(xref, lengths)
            loss = self.criterion(out, yref)
            loss.backward()
            agem.reference_gradients = [
                (n, p.grad)
                for n, p in self.model.named_parameters()]

        self.optimizer.zero_grad()

        if lengths is None:
            out = self.model(x)
        else:
            out = self.model(x, lengths)

        if multi_head:
            to_zero = list(set(range(self.model.layers['out'].weight.size(0))) - set([task_id*2, task_id*2+1]))
            out[:, to_zero] = 0.

        loss = self.criterion(out, y)
        loss += self.add_penalties()
        agem.project_gradients(self.model)
        loss.backward()

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        metric = self.eval_metric(out, y) if self.eval_metric else None
        return loss.item(), metric


    def train_lwf(self, x,y, lwf, task_id, lengths=None):
        self.model.train()

        self.optimizer.zero_grad()

        if lengths is None:
            out = self.model(x)
        else:
            out = self.model(x, lengths)

        loss = self.criterion(out, y)
        loss += self.add_penalties()
        loss += lwf.penalty(out, x, task_id)
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric        


    def train_mas(self, x, y, mas, task_id, truncated_time=0, lengths=None):

        self.model.train()

        self.optimizer.zero_grad()

        if lengths is None:
            out = self.model(x)
        else:
            out = self.model(x, lengths)

        loss = self.criterion(out, y)
        loss += self.add_penalties()
        loss += mas.penalty(task_id)
        loss.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()

        mas.compute_importance(self.optimizer, x, truncated_time)

        metric = self.eval_metric(out, y) if self.eval_metric else None

        return loss.item(), metric


    def add_penalties(self):
        penalty = torch.zeros(1, device=self.device).squeeze()
        if self.penalties:
            
            if 'l1' in self.penalties.keys():
                penalty = l1_penalty(self.model, self.penalties['l1'], self.device)
            
        return penalty


def l1_penalty(model, lamb, device):

    penalty = torch.tensor(0.).to(device)

    for p in model.parameters():
        if p.requires_grad:
            penalty += torch.sum(torch.abs(p))

    return lamb * penalty
