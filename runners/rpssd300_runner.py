import time

from torch import save as torch_save, load as torch_load
from torch.cuda import empty_cache as torch_cuda_empty_cache
from torch.optim import SGD
from torch.utils.data import DataLoader

from models import RPSSD300
from utils.datasets import ssd_collate

from math import isnan


class RPSSD300Runner:
    @staticmethod
    def pre_init(load_pretrained_base=True, **config):
        model: RPSSD300 = RPSSD300(load_pretrained_base)

        biases, not_biases = RPSSD300Runner._get_model_parameters(model)

        if config:
            lr = config['lr']
            opt = SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], **config)
        else:
            opt = SGD(params=[{'params': biases, 'lr': 2 * .001}, {'params': not_biases}], lr=.001, momentum=.9,
                      weight_decay=.0005)

        return model, opt

    @staticmethod
    def _get_model_parameters(model):
        """Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. All credits to @sgrvinod.
        """

        biases = list()
        not_biases = list()

        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        return biases, not_biases

    def _adjust_learning_rate(self, optimizer, scale):
        """Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection. All credits to @sgrvinod.
        """

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale

        self.out("\tNOTE: Decaying learning rate to %f\n" % (optimizer.param_groups[1]['lr'],))

    def __init__(self, model: RPSSD300, optimizer, dataset, batch_size=32, iterations: tuple = None,
                 decay_lr_iterations: list = None, gamma: float = .1, use_cuda=False, logfile=None):
        self.model = model
        self.use_cuda = use_cuda
        self.model.set_cuda(use_cuda)
        if use_cuda:
            self.model = self.model.cuda()
            self.model.loss_function = self.model.loss_function.cuda()
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size

        self.iterations = iterations
        self.decay_lr_iterations = decay_lr_iterations
        self.gamma = gamma

        self.iteration_losses = []

        self.start_time = time.time()
        self.end_time = None

        self.logfile = logfile

    def out(self, value):
        if self.logfile:
            print(value, file=open(self.logfile, 'a'))
        else:
            print(value)

    def has_exceeded_max_iterations(self):
        return self.count_processed_batches() >= self.iterations[1]

    def count_processed_batches(self):
        return len(self.iteration_losses)

    def train_(self, model, optimizer, loader, savefile):
        model.train()

        for data, target_locs, target_cls in loader:
            if self.has_exceeded_max_iterations():
                return model, optimizer

            if self.use_cuda:
                data, target_locs, target_cls = data.cuda(), target_locs.cuda(), target_cls.cuda()

            out = model(data)
            loss = model.loss(target_locs, target_cls)

            self.iteration_losses.append(loss.item())

            if isnan(loss.item()):
                self.out(f'nan loss on iteration {self.count_processed_batches()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del data, target_locs, target_cls, out, loss
            torch_cuda_empty_cache()

            if self.count_processed_batches() * self.batch_size % len(self.dataset) == 0:
                epoch = self.count_processed_batches() * self.batch_size // len(self.dataset)
                see_an_epoch_behind = -len(self.dataset) // self.batch_size
                avg_epoch_loss = sum(self.iteration_losses[see_an_epoch_behind:]) / abs(see_an_epoch_behind)
                self.out(f'Epoch #{epoch}, iteration {self.count_processed_batches()}: {avg_epoch_loss} average loss')

            torch_save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration_losses': self.iteration_losses,
                'use_cuda': self.use_cuda,
                'start_time': self.start_time,
                'end_time': self.end_time
            }, savefile)

        return model, optimizer

    def train(self, savefile):
        epoch = self.count_processed_batches() * self.batch_size // len(self.dataset)
        self.out(f'Starting training at epoch {epoch + 1} with {self.count_processed_batches()} processed iterations\n')

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=ssd_collate,
                            num_workers=4, pin_memory=True, drop_last=True)

        model = self.model
        optimizer = self.optimizer

        while not self.has_exceeded_max_iterations():
            if self.count_processed_batches() in self.decay_lr_iterations:
                self._adjust_learning_rate(self.optimizer, self.gamma)

            model, optimizer = self.train_(model, optimizer, loader, savefile)

        self.end_time = time.time()
        self.out(f'\nFinished training; training elapsed for {self.end_time - self.start_time} seconds')

    def load(self, checkpoint, not_to_load=None):
        loader = torch_load(checkpoint)

        if not_to_load:
            for k in not_to_load:
                del loader[k]

        if 'model' in loader.keys():
            self.model.load_state_dict(loader['model'])

        if 'optimizer' in loader.keys():
            self.optimizer.load_state_dict(loader['optimizer'])

        if 'iteration_losses' in loader.keys():
            self.iteration_losses = loader['iteration_losses']
            self.iterations = len(self.iteration_losses), self.iterations[1]

        if 'use_cuda' in loader.keys():
            self.use_cuda = loader['use_cuda']

            self.model.set_cuda(self.use_cuda)
            if self.use_cuda:
                self.model = self.model.cuda()
                self.model.loss_function = self.model.loss_function.cuda()

        if 'start_time' in loader.keys():
            self.start_time = loader['start_time']

        if 'end_time' in loader.keys():
            self.end_time = loader['end_time']
