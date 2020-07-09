import time
from collections import OrderedDict
from typing import List, Tuple

from torch import save as torch_save, load as torch_load, max as torch_max, no_grad as torch_no_grad
from torch.cuda import empty_cache as torch_cuda_empty_cache
from torch.optim import SGD
from torch.utils.data import DataLoader

from sklearn.metrics import recall_score, precision_score, accuracy_score

from models import RPYOLOv3

from math import isnan


class RPYOLOv3Runner:
    @staticmethod
    def pre_init(model_kwargs, learning_rate, batch_size, decay, momentum):
        """Adapted from https://github.com/marvis/pytorch-yolo2. All credits to @marvis.
        """

        model: RPYOLOv3 = RPYOLOv3(**model_kwargs)

        # Not sure what these (lines 21-28) are for, but they are included in https://github.com/marvis/pytorch-yolo2.
        params_dict = dict(model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key.find('.bn') >= 0 or key.find('.bias') >= 0:
                params += [{'params': [value], 'weight_decay': 0.0}]
            else:
                params += [{'params': [value], 'weight_decay': decay * batch_size}]

        optimizer = SGD(model.parameters(), lr=learning_rate / batch_size, momentum=momentum, dampening=0,
                        weight_decay=decay * batch_size)

        return model, optimizer

    @staticmethod
    def _handle_backbone_weight_type(backbone_weight):
        if type(backbone_weight) == str:
            return torch_load(backbone_weight)
        elif type(backbone_weight) == OrderedDict:
            return backbone_weight
        else:
            raise ValueError('backbone_weight type is unknown')

    def __init__(self, model: RPYOLOv3, optimizer, dataset, validation_dataset, batch_size=64, iterations: tuple = None,
                 initial_lr=.001, decay_steps_scales: List[Tuple] = None, use_cuda=False, logfile=None):
        self.model = model

        self.use_cuda = use_cuda
        if use_cuda:
            self.model = self.model.cuda()
            self.model.loss_function = self.model.loss_function.cuda()
        self.optimizer = optimizer

        self.initial_lr = initial_lr
        if decay_steps_scales:
            self.steps = [float(x[0]) for x in decay_steps_scales]
            self.scales = [float(x[1]) for x in decay_steps_scales]

        self.dataset = dataset
        self.batch_size = batch_size

        self.validation_dataset = validation_dataset

        self.seen_images = 0
        self.iteration_losses = []

        self.burn_in = 1000
        self.power = 4

        self.iterations = iterations

        self.start_time = time.time()
        self.end_time = None

        self.logfile = logfile

    def out(self, value):
        if self.logfile:
            print(value, file=open(self.logfile, 'a'))
        else:
            print(value)

    def detecting(self, reset_time=False, backbone_weight=None, large_anchor_boxes=None, medium_anchor_boxes=None,
                  small_anchor_boxes=None, dataset=None, batch_size=None, iterations: tuple = None, initial_lr=None,
                  decay_steps_scales: List[Tuple] = None, seen_images=None, iteration_losses=None):
        if reset_time:
            self.start_time = time.time()
            self.end_time = None

        if backbone_weight:
            if self.model.is_recognizing():
                self.model.load_state_dict(self._handle_backbone_weight_type(backbone_weight))
            else:
                raise ValueError('model must be in recognizing state')

        if large_anchor_boxes:
            self.model.large_predictor_loss.set_anchors(large_anchor_boxes, 32)
        if medium_anchor_boxes:
            self.model.medium_predictor_loss.set_anchors(medium_anchor_boxes, 16)
        if small_anchor_boxes:
            self.model.small_predictor_loss.set_anchors(small_anchor_boxes, 8)

        self.model.detecting()

        if dataset:
            self.dataset = dataset

        if initial_lr:
            self.initial_lr = initial_lr
        if decay_steps_scales:
            self.steps = [float(x[0]) for x in decay_steps_scales]
            self.scales = [float(x[1]) for x in decay_steps_scales]

        if batch_size:
            self.batch_size = batch_size

        if seen_images:
            self.seen_images = seen_images

        if iteration_losses:
            self.iteration_losses = iteration_losses
        else:
            self.iteration_losses = []

        if iterations:
            self.iterations = iterations

    def recognizing(self, reset_time=False, backbone_weight=None, dataset=None, validation_dataset=None,
                    batch_size=None, iterations: tuple = None, initial_lr=None, decay_steps_scales: List[Tuple] = None,
                    seen_images=None, iteration_losses=None):
        if reset_time:
            self.start_time = time.time()
            self.end_time = None

        self.model.recognizing()

        if backbone_weight:
            if self.model.is_recognizing():
                self.model.load_state_dict(self._handle_backbone_weight_type(backbone_weight))
            else:
                raise ValueError('model must be in recognizing state')

        if dataset:
            self.dataset = dataset

        if validation_dataset:
            self.validation_dataset = validation_dataset

        if initial_lr:
            self.initial_lr = initial_lr

        if decay_steps_scales:
            self.steps = [float(x[0]) for x in decay_steps_scales]
            self.scales = [float(x[1]) for x in decay_steps_scales]

        if batch_size:
            self.batch_size = batch_size

        if seen_images:
            self.seen_images = seen_images

        if iteration_losses:
            self.iteration_losses = iteration_losses
        else:
            self.iteration_losses = []

        if iterations:
            self.iterations = iterations

    def _adjust_poly_learning_rate(self, optimizer, batch):
        if batch < self.burn_in:
            lr = self.initial_lr * pow(batch / self.burn_in, self.power)
        else:
            lr = self.initial_lr * pow(1 - batch / self.iterations[1], self.power)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def _adjust_step_learning_rate(self, optimizer, batch):
        """Adapted from https://github.com/marvis/pytorch-yolo2. All credits to @marvis.
        """

        lr = self.initial_lr

        for i in range(len(self.steps)):
            scale = self.scales[i] if i < len(self.scales) else 1
            if batch >= self.steps[i]:
                lr = lr * scale
                if batch == self.steps[i]:
                    break
            else:
                break

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr / self.batch_size

        return lr

    def adjust_learning_rate(self, optimizer, batch):
        if self.model.is_recognizing():
            self._adjust_poly_learning_rate(optimizer, batch)
        else:
            self._adjust_step_learning_rate(optimizer, batch)

    def has_exceeded_max_iterations(self):
        return self.count_processed_batches() >= self.iterations[1]

    def count_processed_batches(self):
        return len(self.iteration_losses)

    def change_dataset(self, dataset):
        self.dataset = dataset

    def train_(self, model, optimizer, loader, savefile):
        model.train()

        self.adjust_learning_rate(self.optimizer, self.count_processed_batches())

        for data, target in loader:
            if self.has_exceeded_max_iterations():
                return model, optimizer

            self.adjust_learning_rate(self.optimizer, self.count_processed_batches())

            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            out = model(data)
            loss = model.loss(target)

            if isnan(loss.item()):
                self.out(f'nan loss on iteration {self.count_processed_batches()}')

            self.iteration_losses.append(loss.item())
            self.seen_images += data.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del data, target, out, loss
            torch_cuda_empty_cache()

            if self.count_processed_batches() * self.batch_size % len(self.dataset) == 0:
                epoch = self.count_processed_batches() * self.batch_size // len(self.dataset)
                see_an_epoch_behind = -len(self.dataset) // self.batch_size
                avg_epoch_loss = sum(self.iteration_losses[see_an_epoch_behind:]) / abs(see_an_epoch_behind)
                self.out(f'Epoch #{epoch}, iteration {self.count_processed_batches()}: {avg_epoch_loss} average loss')
                self.validate(model)
                model.train()

            torch_save({
                'model': model.state_dict(),
                'large_predictor_loss': model.large_predictor_loss.state_dict(),
                'medium_predictor_loss': model.medium_predictor_loss.state_dict(),
                'small_predictor_loss': model.small_predictor_loss.state_dict(),
                'optimizer': optimizer.state_dict(),
                'use_cuda': self.use_cuda,
                'seen_images': self.seen_images,
                'iteration_losses': self.iteration_losses,
                'start_time': self.start_time,
                'end_time': self.end_time
            }, savefile)

        return model, optimizer

    def train(self, savefile):
        epoch = self.count_processed_batches() * self.batch_size // len(self.dataset)
        self.out(f'Starting training at epoch {epoch + 1} with {self.count_processed_batches()} processed iterations\n')

        model = self.model
        optimizer = self.optimizer

        while not self.has_exceeded_max_iterations():
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                pin_memory=True, drop_last=True)
            model, optimizer = self.train_(model, optimizer, loader, savefile)

        self.end_time = time.time()
        self.out(f'\nFinished training; training elapsed for {self.end_time - self.start_time} seconds')

    def validate(self, model):
        if model.is_recognizing():
            return self._validate_recognition(model)
        else:
            return self._validate_detection(model)

    def _validate_recognition(self, model):
        validation_loader = DataLoader(self.validation_dataset, batch_size=1, shuffle=True)
        results = self.eval_recognition(model, validation_loader)

        iteration = self.count_processed_batches()
        epoch = iteration * self.batch_size // len(self.dataset)
        self.out(f'\tRecognition validation on epoch #{epoch}, iteration { iteration }: '
                 f'{results["accuracy"] * 100:.2f} accuracy, {results["recall"] * 100:.2f} recall, '
                 f'{results["precision"] * 100:.2f} precision, {results["average_loss"]} average validation loss')

        return results

    def eval_recognition(self, model, loader):
        model.eval()

        loss = 0.0
        predictions = []
        targets = []

        with torch_no_grad():
            for data, target in loader:
                out = model(data)

                out_loss = model.loss(target)
                loss += out_loss.item()

                predictions.append(torch_max(out, 1)[1].item())
                targets.append(target.item())

        return {
            'accuracy': accuracy_score(targets, predictions),
            'recall': recall_score(targets, predictions, average='micro'),
            'precision': precision_score(targets, predictions, average='micro'),
            'average_loss': loss / len(loader)
        }

    def _validate_detection(self, model):
        return {}

    def load(self, checkpoint, not_to_load=None):
        loader = torch_load(checkpoint)

        if not_to_load:
            for k in not_to_load:
                del loader[k]

        if 'model' in loader.keys():
            self.model.load_state_dict(loader['model'])

        if 'large_predictor_loss' in loader.keys():
            self.model.large_predictor_loss.load_state_dict(loader['large_predictor_loss'])

        if 'medium_predictor_loss' in loader.keys():
            self.model.medium_predictor_loss.load_state_dict(loader['medium_predictor_loss'])

        if 'small_predictor_loss' in loader.keys():
            self.model.small_predictor_loss.load_state_dict(loader['small_predictor_loss'])

        if 'optimizer' in loader.keys():
            self.optimizer.load_state_dict(loader['optimizer'])

        if 'iteration_losses' in loader.keys():
            self.iteration_losses = loader['iteration_losses']
            self.iterations = len(self.iteration_losses), self.iterations[1]

        if 'use_cuda' in loader.keys():
            self.use_cuda = loader['use_cuda']
            if self.use_cuda:
                self.model = self.model.cuda()
                self.model.loss_function = self.model.loss_function.cuda()

        if 'seen_images' in loader.keys():
            self.seen_images = loader['seen_images']
            if not self.model.is_recognizing():
                self.model.large_predictor_loss.set_seen_images(self.seen_images)
                self.model.medium_predictor_loss.set_seen_images(self.seen_images)
                self.model.small_predictor_loss.set_seen_images(self.seen_images)

        if 'start_time' in loader.keys():
            self.start_time = loader['start_time']

        if 'end_time' in loader.keys():
            self.end_time = loader['end_time']
