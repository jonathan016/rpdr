"""
File Description
(CODENAME: 15_224)
-----------------
Model: MDRY2 (Darknet-33)
Optimizer: Adam
    - Learning rate: 0.0001
    - Weight decay: 0.0005
Loss function: CrossEntropyLoss
    - Reduction: mean (default)
Training:
    - Duration: 100 epochs
    - Dataset:
        * 150 item usage
        * 16 batch size

Additional information:
- Grayscale images and maintain number of image channels
- No learning rate decay
"""

import os
import sys
import time
from argparse import ArgumentParser
from random import randint, uniform

import torch
import torch.cuda as cuda
from PIL import ImageFilter
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, ColorJitter, Normalize, Lambda, \
    RandomResizedCrop, RandomErasing, RandomRotation, RandomPerspective, ToPILImage


def maybe_blur(image):
    return image.filter(ImageFilter.BoxBlur(randint(0, 7)))


def maybe_random_crop(image):
    if randint(0, 100) <= 35:
        return RandomResizedCrop(size=image.size, scale=(0.5, 1.0), ratio=(1., 1.))(image)
    return image


def maybe_random_erase(image):
    if randint(0, 100) <= 7:
        return ToPILImage()(RandomErasing(p=1.)(ToTensor()(image)))
    return image


def maybe_rotate(image):
    if randint(0, 100) <= 4:
        r = randint(0, 100)
        if r <= 20:
            return RandomRotation(degrees=90)(image)
        elif r <= 50:
            return RandomRotation(degrees=45)(image)
        else:
            return RandomRotation(degrees=30)(image)
    return image


def maybe_random_perspective(image):
    if randint(0, 100) <= 2:
        return RandomPerspective(distortion_scale=randint(4, 10) / 10, p=1.)(image)
    return image


def maybe_darken_a_lot(image):
    if randint(0, 100) <= 30:
        brightness = uniform(.5, .8)
        saturation = uniform(1., 1.5)
        return ColorJitter(brightness=(brightness, brightness), saturation=(saturation, saturation))(image)
    return image


def out(value):
    global logfile

    if logfile:
        print(value, file=open(logfile, 'a'))
    else:
        print(value)


def train(model, optimizer, criterion, epoch, train_data, val_data, save_file, best_file):
    global force_cuda

    if os.path.exists(best_file):
        out('Loading best model')
        loader = torch.load(best_file)
        model.load_state_dict(loader['state_dict'])
        best_val_acc = loader['best_val']
    else:
        best_val_acc = None

    model.train()

    total_loss = 0.0
    total_acc = 0.0
    total_img = 0
    iteration_losses = []

    for i, data in enumerate(train_data):
        inputs, labels = data
        if force_cuda and cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predictions = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        iteration_losses.append(loss.data.item())
        total_loss += loss.data.item()
        total_acc += torch.sum(predictions == labels.data).item()
        total_img += labels.size(0)

        del inputs, labels, outputs, predictions
        cuda.empty_cache()

    out(f'Training #{epoch}: {total_acc / total_img} accuracy and {total_loss / total_img} loss')
    val_loss, val_acc = eval(model, criterion, val_data)
    out(f'Validation #{epoch}: {val_acc} accuracy and {val_loss} loss')

    if best_val_acc is None or best_val_acc < val_acc:
        best_val_acc = val_acc
        torch.save({
            'state_dict': model.state_dict(),
            'best_val': best_val_acc
        }, best_file)

    # Save per epoch
    saved_iteration_losses = iteration_losses
    if os.path.exists(save_file):
        saved_iteration_losses = torch.load(save_file)['iteration_losses']
        saved_iteration_losses.extend(iteration_losses)

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'iteration_losses': saved_iteration_losses,
        'last_epoch': epoch
    }, save_file)

    return model, optimizer, criterion


def eval(model, criterion, loader):
    global force_cuda

    model.eval()

    loss = 0.0
    acc = 0.0
    total = 0

    for i, data in enumerate(loader):
        with torch.no_grad():
            inputs, labels = data
            if force_cuda and cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss += loss.data.item()
            acc += torch.sum(prediction == labels.data).item()
            total += labels.size(0)

            del inputs, labels, outputs, prediction
            cuda.empty_cache()

    avg_loss = loss / total
    avg_acc = acc / total

    return avg_loss, avg_acc


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', required=True, help='Location of where to find this project\'s modules')
    parser.add_argument('-t', default='./rpdr-config-results/data/cropped', help='Training data root')
    parser.add_argument('-e', default='./rpdr-config-results/data/in_situ_jpgs', help='Evaluation data root')
    parser.add_argument('-ei', default='./rpdr/val_test/recog_val_test.json', help='Evaluation indices path')
    parser.add_argument('-ef', default='./rpdr/val_test/recog_val_test_classes_files.json',
                        help='Evaluation classes-files path')
    parser.add_argument('-l', default='./rpdr-config-results/results/mdry2/15.log', help='Where to save logs')
    parser.add_argument('-w', default='./rpdr-config-results/results/mdry2/15_224.pth.tar',
                        help='Epoch checkpoint output path')
    parser.add_argument('-b', default='./rpdr-config-results/results/mdry2/15_224_best_model.pth.tar',
                        help='Weight output path')
    parser.add_argument('-nc', action='store_false')

    args = parser.parse_args()

    # Allows for importing modules of this project
    sys.path.append(args.m)
    from models.proposed import MDRY2
    from utils.datasets import UsageBasedDataset, RecognitionDataset

    train_root = args.t
    eval_root = args.e
    eval_indices = args.ei
    eval_files = args.ef
    logfile = args.l
    save = args.w
    best = args.b
    force_cuda = args.nc

    # Construct model, optimizer, and criterion
    darknet33: MDRY2 = MDRY2()
    darknet33.recognizing()

    if force_cuda and cuda.is_available():
        darknet33 = darknet33.cuda()
    adam = Adam(darknet33.parameters(), lr=.0001, weight_decay=.0005)
    crit = CrossEntropyLoss()

    # Construct dataset variables
    image_resolution = (224, 224)

    # Construct training dataset and loader
    train_transform = Compose([
        Lambda(maybe_blur),
        Lambda(maybe_darken_a_lot),
        Lambda(maybe_rotate),
        Lambda(maybe_random_perspective),
        Lambda(maybe_random_crop),
        Lambda(maybe_random_erase),
        ColorJitter(brightness=(.1, .8), contrast=.05, saturation=.05, hue=.005),
        Resize(image_resolution),
        Grayscale(num_output_channels=3),
        ToTensor(),
        Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    train_dataset = UsageBasedDataset(train_root, usage=150, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)

    # Construct evaluation transformation
    eval_transform = Compose([
        Resize(image_resolution), Grayscale(num_output_channels=3), ToTensor(), Normalize((.5, .5, .5), (.5, .5, .5))])

    # Construct validation dataset and loader
    val_dataset = RecognitionDataset(
        eval_root, eval_indices, eval_files, RecognitionDataset.VAL, transform=eval_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Start training
    start_time = time.time()
    if os.path.exists(save):
        saved_checkpoint = torch.load(save)
        start_epoch = saved_checkpoint['last_epoch'] + 1
        adam.load_state_dict(saved_checkpoint['optimizer'])
        crit.load_state_dict(saved_checkpoint['criterion'])
    else:
        start_epoch = 1

    for epoch in range(start_epoch, 101):
        darknet33, adam, crit = train(darknet33, adam, crit, epoch, train_loader, val_loader, save, best)
    end_time = time.time()
    out(f'Initial recognition training elapsed for {end_time - start_time} seconds')

    # Construct test dataset and loader
    test_dataset = RecognitionDataset(
        eval_root, eval_indices, eval_files, RecognitionDataset.TEST, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Test model
    darknet33.load_state_dict(torch.load(best)['state_dict'])
    test_loss, test_acc = eval(darknet33, crit, test_loader)
    out('\n=========')
    out(f'Test Average Loss: {test_loss}')
    out(f'Test Average Accuracy: {test_acc}')
