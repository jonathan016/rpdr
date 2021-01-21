"""
File Description
(CODENAME: 3_416)
-----------------
Model: MDRY2
Optimizer: SGD
    - Learning rate: 0.001
    - Momentum: 0.9
    - Weight decay: 0.0005
Training:
    - Duration: 100 epochs
    - Dataset:
        * 300 maximum data usage
        * 64 batch size:
            - 8 on DataLoader
            - 8 accumulations then update
"""

import os
import sys
import time
from argparse import ArgumentParser
from random import randint

import torch
import torch.cuda as cuda
from PIL import ImageFilter
from torch import LongTensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Lambda, ColorJitter, ToPILImage, Normalize


def out(value):
    global logfile

    if logfile:
        print(value, file=open(logfile, 'a'))
    else:
        print(value)


def change_optimizer_learning_rate(optimizer, learning_rate, total_batch_count, real_batch_size, burn_in, power,
                                   steps_scales):
    new_learning_rate = learning_rate
    if total_batch_count < burn_in:
        new_learning_rate = learning_rate * ((total_batch_count / burn_in) ** power)

    for step, scale in steps_scales:
        if total_batch_count >= step:
            new_learning_rate = learning_rate * scale

    # In @pjreddie's implementation, the division is on the learning rate as can be seen on lines 545 and 553
    # on `convolutional_layer.c`'s `update_convolutional_layer` function. The sample code from line 545,
    # `axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);`, is actually used for updating the
    # convolutional layer parameter (in this case, bias) by `learning_rate/batch`. This is multiplied to each
    # bias in the layer, as can be seen on `blas.c`'s `axpy_cpu` function in lines 178-182.
    new_learning_rate /= real_batch_size

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_learning_rate

    return optimizer


def train(model, optimizer, epoch, train_data, val_containing_data, val_non_containing_data, save_file, best_file,
          confidence_threshold, nms_threshold, update_per_x_accumulations, original_learning_rate,
          real_batch_size, burn_in, power, steps_scales):
    global force_cuda

    if os.path.exists(save_file):
        loader = torch.load(save_file)
        seen_images = loader['seen_images']

        model.large_predictor_loss.layer.use_cuda = loader['large_loss_cuda']
        model.large_predictor_loss.layer.seen_images = seen_images

        model.medium_predictor_loss.layer.use_cuda = loader['medium_loss_cuda']
        model.medium_predictor_loss.layer.seen_images = seen_images

        model.small_predictor_loss.layer.use_cuda = loader['small_loss_cuda']
        model.small_predictor_loss.layer.seen_images = seen_images

    model.train()

    total_loss = 0.0
    iteration_losses = []

    optimizer.zero_grad()

    break_train_limit = (len(train_data) // update_per_x_accumulations) * update_per_x_accumulations

    batch_loss = 0.

    for i, data in enumerate(train_data, start=1):
        if i > break_train_limit:
            # Prevents accumulating loss without back-propagating
            break

        inputs, targets = data
        if force_cuda and cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()

        model(inputs)
        loss = model.loss(targets)
        loss.backward()

        batch_loss += loss.data.item()

        batch_count_by_data_loader = model.large_predictor_loss.layer.seen_images / train_data.batch_size
        if batch_count_by_data_loader % update_per_x_accumulations == 0:
            total_batch_count = batch_count_by_data_loader / update_per_x_accumulations
            out(f'Batch #{int(total_batch_count)}: {batch_loss}')
            change_optimizer_learning_rate(
                optimizer, original_learning_rate, total_batch_count, real_batch_size, burn_in, power, steps_scales)

            optimizer.step()
            optimizer.zero_grad()

            batch_loss = 0.

        iteration_losses.append(loss.data.item())
        total_loss += loss.data.item()

        del inputs, targets
        cuda.empty_cache()

    total_batch = len(train_data) // update_per_x_accumulations
    out(f'Training #{epoch}: {total_loss} total loss, {total_loss / total_batch} average loss')

    # Perform validation per 5 epochs since validation takes 1 hour and possibly save best performing model
    if epoch % 5 == 0:
        validate_and_possibly_save_best(
            model, epoch, best_file, val_containing_data, val_non_containing_data, confidence_threshold, nms_threshold)

    # Save per epoch
    saved_iteration_losses = iteration_losses
    if os.path.exists(save_file):
        saved_iteration_losses = torch.load(save_file)['iteration_losses']
        saved_iteration_losses.extend(iteration_losses)

    assert model.large_predictor_loss.layer.seen_images == model.medium_predictor_loss.layer.seen_images == \
           model.small_predictor_loss.layer.seen_images
    torch.save({
        'model': model.state_dict(),
        'large_loss_cuda': model.large_predictor_loss.layer.use_cuda,
        'medium_loss_cuda': model.medium_predictor_loss.layer.use_cuda,
        'small_loss_cuda': model.small_predictor_loss.layer.use_cuda,
        'seen_images': model.large_predictor_loss.layer.seen_images,
        'optimizer': optimizer.state_dict(),
        'iteration_losses': saved_iteration_losses,
        'last_epoch': epoch
    }, save_file)

    return model, optimizer


def validate_and_possibly_save_best(model, epoch, best_file, val_containing_data, val_non_containing_data,
                                    confidence_threshold, nms_threshold):
    if os.path.exists(best_file):
        loader = torch.load(best_file)
        best_val_mean_average_precision = loader['mAP']
        best_val_overall_recall = loader['overall_recall']
        best_val_overall_precision = loader['overall_precision']
    else:
        best_val_mean_average_precision = None
        best_val_overall_recall = None
        best_val_overall_precision = None

    val_containing_predictions, val_containing_targets, val_predictions, val_targets, val_detect_fps, val_forward_fps \
        = eval(model, val_containing_data, val_non_containing_data, confidence_threshold, nms_threshold)
    val_containing_iou50, val_containing_iou25, val_containing_iou10, val_containing_by_box_centroid, val_iou50, \
    val_iou25, val_iou10, val_by_box_centroid = calculate_metrics(
        val_containing_predictions, val_containing_targets, val_predictions, val_targets)

    box_centroid_mAP = val_by_box_centroid.mean_average_precision
    box_centroid_overall_recall = val_containing_by_box_centroid.overall_recall
    box_centroid_overall_precision = val_containing_by_box_centroid.overall_precision

    out(f'Validation #{epoch}: {val_detect_fps} FPS, {val_forward_fps} Forward FPS')

    out('Containing mAP:\t\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_containing_iou50.mean_average_precision * 100, val_containing_iou25.mean_average_precision * 100,
        val_containing_iou10.mean_average_precision * 100, val_containing_by_box_centroid.mean_average_precision * 100))
    out('Containing Overall Recall:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_containing_iou50.overall_recall * 100, val_containing_iou25.overall_recall * 100,
        val_containing_iou10.overall_recall * 100, box_centroid_overall_recall * 100))
    out('Containing Overall Precision:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_containing_iou50.overall_precision * 100, val_containing_iou25.overall_precision * 100,
        val_containing_iou10.overall_precision * 100, box_centroid_overall_precision * 100))

    out('Full mAP:\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_iou50.mean_average_precision * 100, val_iou25.mean_average_precision * 100,
        val_iou10.mean_average_precision * 100, box_centroid_mAP * 100))
    out('Full Overall Recall:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_iou50.overall_recall * 100, val_iou25.overall_recall * 100, val_iou10.overall_recall * 100,
        val_by_box_centroid.overall_recall * 100))
    out('Full Overall Precision:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_iou50.overall_precision * 100, val_iou25.overall_precision * 100, val_iou10.overall_precision * 100,
        val_by_box_centroid.overall_precision * 100))

    if best_val_mean_average_precision is None:
        torch.save({
            'state_dict': model.state_dict(),
            'mAP': box_centroid_mAP,
            'overall_recall': box_centroid_overall_recall,
            'overall_precision': box_centroid_overall_precision
        }, best_file)
    elif best_val_mean_average_precision < box_centroid_mAP:
        torch.save({
            'state_dict': model.state_dict(),
            'mAP': box_centroid_mAP,
            'overall_recall': box_centroid_overall_recall,
            'overall_precision': box_centroid_overall_precision
        }, best_file)
    else:
        has_higher_combined_value = (best_val_overall_recall + best_val_overall_precision) < (
                box_centroid_overall_recall + box_centroid_overall_precision)
        has_smaller_difference = abs(best_val_overall_recall - best_val_overall_precision) > abs(
            box_centroid_overall_recall - box_centroid_overall_precision)

        if has_higher_combined_value and has_smaller_difference:
            torch.save({
                'state_dict': model.state_dict(),
                'mAP': box_centroid_mAP,
                'overall_recall': box_centroid_overall_recall,
                'overall_precision': box_centroid_overall_precision
            }, best_file)


def calculate_metrics(containing_predictions, containing_targets, predictions, targets):
    containing_iou50 = DetectionMetric(DetectionMetric.YOLO, 120)
    containing_iou25 = DetectionMetric(DetectionMetric.YOLO, 120)
    containing_iou10 = DetectionMetric(DetectionMetric.YOLO, 120)
    containing_by_box_centroid = DetectionByBoxCenterPointMetric(DetectionMetric.YOLO, 120)

    containing_iou50.calculate_metrics(containing_predictions, containing_targets, iou_threshold=.5)
    containing_iou25.calculate_metrics(containing_predictions, containing_targets, iou_threshold=.25)
    containing_iou10.calculate_metrics(containing_predictions, containing_targets, iou_threshold=.1)
    containing_by_box_centroid.calculate_metrics(containing_predictions, containing_targets)

    iou50 = DetectionMetric(DetectionMetric.YOLO, 120)
    iou25 = DetectionMetric(DetectionMetric.YOLO, 120)
    iou10 = DetectionMetric(DetectionMetric.YOLO, 120)
    by_box_centroid = DetectionByBoxCenterPointMetric(DetectionMetric.YOLO, 120)

    iou50.calculate_metrics(predictions, targets, iou_threshold=.5)
    iou25.calculate_metrics(predictions, targets, iou_threshold=.25)
    iou10.calculate_metrics(predictions, targets, iou_threshold=.1)
    by_box_centroid.calculate_metrics(predictions, targets)

    return containing_iou50, containing_iou25, containing_iou10, containing_by_box_centroid, iou50, iou25, iou10, \
           by_box_centroid


def eval(model, eval_containing_loader, eval_non_containing_loader, confidence_threshold, nms_threshold):
    global force_cuda

    model.eval()

    total_fps = []
    total_forward_fps = []
    predictions = ValueContainer(ValueContainer.YOLO)
    targets = ValueContainer(ValueContainer.YOLO)

    for i, data in enumerate(eval_containing_loader):
        with torch.no_grad():
            data, target = data
            if force_cuda and cuda.is_available():
                data = data.cuda()

            _, forward_fps = operation_per_second(model, [data])
            total_forward_fps.append(forward_fps)

            arguments = [data, confidence_threshold, nms_threshold]
            output, fps = operation_per_second(model.detect_objects, arguments)
            total_fps.append(fps)

            if (i + 1) % 400 == 0:
                out(f'Containing #{i + 1} Output:\n{output}')
                out(f'Containing #{i + 1} Target:\n{[LongTensor(target).numpy().tolist()]}\n')

            predictions['values'].extend(output)
            targets['values'].append(LongTensor(target).numpy().tolist())

            del data, target, output
            cuda.empty_cache()

    containing_predictions, containing_targets = predictions, targets

    for i, data in enumerate(eval_non_containing_loader):
        with torch.no_grad():
            data, target = data
            if force_cuda and cuda.is_available():
                data = data.cuda()

            _, forward_fps = operation_per_second(model, [data])
            total_forward_fps.append(forward_fps)

            arguments = [data, confidence_threshold, nms_threshold]
            output, fps = operation_per_second(model.detect_objects, arguments)
            total_fps.append(fps)

            if (i + 1) % 4000 == 0:
                out(f'Non Containing #{i + 1}:\n{output}')
                out(f'Non Containing #{i + 1} Target:\n{[LongTensor(target).numpy().tolist()]}]\n')

            predictions['values'].extend(output)
            targets['values'].append(LongTensor(target).numpy().tolist())

            del data, target, output
            cuda.empty_cache()

    detect_fps = sum(total_fps) / len(total_fps)
    forward_fps = sum(total_forward_fps) / len(total_forward_fps)

    return containing_predictions, containing_targets, predictions, targets, detect_fps, forward_fps


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', default='./rpdr', help='Location of where to find this project\'s modules')
    parser.add_argument('-t', default='./rpdr-config-results/data/cropped', help='Training data root')
    parser.add_argument('-e', default='./rpdr-config-results/data/frames', help='Evaluation data root')
    parser.add_argument('-c', default='./rpdr/val_test/detect_val_test_coordinates.json',
                        help='Coordinates file path')
    parser.add_argument('-tc', default='./rpdr/val_test/detect_test_frames-containing.json',
                        help='Test frames (containing) file path')
    parser.add_argument('-tnc', default='./rpdr/val_test/detect_test_frames-not_containing.json',
                        help='Test frames (containing) file path')
    parser.add_argument('-vc', default='./rpdr/val_test/detect_val_frames-containing.json',
                        help='Validation frames (containing) file path')
    parser.add_argument('-vnc', default='./rpdr/val_test/detect_val_frames-not_containing.json',
                        help='Validation frames (containing) file path')
    parser.add_argument('-l', default='./rpdr-config-results/results/mdry2/3_416.log', help='Where to save logs')
    parser.add_argument('-w', default='./rpdr-config-results/results/mdry2/3_416.pth.tar',
                        help='Epoch checkpoint output path')
    parser.add_argument('-b', default='./rpdr-config-results/results/mdry2/3_416_best_model.pth.tar',
                        help='Weight output path')
    parser.add_argument('-bb', default='./rpdr-config-results/results/mdry2/11_224_best_model.pth.tar',
                        help='Backbone weight')
    parser.add_argument('-tmp', default='./rpdr-config-results/results/mdry2/3_416_test_metric.pth.tar',
                        help='Test metric output path')
    parser.add_argument('-th', default=.004, type=float, help='Confidence threshold for testing')
    parser.add_argument('-nc', action='store_false')
    parser.add_argument('-ft', default=False, help='Force test only without training. Best weight file must exist')

    args = parser.parse_args()

    # Allows for importing modules of this project
    sys.path.append(args.m)
    from models.proposed import MDRY2
    from utils.datasets import GroZiDetectionDataset, DetectionRecognitionDataset
    from utils.metrics import DetectionMetric, DetectionByBoxCenterPointMetric, ValueContainer, operation_per_second

    train_root = args.t
    eval_root = args.e
    coordinates_path = args.c
    test_containing_indices = args.tc
    test_non_containing_indices = args.tnc
    val_containing_indices = args.vc
    val_non_containing_indices = args.vnc
    logfile = args.l
    save = args.w
    best = args.b
    backbone = args.bb
    test_metric_path = args.tmp
    threshold = args.th
    force_cuda = args.nc
    force_test_only = args.ft

    # Construct model, optimizer, and criterion
    mdry2: MDRY2 = MDRY2()
    mdry2.recognizing()

    if force_cuda and cuda.is_available():
        mdry2 = mdry2.cuda()
    # Load backbone weight
    if not force_test_only:
        mdry2.load_state_dict(torch.load(backbone)['state_dict'])

    # Set model to detecting state
    mdry2.detecting()
    if force_cuda and cuda.is_available():
        mdry2 = mdry2.cuda()

    if not force_test_only:
        # Initialize training parameters
        data_loader_batch_size = 8
        subdivision = 8
        optimizer_initial_learning_rate = .001
        burn_in = 1000
        power = 4
        epochs = 100
        max_data_usage = 300
        max_data_in_combined = 10

        # Construct training dataset and loader
        train_transform = Compose([
            Lambda(lambda x: x.filter(ImageFilter.BoxBlur(randint(0, 7)))),
            ColorJitter(brightness=(.1, .8), contrast=.05, saturation=.05, hue=.005),
            ToTensor(),
            Normalize((.5, .5, .5), (.5, .5, .5)),
            ToPILImage()
        ])
        train_dataset = GroZiDetectionDataset(train_root, model=GroZiDetectionDataset.YOLO, transform=train_transform,
                                              min_resize=70, max_resize=200, max_data_usage=max_data_usage,
                                              data_in_combined=max_data_in_combined, max_object=15,
                                              seen_images=0, batch_size=data_loader_batch_size)
        train_loader = DataLoader(train_dataset, batch_size=data_loader_batch_size, shuffle=True, pin_memory=True,
                                  drop_last=True, num_workers=4)

        # Initialize additional parameters depending on training data size
        estimated_batches_per_epoch = len(train_loader) / subdivision
        steps = [int(epochs * .8 * estimated_batches_per_epoch), int(epochs * .9 * estimated_batches_per_epoch)]
        scales = [.1, .1]
        steps_scales = zip(steps, scales)
        real_batch_size = train_dataset.batch_size * subdivision
        optimizer_initial_learning_rate /= real_batch_size

        # Initialize optimizer
        sgd = SGD(mdry2.parameters(), lr=optimizer_initial_learning_rate, momentum=.9, weight_decay=.0005)

        # Construct evaluation transformation
        eval_transform = Compose([Resize((416, 416)), ToTensor()])

        # Construct validation dataset and loader
        val_containing_dataset = DetectionRecognitionDataset(
            eval_root, DetectionRecognitionDataset.FileLocations(val_containing_indices, coordinates_path),
            DetectionRecognitionDataset.YOLO, resize_to=(416, 416), transform=eval_transform, is_containing=True)
        val_containing_loader = DataLoader(val_containing_dataset, batch_size=1, shuffle=False)

        val_non_containing_dataset = DetectionRecognitionDataset(
            eval_root, DetectionRecognitionDataset.FileLocations(val_non_containing_indices, coordinates_path),
            DetectionRecognitionDataset.YOLO, resize_to=(416, 416), transform=eval_transform, is_containing=False)
        val_non_containing_loader = DataLoader(val_non_containing_dataset, batch_size=1, shuffle=False)

        # Start training
        start_time = time.time()
        if os.path.exists(save):
            saved_checkpoint = torch.load(save)
            start_epoch = saved_checkpoint['last_epoch'] + 1
            sgd.load_state_dict(saved_checkpoint['optimizer'])
            train_loader.dataset.seen_images = saved_checkpoint['seen_images']
        else:
            start_epoch = 1

        for epoch in range(start_epoch, epochs + 1):
            mdry2, sgd = train(
                mdry2, sgd, epoch, train_loader, val_containing_loader, val_non_containing_loader, save, best,
                confidence_threshold=threshold, nms_threshold=.4, update_per_x_accumulations=subdivision,
                original_learning_rate=optimizer_initial_learning_rate, real_batch_size=real_batch_size,
                burn_in=burn_in, power=power, steps_scales=steps_scales)
        end_time = time.time()
        out(f'Detection and recognition training elapsed for {end_time - start_time} seconds')

    eval_transform = Compose([Resize((416, 416)), ToTensor()])
    # Construct test dataset and loader
    test_containing_dataset = DetectionRecognitionDataset(
        eval_root, DetectionRecognitionDataset.FileLocations(test_containing_indices, coordinates_path),
        DetectionRecognitionDataset.YOLO, resize_to=(416, 416), transform=eval_transform, is_containing=True)
    test_containing_loader = DataLoader(test_containing_dataset, batch_size=1, shuffle=False)

    test_non_containing_dataset = DetectionRecognitionDataset(
        eval_root, DetectionRecognitionDataset.FileLocations(test_non_containing_indices, coordinates_path),
        DetectionRecognitionDataset.YOLO, resize_to=(416, 416), transform=eval_transform, is_containing=False)
    test_non_containing_loader = DataLoader(test_non_containing_dataset, batch_size=1, shuffle=False)

    # Test model
    mdry2.load_state_dict(torch.load(best)['state_dict'])

    containing_predictions, containing_targets, predictions, targets, test_detect_fps, test_forward_fps = eval(
        mdry2, test_containing_loader, test_non_containing_loader, confidence_threshold=threshold, nms_threshold=.4)
    containing_metric_iou50, containing_metric_iou25, containing_metric_iou10, \
    containing_metric_by_box_centroid, metric_iou50, metric_iou25, metric_iou10, metric_by_box_centroid = \
        calculate_metrics(containing_predictions, containing_targets, predictions, targets)

    out('\n=========')
    out(f'Using confidence threshold {threshold}')
    out('\n=========')
    out(f'FPS: {test_detect_fps} FPS')
    out(f'FPS: {test_forward_fps} Forward FPS')
    out('Containing mAP:\t\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        containing_metric_iou50.mean_average_precision * 100, containing_metric_iou25.mean_average_precision * 100,
        containing_metric_iou10.mean_average_precision * 100,
        containing_metric_by_box_centroid.mean_average_precision * 100))
    out('Containing Overall Recall:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        containing_metric_iou50.overall_recall * 100, containing_metric_iou25.overall_recall * 100,
        containing_metric_iou10.overall_recall * 100, containing_metric_by_box_centroid.overall_recall * 100))
    out('Containing Overall Precision:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        containing_metric_iou50.overall_precision * 100, containing_metric_iou25.overall_precision * 100,
        containing_metric_iou10.overall_precision * 100, containing_metric_by_box_centroid.overall_precision * 100))

    out('Full mAP:\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        metric_iou50.mean_average_precision * 100, metric_iou25.mean_average_precision * 100,
        metric_iou10.mean_average_precision * 100, metric_by_box_centroid.mean_average_precision * 100))
    out('Full Overall Recall:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        metric_iou50.overall_recall * 100, metric_iou25.overall_recall * 100, metric_iou10.overall_recall * 100,
        metric_by_box_centroid.overall_recall * 100))
    out('Full Overall Precision:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        metric_iou50.overall_precision * 100, metric_iou25.overall_precision * 100,
        metric_iou10.overall_precision * 100, metric_by_box_centroid.overall_precision * 100))
    torch.save({
        'containing_only': {
            '@50': containing_metric_iou50,
            '@25': containing_metric_iou25,
            '@10': containing_metric_iou10,
            '@Center': containing_metric_by_box_centroid,
        },
        'full': {
            '@50': metric_iou50,
            '@25': metric_iou25,
            '@10': metric_iou10,
            '@Center': metric_by_box_centroid,
        }
    }, test_metric_path)
