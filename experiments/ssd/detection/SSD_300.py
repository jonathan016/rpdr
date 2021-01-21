"""
File Description
(CODENAME: D6_300)
-----------------
Model: RPSSD300
Optimizer: SGD
    - Learning rate: 0.001
    - Momentum: 0.9
    - Weight decay: 0.0005
Training:
    - Duration: 50 epochs
    - Dataset:
        * 200 maximum data usage
        * 32 batch size:
            - 32 on DataLoader
            - 1 accumulations then update

Additional information:
- Cluttered background noise on combined image
- Using shelf background
"""

import os
import sys
import time
from argparse import ArgumentParser
from random import randint, uniform

import torch
import torch.cuda as cuda
from PIL import ImageFilter
from torch.optim import SGD
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


def change_optimizer_learning_rate(optimizer, total_batch_count, steps, scale):
    for param_group in optimizer.param_groups:
        for step in steps:
            if total_batch_count >= step:
                param_group['lr'] = param_group['lr'] * scale

    return optimizer


def train(model, optimizer, epoch, train_data, val_containing_data, val_non_containing_data, save_file, best_file,
          confidence_threshold, nms_threshold, keep_predictions, update_per_x_accumulations, steps, scale):
    global force_cuda

    if os.path.exists(save_file):
        loader = torch.load(save_file)
        model.loss_function.load_state_dict(loader['loss_state_dict'])
        iteration_losses = torch.load(save_file)['iteration_losses']
    else:
        iteration_losses = []

    model.train()

    total_loss = 0.0

    optimizer.zero_grad()

    break_train_limit = (len(train_data) // update_per_x_accumulations) * update_per_x_accumulations

    batch_loss = 0.

    for i, data in enumerate(train_data, start=1):
        if i > break_train_limit:
            # Prevents accumulating loss without back-propagating
            break

        inputs, target_locs, target_cls = data
        if force_cuda and cuda.is_available():
            inputs = inputs.cuda()
            target_locs = [loc.cuda() for loc in target_locs]
            target_cls = [cls.cuda() for cls in target_cls]

        model(inputs)
        loss = model.loss(target_locs, target_cls)
        loss.backward()

        batch_loss += loss.data.item()

        iteration_losses.append(loss.data.item())
        total_loss += loss.data.item()

        batch_count_by_data_loader = len(iteration_losses)
        if batch_count_by_data_loader % update_per_x_accumulations == 0:
            total_batch_count = batch_count_by_data_loader / update_per_x_accumulations
            out(f'Batch #{int(total_batch_count)}: {batch_loss}')
            change_optimizer_learning_rate(optimizer, total_batch_count, steps, scale)

            optimizer.step()
            optimizer.zero_grad()

            batch_loss = 0.

        del inputs, target_locs, target_cls
        cuda.empty_cache()

    total_batch = len(train_data) // update_per_x_accumulations
    out(f'Training #{epoch}: {total_loss} total loss, {total_loss / total_batch} average loss')

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_state_dict': model.loss_function.state_dict(),
        'iteration_losses': iteration_losses,
        'last_epoch': epoch
    }, save_file)

    if epoch % 5 == 0:
        validate_and_possibly_save_best(
            model, epoch, best_file, val_containing_data, val_non_containing_data, confidence_threshold, nms_threshold,
            keep_predictions)

    return model, optimizer


def validate_and_possibly_save_best(model, epoch, best_file, val_containing_data, val_non_containing_data,
                                    confidence_threshold, nms_threshold, keep_predictions):
    if os.path.exists(best_file):
        loader = torch.load(best_file)
        best_val_mean_average_precision = loader['mAP']
        best_val_overall_recall = loader['overall_recall']
        best_val_overall_precision = loader['overall_precision']
    else:
        best_val_mean_average_precision = None
        best_val_overall_recall = None
        best_val_overall_precision = None

    # Training data as validation data
    trainval_containing_dataset = EvalGroZiDetectionDataset(
        train_root, model=EvalGroZiDetectionDataset.SSD, transform=train_transform,
        combined_transform=combined_transform, min_resize=20, max_resize=480, max_data_usage=15,
        max_data_in_combined=10, do_flip_augment=False, background_noise='./rpdr-config-results/data/backgrounds')
    trainval_containing_data = DataLoader(
        trainval_containing_dataset, batch_size=1, collate_fn=ssd_collate, shuffle=False)
    trainval_non_containing_dataset = EvalGroZiDetectionDataset(
        train_root, model=EvalGroZiDetectionDataset.SSD, transform=train_transform,
        combined_transform=combined_transform, min_resize=20, max_resize=480, max_data_usage=2,
        max_data_in_combined=10, do_flip_augment=False, background_noise='./rpdr-config-results/data/backgrounds')
    trainval_non_containing_data = DataLoader(
        trainval_non_containing_dataset, batch_size=1, collate_fn=ssd_collate, shuffle=False)

    trainval_predictions, trainval_targets, trainval_detect_fps, trainval_forward_fps = eval(
        model, trainval_containing_data, trainval_non_containing_data, confidence_threshold, nms_threshold,
        keep_predictions)
    trainval_iou50, trainval_iou25, trainval_iou10, trainval_by_box_centroid = calculate_metrics(
        trainval_predictions, trainval_targets)

    # Real evaluation data
    val_predictions, val_targets, val_detect_fps, val_forward_fps = eval(
        model, val_containing_data, val_non_containing_data, confidence_threshold, nms_threshold, keep_predictions)
    val_iou50, val_iou25, val_iou10, val_by_box_centroid = calculate_metrics(val_predictions, val_targets)

    # Print values
    out(f'Validation #{epoch}: {val_detect_fps} FPS, {val_forward_fps} Forward FPS')
    out(f'Validation on TrainVal #{epoch}: {trainval_detect_fps} FPS, {trainval_forward_fps} Forward FPS')

    box_centroid_mAP = val_by_box_centroid.mean_average_precision
    box_centroid_overall_recall = val_by_box_centroid.overall_recall
    box_centroid_overall_precision = val_by_box_centroid.overall_precision

    out('mAP:\t\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_iou50.mean_average_precision * 100, val_iou25.mean_average_precision * 100,
        val_iou10.mean_average_precision * 100, box_centroid_mAP * 100))
    out('Overall Recall:\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_iou50.overall_recall * 100, val_iou25.overall_recall * 100, val_iou10.overall_recall * 100,
        box_centroid_overall_recall * 100))
    out('Overall Precision:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        val_iou50.overall_precision * 100, val_iou25.overall_precision * 100, val_iou10.overall_precision * 100,
        box_centroid_overall_precision * 100))
    out('mAP on TrainVal:\t\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        trainval_iou50.mean_average_precision * 100, trainval_iou25.mean_average_precision * 100,
        trainval_iou10.mean_average_precision * 100, trainval_by_box_centroid.mean_average_precision * 100))
    out('Overall Recall on TrainVal:\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        trainval_iou50.overall_recall * 100, trainval_iou25.overall_recall * 100, trainval_iou10.overall_recall * 100,
        trainval_by_box_centroid.overall_recall * 100))
    out('Overall Precision on TrainVal:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        trainval_iou50.overall_precision * 100, trainval_iou25.overall_precision * 100,
        trainval_iou10.overall_precision * 100, trainval_by_box_centroid.overall_precision * 100))

    # Save best model
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


def calculate_metrics(predictions, targets):
    global force_cuda

    iou50 = DetectionMetric(DetectionMetric.SSD, 120)
    iou25 = DetectionMetric(DetectionMetric.SSD, 120)
    iou10 = DetectionMetric(DetectionMetric.SSD, 120)
    by_box_centroid = DetectionByBoxCenterPointMetric(DetectionMetric.SSD, 120)

    iou50.calculate_metrics(predictions, targets, iou_threshold=.5, ssd_is_cuda=force_cuda and cuda.is_available())
    iou25.calculate_metrics(predictions, targets, iou_threshold=.25, ssd_is_cuda=force_cuda and cuda.is_available())
    iou10.calculate_metrics(predictions, targets, iou_threshold=.1, ssd_is_cuda=force_cuda and cuda.is_available())
    by_box_centroid.calculate_metrics(predictions, targets, ssd_is_cuda=force_cuda and cuda.is_available())

    return iou50, iou25, iou10, by_box_centroid


def eval(model, eval_containing_loader, eval_non_containing_loader, confidence_threshold, nms_threshold,
         keep_predictions):
    global force_cuda

    model.eval()

    total_fps = []
    total_forward_fps = []
    predictions = ValueContainer(ValueContainer.SSD)
    targets = ValueContainer(ValueContainer.SSD)

    for i, data in enumerate(eval_containing_loader):
        with torch.no_grad():
            input_image, target_locs, target_cls = data
            if force_cuda and cuda.is_available():
                input_image = input_image.cuda()
                target_locs = [loc.cuda() for loc in target_locs]
                target_cls = [cls.cuda() for cls in target_cls]

            _, forward_fps = operation_per_second(model, [input_image])
            total_forward_fps.append(forward_fps)

            arguments = [input_image, confidence_threshold, nms_threshold, keep_predictions]
            output, fps = operation_per_second(model.detect_objects, arguments)
            boxes, labels, scores = output
            total_fps.append(fps)

            if (i + 1) % 400 == 0:
                out(f'Containing #{i + 1} Output:\n{boxes, labels, scores}')
                out(f'Containing #{i + 1} Target:\n{target_locs, target_cls}\n')

            predictions['boxes'].extend(boxes)
            predictions['labels'].extend(labels)
            predictions['scores'].extend(scores)
            targets['boxes'].extend(target_locs)
            targets['labels'].extend(target_cls)

            del data, target_locs, target_cls, output
            cuda.empty_cache()

    for i, data in enumerate(eval_non_containing_loader):
        with torch.no_grad():
            input_image, target_locs, target_cls = data
            if force_cuda and cuda.is_available():
                input_image = input_image.cuda()
                target_locs = [loc.cuda() for loc in target_locs]
                target_cls = [cls.cuda() for cls in target_cls]

            _, forward_fps = operation_per_second(model, [input_image])
            total_forward_fps.append(forward_fps)

            arguments = [input_image, confidence_threshold, nms_threshold, keep_predictions]
            output, fps = operation_per_second(model.detect_objects, arguments)
            boxes, labels, scores = output
            total_fps.append(fps)

            if (i + 1) % 4000 == 0:
                out(f'Non Containing #{i + 1} Output:\n{boxes, labels, scores}')
                out(f'Non Containing #{i + 1} Target:\n{target_locs, target_cls}\n')

            predictions['boxes'].extend(boxes)
            predictions['labels'].extend(labels)
            predictions['scores'].extend(scores)
            targets['boxes'].extend(target_locs)
            targets['labels'].extend(target_cls)

            del data, target_locs, target_cls, output
            cuda.empty_cache()

    detect_fps = sum(total_fps) / len(total_fps)
    forward_fps = sum(total_forward_fps) / len(total_forward_fps)

    return predictions, targets, detect_fps, forward_fps


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
    parser.add_argument('-l', default='./rpdr-config-results/results/ssd/D6_300.log', help='Where to save logs')
    parser.add_argument('-w', default='./rpdr-config-results/results/ssd/D6_300.pth.tar',
                        help='Epoch checkpoint output path')
    parser.add_argument('-b', default='./rpdr-config-results/results/ssd/D6_300_best_model.pth.tar',
                        help='Weight output path')
    parser.add_argument('-bb', default='./rpdr-config-results/results/ssd/base/3_300_best_model.pth.tar',
                        help='Backbone weight')
    parser.add_argument('-tmp', default='./rpdr-config-results/results/ssd/D6_300_test_metric.pth.tar',
                        help='Test metric output path')
    parser.add_argument('-th', default=.01, type=float, help='Confidence threshold for testing')
    parser.add_argument('-tkp', default=20, type=int, help='Total predictions to keep (maximum limit)')
    parser.add_argument('-nc', action='store_false')
    parser.add_argument('-ft', default=False, help='Force test only without training. Best weight file must exist')

    args = parser.parse_args()

    # Allows for importing modules of this project
    sys.path.append(args.m)
    from models.proposed import RPSSD300
    from utils.datasets import GroZiDetectionDataset, DetectionRecognitionDataset, EvalGroZiDetectionDataset, \
        ssd_collate
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
    total_keep_predictions = args.tkp
    force_cuda = args.nc
    force_test_only = args.ft

    # Construct model, optimizer, and criterion
    rpssd300: RPSSD300 = RPSSD300(torch.load(backbone, map_location=torch.device('cpu'))['state_dict'])
    if force_cuda and cuda.is_available():
        rpssd300 = rpssd300.cuda()

    if not force_test_only:
        # Initialize training parameters
        epochs = 50
        data_loader_batch_size = 32
        subdivision = 1
        max_data_usage = 200
        data_in_combined = (3, 10)

        # Construct training dataset and loader
        train_transform = Compose([
            Lambda(maybe_blur),
            Lambda(maybe_darken_a_lot),
            Lambda(maybe_rotate),
            Lambda(maybe_random_perspective),
            Lambda(maybe_random_crop),
            Lambda(maybe_random_erase),
            ColorJitter(brightness=(.1, .8), contrast=.05, saturation=.05, hue=.005),
            ToTensor(),
            Normalize((.5, .5, .5), (.5, .5, .5)),
            ToPILImage()
        ])
        combined_transform = Compose([Grayscale(num_output_channels=3)])
        train_dataset = GroZiDetectionDataset(
            train_root, model=GroZiDetectionDataset.SSD, transform=train_transform,
            combined_transform=combined_transform, min_resize=50, max_resize=200, max_data_usage=max_data_usage,
            data_in_combined=data_in_combined, do_flip_augment=False,
            background_noise=('./rpdr-config-results/data/backgrounds', './rpdr-config-results/data/shelf_backgrounds'))
        train_loader = DataLoader(train_dataset, batch_size=data_loader_batch_size, shuffle=True, pin_memory=True,
                                  collate_fn=ssd_collate, drop_last=True)

        # Initialize additional parameters depending on training data size
        estimated_batches_per_epoch = len(train_loader) / subdivision
        estimated_batches = epochs * estimated_batches_per_epoch
        steps = [int(estimated_batches * (2 / 3)), int(estimated_batches * (5 / 6))]
        scale = .1

        # Initialize optimizer
        biases, not_biases = _get_model_parameters(rpssd300)
        sgd = SGD(params=[{'params': biases, 'lr': 2 * .001}, {'params': not_biases}], lr=.001, momentum=.9,
                  weight_decay=.0005)

        # Construct evaluation transformation
        eval_transform = Compose([Resize((300, 300)), Grayscale(num_output_channels=3), ToTensor()])

        # Construct validation dataset and loader
        val_containing_dataset = DetectionRecognitionDataset(
            eval_root, DetectionRecognitionDataset.FileLocations(val_containing_indices, coordinates_path),
            DetectionRecognitionDataset.SSD, resize_to=(300, 300), transform=eval_transform, is_containing=True)
        val_containing_loader = DataLoader(val_containing_dataset, batch_size=1, collate_fn=ssd_collate, shuffle=False)

        val_non_containing_dataset = DetectionRecognitionDataset(
            eval_root, DetectionRecognitionDataset.FileLocations(val_non_containing_indices, coordinates_path),
            DetectionRecognitionDataset.SSD, resize_to=(300, 300), transform=eval_transform, is_containing=False)
        val_non_containing_loader = DataLoader(
            val_non_containing_dataset, batch_size=1, collate_fn=ssd_collate, shuffle=False)

        # Start training
        start_time = time.time()
        if os.path.exists(save):
            saved_checkpoint = torch.load(save)
            start_epoch = saved_checkpoint['last_epoch'] + 1
            sgd.load_state_dict(saved_checkpoint['optimizer'])
        else:
            start_epoch = 1

        for epoch in range(start_epoch, epochs + 1):
            rpssd300, sgd = train(
                rpssd300, sgd, epoch, train_loader, val_containing_loader, val_non_containing_loader, save, best,
                confidence_threshold=threshold, nms_threshold=.4, keep_predictions=total_keep_predictions,
                update_per_x_accumulations=subdivision, steps=steps, scale=scale)
        end_time = time.time()
        out(f'Detection and recognition training elapsed for {end_time - start_time} seconds')

    eval_transform = Compose([Resize((300, 300)), Grayscale(num_output_channels=3), ToTensor()])
    # Construct test dataset and loader
    test_containing_dataset = DetectionRecognitionDataset(
        eval_root, DetectionRecognitionDataset.FileLocations(test_containing_indices, coordinates_path),
        DetectionRecognitionDataset.SSD, resize_to=(300, 300), transform=eval_transform, is_containing=True)
    test_containing_loader = DataLoader(test_containing_dataset, batch_size=1, collate_fn=ssd_collate, shuffle=False)

    test_non_containing_dataset = DetectionRecognitionDataset(
        eval_root, DetectionRecognitionDataset.FileLocations(test_non_containing_indices, coordinates_path),
        DetectionRecognitionDataset.SSD, resize_to=(300, 300), transform=eval_transform, is_containing=False)
    test_non_containing_loader = DataLoader(
        test_non_containing_dataset, batch_size=1, collate_fn=ssd_collate, shuffle=False)

    # Test model
    rpssd300.load_state_dict(torch.load(best)['state_dict'])

    predictions, targets, test_detect_fps, test_forward_fps = eval(
        rpssd300, test_containing_loader, test_non_containing_loader, confidence_threshold=threshold,
        nms_threshold=.4, keep_predictions=total_keep_predictions)
    metric_iou50, metric_iou25, metric_iou10, metric_by_box_centroid = calculate_metrics(predictions, targets)

    out('\n=========')
    out(f'Using confidence threshold {threshold}')
    out('\n=========')
    out(f'FPS: {test_detect_fps} FPS')
    out(f'FPS: {test_forward_fps} Forward FPS')

    out('mAP:\t\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        metric_iou50.mean_average_precision * 100, metric_iou25.mean_average_precision * 100,
        metric_iou10.mean_average_precision * 100, metric_by_box_centroid.mean_average_precision * 100))
    out('Overall Recall:\t\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        metric_iou50.overall_recall * 100, metric_iou25.overall_recall * 100, metric_iou10.overall_recall * 100,
        metric_by_box_centroid.overall_recall * 100))
    out('Overall Precision:\t@50 {0:08.5f}, @25 {1:08.5f}, @10 {2:08.5f}, @Center {3:08.5f}'.format(
        metric_iou50.overall_precision * 100, metric_iou25.overall_precision * 100,
        metric_iou10.overall_precision * 100, metric_by_box_centroid.overall_precision * 100))
    torch.save({
        '@50': metric_iou50,
        '@25': metric_iou25,
        '@10': metric_iou10,
        '@Center': metric_by_box_centroid
    }, test_metric_path)
