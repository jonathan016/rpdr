from argparse import ArgumentParser

from torchvision.transforms import Compose, Resize, ToTensor
from yaml import safe_load
from torch.cuda import is_available as torch_cuda_is_available

from runners import RPYOLOv3Runner
from utils.datasets import GroZiDetectionDataset, get_recognition_dataset, RecognitionDataset


def train_initial(yaml_config, resume, no_load_resume, logfile):
    start_iter, end_iter = max(0, int(yaml_config['start_iter'])), max(0, int(yaml_config['end_iter']))
    resolution = max(224, int(yaml_config['resolution']))
    use_cuda = yaml_config['cuda'] if yaml_config['cuda'] is not None else torch_cuda_is_available()

    optimizer_config = yaml_config['optimizer']
    lr = optimizer_config['lr']
    momentum = optimizer_config['momentum']
    weight_decay = optimizer_config['weight_decay']

    dataset_config = yaml_config['dataset']
    item_source = dataset_config['source']
    batch_size = max(2, dataset_config['batch_size'])
    item_usage = max(1, dataset_config['item_usage']) if dataset_config['item_usage'] else None
    increase_by_factor = max(1, dataset_config['increase_by_factor']) if dataset_config['increase_by_factor'] else None

    validation_config = dataset_config['validation']
    validation_source = validation_config['source']
    indices_path = validation_config['indices_path']
    files_path = validation_config['files_path']

    savefile = yaml_config['savefile']

    logfile = logfile if logfile else None

    model, optimizer = RPYOLOv3Runner.pre_init({}, learning_rate=lr, momentum=momentum, decay=weight_decay,
                                               batch_size=batch_size)
    dataset = get_recognition_dataset(item_source, resolution, increase_by_factor, item_usage)
    val_dataset = RecognitionDataset(validation_source, indices_path, files_path, RecognitionDataset.VAL,
                                     transform=Compose([Resize((resolution, resolution)), ToTensor()]))
    runner = RPYOLOv3Runner(model, optimizer, dataset, val_dataset, iterations=(start_iter, end_iter),
                            batch_size=batch_size, use_cuda=use_cuda, initial_lr=lr, logfile=logfile)

    if resume is None:
        runner.recognizing(reset_time=True)
    else:
        runner.recognizing()
        if no_load_resume:
            no_load_resume = no_load_resume.split(',')
            runner.load(resume, no_load_resume)
        else:
            runner.load(resume)

    runner.train(savefile)


def train_finetune(yaml_config, resume, no_load_resume, logfile):
    start_iter, end_iter = max(0, int(yaml_config['start_iter'])), max(0, int(yaml_config['end_iter']))
    resolution = max(224, int(yaml_config['resolution']))
    backbone_weight = yaml_config['backbone_weight']
    use_cuda = yaml_config['cuda'] if yaml_config['cuda'] is not None else torch_cuda_is_available()

    optimizer_config = yaml_config['optimizer']
    lr = optimizer_config['lr']
    momentum = optimizer_config['momentum']
    weight_decay = optimizer_config['weight_decay']

    dataset_config = yaml_config['dataset']
    item_source = dataset_config['source']
    batch_size = max(2, dataset_config['batch_size'])
    item_usage = max(1, dataset_config['item_usage']) if dataset_config['item_usage'] else None
    increase_by_factor = max(1, dataset_config['increase_by_factor']) if dataset_config['increase_by_factor'] else None

    validation_config = dataset_config['validation']
    validation_source = validation_config['source']
    indices_path = validation_config['indices_path']
    files_path = validation_config['files_path']

    savefile = yaml_config['savefile']

    logfile = logfile if logfile else None

    model, optimizer = RPYOLOv3Runner.pre_init({}, learning_rate=lr, momentum=momentum, decay=weight_decay,
                                               batch_size=batch_size)
    dataset = get_recognition_dataset(item_source, resolution, increase_by_factor, item_usage)
    val_dataset = RecognitionDataset(validation_source, indices_path, files_path, RecognitionDataset.VAL,
                                     transform=Compose([Resize((resolution, resolution)), ToTensor()]))
    runner = RPYOLOv3Runner(model, optimizer, dataset, val_dataset, iterations=(start_iter, end_iter),
                            batch_size=batch_size, use_cuda=use_cuda, initial_lr=lr, logfile=logfile)

    if resume is None:
        runner.recognizing(reset_time=True, backbone_weight=backbone_weight)
    else:
        runner.recognizing(backbone_weight=backbone_weight)
        if no_load_resume:
            no_load_resume = no_load_resume.split(',')
            runner.load(resume, no_load_resume)
        else:
            runner.load(resume)

    runner.train(savefile)


def train_detecting(yaml_config, resume, no_load_resume, logfile):
    start_iter, end_iter = max(0, int(yaml_config['start_iter'])), max(0, int(yaml_config['end_iter']))
    decay_lr_iterations = list(map(int, yaml_config['decay_lr']['iterations']))
    decay_lr_scales = list(map(float, yaml_config['decay_lr']['scales']))
    backbone_weight = yaml_config['backbone_weight']
    use_cuda = yaml_config['cuda'] if yaml_config['cuda'] is not None else torch_cuda_is_available()

    anchor_boxes_config = yaml_config['anchor_boxes']
    large_anchor_boxes = list(map(int, anchor_boxes_config['large']))
    medium_anchor_boxes = list(map(int, anchor_boxes_config['medium']))
    small_anchor_boxes = list(map(int, anchor_boxes_config['small']))

    optimizer_config = yaml_config['optimizer']
    lr = optimizer_config['lr']
    momentum = optimizer_config['momentum']
    weight_decay = optimizer_config['weight_decay']

    dataset_config = yaml_config['dataset']
    item_source = dataset_config['source']
    batch_size = max(2, dataset_config['batch_size'])
    to_tensor = dataset_config['to_tensor']

    combined = dataset_config['combined']
    max_data_in_combined = max(1, int(combined['max_data']))
    max_data_usage = max(1, int(combined['max_usage']))
    min_resize, max_resize = int(combined['min_resize']), int(combined['max_resize'])

    savefile = yaml_config['savefile']

    logfile = logfile if logfile else None

    model, optimizer = RPYOLOv3Runner.pre_init({'anchor_boxes': anchor_boxes_config}, learning_rate=lr,
                                               momentum=momentum,
                                               decay=weight_decay, batch_size=batch_size)
    dataset = GroZiDetectionDataset(root=item_source, model=GroZiDetectionDataset.YOLO, seen_images=0,
                                    max_object=15, batch_size=batch_size, max_data_usage=max_data_usage,
                                    max_data_in_combined=max_data_in_combined, transform_to_tensor=to_tensor,
                                    min_resize=min_resize, max_resize=max_resize)
    runner = RPYOLOv3Runner(model, optimizer, dataset, validation_dataset=None, iterations=(start_iter, end_iter),
                            batch_size=batch_size, use_cuda=use_cuda, initial_lr=lr, logfile=logfile)

    if resume is None:
        runner.detecting(reset_time=True, backbone_weight=backbone_weight, large_anchor_boxes=large_anchor_boxes,
                         medium_anchor_boxes=medium_anchor_boxes, small_anchor_boxes=small_anchor_boxes,
                         decay_steps_scales=list(zip(decay_lr_iterations, decay_lr_scales)))
    else:
        runner.detecting(backbone_weight=backbone_weight, large_anchor_boxes=large_anchor_boxes,
                         medium_anchor_boxes=medium_anchor_boxes, small_anchor_boxes=small_anchor_boxes,
                         decay_steps_scales=list(zip(decay_lr_iterations, decay_lr_scales)))
        if no_load_resume:
            no_load_resume = no_load_resume.split(',')
            runner.load(resume, no_load_resume)
        else:
            runner.load(resume)
        dataset = GroZiDetectionDataset(root=item_source, model=GroZiDetectionDataset.YOLO,
                                        seen_images=runner.seen_images, max_object=15, batch_size=batch_size,
                                        max_data_usage=max_data_usage, max_data_in_combined=max_data_in_combined,
                                        transform_to_tensor=to_tensor, min_resize=min_resize, max_resize=max_resize)
        runner.change_dataset(dataset)

    runner.train(savefile)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, help='YAML configuration file location')
    parser.add_argument('--logfile', default=None, help='Where to save all outputs during training')
    parser.add_argument('--resume', default=None, help='Checkpoint path to resume training')
    parser.add_argument('--no_load_resume', default=None,
                        help='Any keys to not be loaded when resuming training, separated by comma (,)')
    parser.add_argument('-i', '--initial', action='store_true')
    parser.add_argument('-f', '--finetune', action='store_true')
    parser.add_argument('-d', '--detection', action='store_true')

    args = parser.parse_args()

    if not any([args.initial, args.finetune, args.detection]):
        raise ValueError('Either initial (-i), finetune (-f), or detection (-d) flag must be passed')

    config = safe_load(open(args.config, 'r'))

    if args.initial:
        train_initial(config['network_recognition_initial'], args.resume, args.no_load_resume, args.logfile)
    elif args.finetune:
        train_finetune(config['network_recognition_finetune'], args.resume, args.no_load_resume, args.logfile)
    elif args.detection:
        train_detecting(config['network_detection'], args.resume, args.no_load_resume, args.logfile)
