from argparse import ArgumentParser
from torch.cuda import is_available as torch_cuda_is_available
from yaml import safe_load

from runners import RPSSD300Runner
from utils.datasets import GroZiDetectionDataset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, help='YAML configuration file location')
    parser.add_argument('--logfile', default=None, help='Where to save all outputs during training')
    parser.add_argument('--resume', default=None, help='Checkpoint path to resume training')
    parser.add_argument('--no_load_resume', default=None,
                        help='Any keys to not be loaded when resuming training, separated by comma (,)')

    args = parser.parse_args()

    yaml_config = safe_load(open(args.config, 'r'))

    network_config = yaml_config['network']
    start_iter, end_iter = max(0, int(network_config['start_iter'])), max(0, int(network_config['end_iter']))
    decay_lr_iterations = list(map(int, network_config['decay_lr']['iterations']))
    gamma = float(network_config['decay_lr']['gamma'])
    pretrained = network_config['pretrained']
    use_cuda = network_config['cuda'] if network_config['cuda'] is not None else torch_cuda_is_available()

    optimizer_config = yaml_config['optimizer']

    dataset_config = yaml_config['dataset']
    item_source = dataset_config['source']
    batch_size = max(2, dataset_config['batch_size'])
    to_tensor = dataset_config['to_tensor']

    combined = dataset_config['combined']
    max_data_in_combined = max(1, int(combined['max_data']))
    max_data_usage = max(1, int(combined['max_usage']))
    min_resize, max_resize = int(combined['min_resize']), int(combined['max_resize'])

    savefile = yaml_config['savefile']

    logfile = args.logfile if args.logfile else None

    # Run runner
    model, optimizer = RPSSD300Runner.pre_init(load_pretrained_base=pretrained, **optimizer_config)
    dataset = GroZiDetectionDataset(root=item_source, model=GroZiDetectionDataset.SSD, max_data_usage=max_data_usage,
                                    max_data_in_combined=max_data_in_combined, transform_to_tensor=to_tensor,
                                    min_resize=min_resize, max_resize=max_resize)
    runner = RPSSD300Runner(model, optimizer, dataset, batch_size=batch_size, iterations=(start_iter, end_iter),
                            logfile=logfile, decay_lr_iterations=decay_lr_iterations, gamma=gamma, use_cuda=use_cuda)

    if args.resume is None:
        runner.train(savefile)
    else:
        if args.no_load_resume:
            no_load_resume = args.no_load_resume.split(',')
            runner.load(args.resume, no_load_resume)
        else:
            runner.load(args.resume)
        runner.train(savefile)
