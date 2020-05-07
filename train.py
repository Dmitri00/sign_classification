import datetime
import os
import time
import torch
import torch.utils.data
import torchvision
from torch import nn

import utils
from dataset_import.import_fake_dataset import import_data as import_fake_data
from dataset_import.import_traffic_sign import import_data
from engines.evaluator import Evaluator
from engines.metrics import get_metrics
from engines.setup_optim import setup_optim
from engines.trainer import Trainer
from logger.logger import Logger
from models.net import Net


def load_data(root, batch_size, num_workers):
    # Data loading code
    print("Loading data")

    print("Loading training data")
    st = time.time()
    datasets, dataloaders = import_fake_data(root, batch_size, num_workers)
    print("Took", time.time() - st)
    return datasets, dataloaders


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print(args)

    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')

    torch.backends.cudnn.benchmark = True
    datasets, dataloaders = load_data(args.data_path, batch_size=args.batch_size, num_workers=args.workers)

    print("Creating model")
    model = Net(num_classes=datasets['train'].num_classes)

    criterion, optimizer, lr_scheduler = setup_optim(model, args)
    logger = Logger(len(dataloaders['train']))
    trainer = Trainer(model, criterion, optimizer, lr_scheduler, device, logger, args.print_freq)
    metrics = get_metrics(criterion)
    evaluator = Evaluator(trainer.trainer_engine, model, metrics, dataloaders['val'], logger)
    if args.test_only:
        evaluator.run()
        return

    print("Start training")
    start_time = time.time()
    trainer.run(dataloaders['train'], args.epochs)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', help='dataset')
    parser.add_argument('--model', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet'
                        )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
