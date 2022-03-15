import argparse
import numpy as np

from dataset import Dataset
from pass_flow import PassFlow


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path',
                        default='data/train.txt',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--test_path',
                        default='data/test.txt',
                        help='Path to training data file (one password per line) (default: data/test.txt)')

    parser.add_argument('--max_length',
                        type=int,
                        default=10,
                        help='The maximum password length (default: 10).')

    parser.add_argument('--epoch',
                        type=int,
                        default=100,
                        help='Number of epochs to train the model (default: 50).')

    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='Batch size (default: 128).')

    parser.add_argument('--architecture',
                        default='resnet',
                        help='Architecture of S and T functions used in the coupling layers (default: resnet)')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate (default: 5e-4).')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-5,
                        help='L2 regularization (only applied to the weight norm scale factors. Default: 5e-5).')

    parser.add_argument('--resume',
                        action='store',
                        help='Resume from checkpoint. File containing the saved model to resume training.')

    parser.add_argument('--test',
                        action='store',
                        help='Load the model for evaluation. File containing the saved model.')

    parser.add_argument('--ds',
                        action='store',
                        help='Use DS. Default False.')

    parser.add_argument('--gs',
                        action='store',
                        help='Use DS+GS. Default False.')

    return parser.parse_args()


args = parse_args()
max_train_sizes = 300000
max_test_size = 2000000

data = Dataset(args.train_path,
               args.test_path,
               args.max_length,
               max_train_size=max_train_sizes,
               max_test_size=max_test_size,
               skip_unk=True)

flow = PassFlow(args.max_length, data, args.architecture, args.lr, args.weight_decay)
flow.model.to(flow.device)

runs = 1
total_matches = []
if args.test:
    flow.load(args.test)
    for _ in range(runs):
        matches, unique = flow.evaluate_sampling(n_samples=10 ** 6, max_batch_size=10 ** 4)
        total_matches.append(len(matches))
    print(f'Found {np.average(total_matches)} matches.')
elif args.ds:
    flow.load(args.ds)
    for _ in range(runs):
        matches, unique = flow.dynamic_sampling(n_samples=10 ** 6, max_batch_size=10 ** 4, sigma=0.12, alpha=5, gamma=3)
        total_matches.append(len(matches))
    print(f'Found {np.average(total_matches)} matches.')
elif args.gs:
    flow.load(args.gs)
    for _ in range(runs):
        matches, unique = flow.dynamic_sampling_gs(n_samples=10 ** 6, max_batch_size=10 ** 4, sigma=0.12, alpha=5, gamma=3)
        total_matches.append(len(matches))
    print(f'Found {np.average(total_matches)} matches.')
else:
    if args.resume:
        flow.load(args.resume)
    else:
        flow.train_model(args.epoch, args.batch_size, n_samples=10 ** 6)
