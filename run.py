import argparse

import torch

from network.net import Recurrent
from utils.config import IN_dim, HIDDEN_dim, LAYER_dim, OUT_dim
from utils.data import loaders, data_size
from utils.model import train_model, load_model, predict_model

torch.manual_seed(1)


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=100,
                        help='Training  Batch size  default: 100')
    parser.add_argument('--iters', type=int, default=6000,
                        help='Number of training iterations  default: 6000')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Model learning rate  default: 0.1')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping default: 1')
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  False: Train model default: True')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = Recurrent(in_dim=IN_dim,
                      hidden_dim=HIDDEN_dim,
                      layer_dim=LAYER_dim,
                      out_dim=OUT_dim)
    if args.load:
        model_name = 'model-coqqs.pkl'
        load_model(model, 'weights/{}'.format(model_name))
        while True:
            predict_model(model)

    else:
        n_batches = int(data_size / args.batch)
        n_epochs = int(args.iters / n_batches)

        train_ds_loader, test_ds_loader = loaders(args.batch)

        train_model(model,
                    train_ds_loader, test_ds_loader,
                    args.clip, args.lr,
                    n_epochs,
                    True)
