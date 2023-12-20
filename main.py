
import argparse


from utility.global_utils import print_args
from train.pretrain import run_pt
from train.inference import infer


def main(**kwargs):
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--seed', dest='seed', type=int, default=1234)
    # basic settings
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=-1)

    # model hyperparams
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='learning rate of adam')
    parser.add_argument('--vocab_size', dest='vocab_size', type=int, default=65239)
    parser.add_argument('--hidden_size', dest='hidden_size',
                        type=int, default=32)
    parser.add_argument('--num_hidden_layers', dest='num_hidden_layers',
                        type=int, default=2, help='number of epochs')
    parser.add_argument('--num_attention_heads', dest='num_attention_heads',
                        type=int, default=2, help='number of epochs')
    parser.add_argument('--intermediate_size', dest='intermediate_size',
                        type=int, default=32, help='number of epochs'),
    parser.add_argument('--epochs', dest='epochs',
                        type=int, default=1, help='number of epochs')


    # training strategies
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='number of batch_size')
    parser.add_argument('--print_result', dest='print_result',
                        type=bool, default=True)
    parser.add_argument('--is_infer', dest='is_infer',
                        type=bool, default=False)
    
    

    
    args = parser.parse_args()
    print_args(args)

    # run experiment
    if not args.is_infer:
        run_pt(args)
    # inference
    else:
        infer(args)


if __name__ == '__main__':
    main()
