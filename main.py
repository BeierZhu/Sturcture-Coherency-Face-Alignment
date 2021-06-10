import os
import argparse
import yaml
from pprint import pprint
from fld import FLD

def parse_args():
    """
    parse args
    :return:args
    """
    new_parser = argparse.ArgumentParser(
        description='PyTorch Facial Landmark Detector parser..')
    new_parser.add_argument('--config', default='')
    new_parser.add_argument('--load_path', type=str, default=None)
    new_parser.add_argument('--resume', action='store_true')
    new_parser.add_argument('--expname', type=str, default=None)
    new_parser.add_argument('--visualize', action='store_true')
    # exclusive arguments
    group = new_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--evaluate', action='store_true')
    group.add_argument('--eval_ckpts', action='store_true')

    return new_parser.parse_args()


def main():
    # parse args and load config
    args = parse_args()    
    with open(args.config) as f:
        config = yaml.load(f)
        
    for k, v in vars(args).items():
        config[k] = v
    pprint(config)
    
    agent = FLD(config)

    if args.evaluate:
        agent.evaluate()
    elif args.eval_ckpts:
        agent.eval_ckpts()
    elif args.train:
        agent.train()
    else:
        raise Warning("Invalid Args")

if __name__ == '__main__':
    main()
