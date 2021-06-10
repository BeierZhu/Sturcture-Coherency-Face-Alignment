import os
import argparse
from metrics import MiscMeter
import numpy as np
from scipy import integrate

def parse_args():
    parser = argparse.ArgumentParser(
            description='implementation of COFW Dataset Analysis')
    parser.add_argument('--nme_path', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    nme_path = args.nme_path

    FR = MiscMeter()
   
    thres = 0.1

    with open(nme_path) as f:
        nmes = f.readlines()

    for nme in nmes:
        if nme in ['', '\n']: continue

        nme_elements = nme.strip('\n').strip(' ').split(' ')
        nme_value = float(nme_elements[0])

        if nme_value > thres:
            print(nme_elements[1])
            FR.update(1)
        else:
            FR.update(0)

    # save result
    save_path = os.path.join(os.path.dirname(nme_path),'analysis.txt')

    with open(save_path,'w') as f:
        full_string = "FR: {:.4f} NUM: {}".format(FR.avg, FR.count)
        print(full_string)
        f.write(full_string + '\n')


if __name__ == '__main__':
    main()

