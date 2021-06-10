import os
import argparse
from metrics import MiscMeter
import numpy as np
from scipy import integrate

def parse_args():
    parser = argparse.ArgumentParser(
            description='implementation of WFLW Dataset Analysis')
    parser.add_argument('--nme_path', type=str, required=True)
    parser.add_argument('--att_path', type=str, required=False, default='/mnt/lustre/share/zhubeier/WFLW/test/test_98pt_attr.txt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    nme_path = args.nme_path
    att_path = args.att_path

    Full = MiscMeter()
    Pose = MiscMeter()
    Expr = MiscMeter()
    Illu = MiscMeter()
    MkUp = MiscMeter()
    Occl = MiscMeter()
    Blur = MiscMeter()

    Full_FR = MiscMeter()
    Pose_FR = MiscMeter()
    Expr_FR = MiscMeter()
    Illu_FR = MiscMeter()
    MkUp_FR = MiscMeter()
    Occl_FR = MiscMeter()
    Blur_FR = MiscMeter()

    thres = 0.1

    Subset = [Pose, Expr, Illu, MkUp, Occl, Blur]
    Subset_FR = [Pose_FR, Expr_FR, Illu_FR, MkUp_FR, Occl_FR, Blur_FR]
    Name = ['Pose', 'Expr','Illu', 'MkUp', 'Occl', 'Blur']
    with open(nme_path) as f:
        nmes = f.readlines()

    with open(att_path) as f:
        atts = f.readlines()

    thres_list = np.linspace(0.005, 0.1, 20, endpoint=True)
    thres_full = np.zeros_like(thres_list)
    thres_subset = [np.zeros_like(thres_list),
                    np.zeros_like(thres_list),
                    np.zeros_like(thres_list),
                    np.zeros_like(thres_list),
                    np.zeros_like(thres_list),
                    np.zeros_like(thres_list)]

    for nme, att in zip(nmes, atts):
        if nme in ['', '\n']: continue
        if att in ['', '\n']: continue

        nme_elements = nme.strip('\n').strip(' ').split(' ')
        att_elements = att.strip('\n').strip(' ').split(' ')

        file_name = att_elements[-1].split('/')[-1]
        # print(file_name)
        # print(nme_elements[-1])
        # assert(file_name in nme_elements[-1])

        att_list = [int(element) for element in att_elements[:-1]]
        nme_value = float(nme_elements[0])
        Full.update(nme_value)

        if nme_value <= 0.1:
            thres_idx = np.where(nme_value<= thres_list)[0][0]
            thres_full[thres_idx] +=1

        if nme_value > thres:
            Full_FR.update(1)
        else:
            Full_FR.update(0)

        for i in range(len(Subset)):
            att = [0]*6
            att[i] = 1
            
            if att_list==att:
                Subset[i].update(nme_value)
                if nme_value <= 0.1:
                    thres_idx = np.where(nme_value<= thres_list)[0][0]
                    thres_subset[i][thres_idx] +=1

                if nme_value > thres:
                    Subset_FR[i].update(1)
                else:
                    Subset_FR[i].update(0) 

    thres_full /= Full.count
    thres_full = np.cumsum(thres_full)
    auc = np.trapz(y=thres_full, x=thres_list)*10
    print('Full AUC {:.4f}'.format(auc))
    for i in range(len(Name)):
        thres_subset[i] /= Subset[i].count
        thres_subset[i] = np.cumsum(thres_subset[i])
        auc = np.trapz(y=thres_subset[i], x=thres_list)*10
        print("{} AUC {:.4f}".format(Name[i], auc))
    # save result
    save_path = os.path.join(os.path.dirname(nme_path),'analysis.txt')

    with open(save_path,'w') as f:
        full_string = "Full ION: {:.4f} NUM: {}".format(Full.avg, Full.count)
        print(full_string)
        f.write(full_string + '\n')

        for i in range(len(Subset)):
            subset_string = "{} ION: {:.4f} NUM: {}".format(Name[i], Subset[i].avg, Subset[i].count)
            print(subset_string)
            f.write(subset_string + '\n')

        full_string = "Full FR: {:.4f} NUM: {}".format(Full_FR.avg, Full_FR.count)
        print(full_string)
        f.write(full_string + '\n')

        for i in range(len(Subset)):
            subset_string = "{} FR: {:.4f} NUM: {}".format(Name[i], Subset_FR[i].avg, Subset_FR[i].count)
            print(subset_string)
            f.write(subset_string + '\n')



    



if __name__ == '__main__':
    main()

