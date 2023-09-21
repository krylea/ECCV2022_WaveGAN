


from fid import calculate_fid_given_paths
import lpips
import torch
import tqdm
import os
import cv2
import numpy as np
import argparse

def LPIPS(root):
    print('Calculating LPIPS...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = {}
    for file in tqdm(files, desc='loading data'):
        cls = file.split('_')[0]
        idx = int(file.split('_')[1][:-4])
        img = lpips.im2tensor(cv2.resize(lpips.load_image(os.path.join(root, file)), (32, 32)))
        data.setdefault(cls, {})[idx] = img

    classes = set([file.split('_')[0] for file in files])
    res = []
    for cls in tqdm(classes):
        temp = []
        files_cls = [file for file in files if file.startswith(cls + '_')]
        for i in range(0, len(files_cls) - 1, 1):
            # print(i, end='\r')
            for j in range(i + 1, len(files_cls), 1):
                img1 = data[cls][i].cuda()
                img2 = data[cls][j].cuda()

                d = model(img1, img2, normalize=True)
                temp.append(d.detach().cpu().numpy())
        res.append(np.mean(temp))
    return np.mean(res)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="flower")
parser.add_argument('--real_dir', type=str, default="results/flower_wavegan_base_index/reals")
parser.add_argument('--fake_dir', type=str,default="results/flower_wavegan_base_index/tests")
parser.add_argument('--num', type=int, default=-1)
parser.add_argument('--invert_rgb', action='store_true')
parser.add_argument('--eval_backbone', type=str, default='inception')
args = parser.parse_args()

real_dir = args.real_dir
fake_dir = args.fake_dir
if __name__ == '__main__':
    invert_rgb = args.invert_rgb and (args.dataset == 'vggface')
    eval_backbone = "Inception_V3" if args.eval_backbone == 'inception' else "clip"

    name = "%s_%s" % (args.dataset, args.num)
    if invert_rgb:
        name += "invert"
    fid_out = "fid_scores.txt" if args.eval_backbone == 'inception' else "fid_clip_scores.txt"

    fid_score = calculate_fid_given_paths(real_dir, fake_dir, torch.device("cuda"), eval_backbone=eval_backbone, invert_rgb=invert_rgb)

    with open(fid_out, 'a') as f:
        f.write("%s:\t%f\n" % (name, fid_score))

    lpips_score = LPIPS(fake_dir)

    with open("lpips_scores.txt", 'a') as f:
        f.write("%s:\t%f\n" % (name, fid_score))