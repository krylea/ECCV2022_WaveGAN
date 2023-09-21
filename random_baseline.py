import os
import random
import shutil

import cv2
import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms
from trainer import Trainer
from utils import get_config, unloader, get_model_list

from fid import calculate_fid_given_paths

def fid(real, fake, gpu):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    #command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)
    command = 'python -m pytorch_fid {} {}'.format(real, fake)
    os.system(command)


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
    print(np.mean(res))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="flower")
parser.add_argument('--real_dir', type=str, default="results/flower_wavegan_base_index/reals")
parser.add_argument('--fake_dir', type=str,default="results/flower_wavegan_base_index/tests")
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_sample_test', type=int, default=3)
parser.add_argument('--num', type=int, default=-1)
parser.add_argument('--invert_rgb', action='store_true')
parser.add_argument('--eval_backbone', type=str, default='inception')
args = parser.parse_args()

conf_file = os.path.join('configs', "%s_lofgan.yaml" % args.dataset)
config = get_config(conf_file)
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)


if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    #torch.cuda.manual_seed(SEED)

    real_dir = args.real_dir
    fake_dir = args.fake_dir
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)

    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)
    os.makedirs(fake_dir, exist_ok=True)

    data = np.load(config['data_root'])
    if args.dataset == 'flower':
        data = data[85:]
        num = 10
    elif args.dataset == 'animal':
        data = data[119:]
        num = 10
    elif args.dataset == 'vggface':
        data = data[1802:]
        num = 30

    if args.num > 0:
        num = args.num

    data_for_gen = data[:, :num, :, :, :]
    data_for_fid = data[:, num:, :, :, :]

    if not os.path.exists(real_dir):
        os.makedirs(real_dir, exist_ok=True)
        for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
            for i in range(data_for_fid.shape[1]):
                idx = i
                real_img = data_for_fid[cls, idx, :, :, :]
                if args.dataset == 'vggface':
                    real_img = real_img * 255
                real_img = Image.fromarray(np.uint8(real_img))
                real_img.save(os.path.join(real_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    if os.path.exists(fake_dir):
        for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
            for i in range(128):
                idx = np.random.choice(data_for_gen.shape[1], 1)
                fake_img = data_for_gen[cls, idx, :, :, :][0]
                if args.dataset == 'vggface':
                    fake_img = fake_img * 255
                fake_img = Image.fromarray(np.uint8(fake_img))
                fake_img.save(os.path.join(fake_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    invert_rgb = args.invert_rgb and (args.dataset == 'vggface')
    eval_backbone = "Inception_V3" if args.eval_backbone == 'inception' else "clip"

    name = "%s_%s" % (args.dataset, num)
    if invert_rgb:
        name += "invert"
    fid_out = "fid_scores.txt" if args.eval_backbone == 'inception' else "fid_clip_scores.txt"

    fid_score = calculate_fid_given_paths(real_dir, fake_dir, torch.device("cuda"), eval_backbone=eval_backbone, invert_rgb=invert_rgb)

    with open(fid_out, 'a') as f:
        f.write("%s:\t%f\n" % (name, fid_score))

    lpips_score = LPIPS(fake_dir)

    with open("lpips_scores.txt", 'a') as f:
        f.write("%s:\t%f\n" % (name, fid_score))

    #fid(real_dir, fake_dir, args.gpu)
    #LPIPS(fake_dir)
