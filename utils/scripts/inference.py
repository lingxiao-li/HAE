import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys
import geoopt.manifolds.stereographic.math as gmath

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run():
    test_opts = TestOptions().parse()
    torch.cuda.set_device(0)
    if test_opts.resize_factors is not None:
        assert len(
            test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results',
                                        'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled',
                                        'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    # run this if only using 1 gpu...
    opts['device'] = 'cuda:0'

    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    results = []
    image_paths = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            x, label = input_batch
            input_cuda, label_cuda = x.cuda().float(), label.cuda()
            tic = time.time()
            result_batch = run_on_batch(input_cuda, net, opts)
            toc = time.time()
            global_time.append(toc - tic)
        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[0][i])
            im_path_g = dataset.paths[global_i]
            im_path = im_path_g.split('/')[-2] + '_' + im_path_g.split('/')[-1]
            origin = torch.zeros(2).cuda().float()
            #print(result_batch[2][i])
            #print(gmath.dist(origin, result_batch[2][i], k=torch.tensor(-1.)))
            np.set_printoptions(linewidth=5000)
            #print(str(result_batch[2][i].cpu().numpy()))
            #print(im_path)
            results.append(str(result_batch[2][i].cpu().numpy()) + ' ' + str(label[i].cpu().numpy()))
            image_paths.append(im_path)

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[0][i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    # for super resolution, save the original, down-sampled, and output
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)),
                                          np.array(input_im.resize(resize_amount, resample=Image.NEAREST)),
                                          np.array(result.resize(resize_amount))], axis=1)
                else:
                    # otherwise, save the original and output
                    res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                          np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1


    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    results_path = os.path.join(opts.exp_dir, 'data.txt')
    image_path = os.path.join(opts.exp_dir, 'data.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))

    print(results)

    with open(stats_path, 'w') as f:
        f.write(result_str)
    with open(results_path, 'w') as f:
        for i in results:
            f.write(i + '\n')
    with open(image_path, 'w') as f:
        for i in image_paths:
            f.write(i + '\n')


def run_on_batch(inputs, net, opts):
    if opts.latent_mask is None:
        result_batch = net(inputs, randomize_noise=False, resize=opts.resize_outputs)
    else:
        latent_mask = [int(l) for l in opts.latent_mask.split(",")]
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject, hyper_feature = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject,
                      alpha=opts.mix_alpha,
                      resize=opts.resize_outputs)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)
    return result_batch


if __name__ == '__main__':
    run()
