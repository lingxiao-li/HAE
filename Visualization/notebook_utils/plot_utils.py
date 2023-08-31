import glob
import random
import sys
import os
import matplotlib.pyplot as plt
import torch
import geoopt.manifolds.stereographic.math as gmath
from argparse import Namespace
from PIL import Image
from notebook_utils.pmath import *



# necessary functions
def tensor2im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))

# rescale function
def rescale(target_radius, x):
    r_change = target_radius/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = x, k = torch.tensor(-1.0)))
    return gmath.mobius_scalar_mul(r = r_change, x = x, k = torch.tensor(-1.0))

# function for generating images with fixed radius (also contains raw geodesic images of 'shorten' images, and stretched images to boundary )
def geo_interpolate_fix_r(x,y,interval,target_radius):
    feature_geo = []
    feature_geo_normalized = []
    images_to_plot_raw_geo = []
    images_to_plot_target_radius = []
    images_to_plot_boundary = []
    dist_to_start = []
    target_radius_ratio = torch.tensor(target_radius/6.2126)
    geodesic_start_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = x, k = torch.tensor(-1.0))
    geodesic_end_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = y, k = torch.tensor(-1.0))
    for i in interval:
        # this is raw image on geodesic, instead of fixed radius
        feature_geo_current = gmath.geodesic(t = torch.tensor(i), x = geodesic_start_short, y = geodesic_end_short, k = torch.tensor(-1.0))

        # here we fix the radius and don't revert them now
        r_change = target_radius/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo.append(feature_geo_current)
        feature_geo_current_target_radius = gmath.mobius_scalar_mul(r = r_change, x = feature_geo_current, k = torch.tensor(-1.0))
        feature_geo_normalized.append(feature_geo_current_target_radius)
        dist = gmath.dist(geodesic_start_short, feature_geo_current_target_radius, k = torch.tensor(-1.0))
        dist_to_start.append(dist)
        #print(feature_geo_current_target_radius.norm())

        # here is to revert the feature to boundary
        r_change_to_boundary = 6.2126/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo_current_target_boundary = gmath.mobius_scalar_mul(r = r_change_to_boundary, x = feature_geo_current, k = torch.tensor(-1.0))
        #print(feature_geo_current_target_boundary.norm())

        # now codes do not affect outputs
        with torch.no_grad():
            image_raw_geo, _, _, _, _ = net.forward(x = feature_geo_current.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image, _, _, _, _ = net.forward(x = feature_geo_current_target_radius.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image_boundary, _, _, _, _ = net.forward(x = feature_geo_current_target_boundary.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        images_to_plot_raw_geo.append(image_raw_geo)
        images_to_plot_target_radius.append(image)
        images_to_plot_boundary.append(image_boundary)
    
    return images_to_plot_raw_geo, images_to_plot_target_radius, images_to_plot_boundary, dist_to_start

def geo_interpolate_fix_r_with_codes(x,y,interval,target_radius):
    # please use this with batch_size = 1
    feature_geo = []
    feature_geo_normalized = []
    images_to_plot_raw_geo = []
    images_to_plot_target_radius = []
    images_to_plot_boundary = []
    dist_to_start = []
    target_radius_ratio = torch.tensor(target_radius/6.2126)
    geodesic_start_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = x, k = torch.tensor(-1.0))
    geodesic_end_short = gmath.mobius_scalar_mul(r = target_radius_ratio, x = y, k = torch.tensor(-1.0))
    for i in interval:
        # this is raw image on geodesic, instead of fixed radius
        feature_geo_current = gmath.geodesic(t = torch.tensor(i), x = geodesic_start_short, y = geodesic_end_short, k = torch.tensor(-1.0))

        # here we fix the radius and don't revert them now
        r_change = target_radius/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo.append(feature_geo_current)
        feature_geo_current_target_radius = gmath.mobius_scalar_mul(r = r_change, x = feature_geo_current, k = torch.tensor(-1.0))
        feature_geo_normalized.append(feature_geo_current_target_radius)
        dist = gmath.dist(geodesic_start_short, feature_geo_current_target_radius, k = torch.tensor(-1.0))
        dist_to_start.append(dist)
        #print(feature_geo_current_target_radius.norm())

        # here is to revert the feature to boundary
        r_change_to_boundary = 6.2126/dist0(gmath.mobius_scalar_mul(r = torch.tensor(1), x = feature_geo_current, k = torch.tensor(-1.0)))
        feature_geo_current_target_boundary = gmath.mobius_scalar_mul(r = r_change_to_boundary, x = feature_geo_current, k = torch.tensor(-1.0))
        #print(feature_geo_current_target_boundary.norm())

        # now codes do not affect outputs
        with torch.no_grad():
            image_raw_geo, _, _, _, _ = net.forward(x = feature_geo_current.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image, _, _, codes_target_radius, _ = net.forward(x = feature_geo_current_target_radius.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            image_boundary, _, _, codes_boundary, _ = net.forward(x = feature_geo_current_target_boundary.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
        images_to_plot_raw_geo.append(image_raw_geo)
        images_to_plot_target_radius.append(image)
        images_to_plot_boundary.append(image_boundary)
    
    return images_to_plot_raw_geo, images_to_plot_target_radius, images_to_plot_boundary, dist_to_start, [codes_target_radius, codes_boundary, feature_geo_current_target_radius, feature_geo_current_target_boundary]


def generate_perturbation_r_with_raw_inv(x, target_radius, interval, seed, size):
    # 3 arguments, raw image feature, target radius and interval(actually the ratio).
    images_perturbed = []
    dist_perturbed = []
    torch.manual_seed(seed = seed)
    perturb = torch.rand(6,512).cuda()
    for i in range(size):
        target_rad_perturb = 6.2126
        ratio = target_rad_perturb/dist0(perturb[i])
        perturb_current = gmath.mobius_scalar_mul(r = ratio, x = perturb[i], k = torch.tensor(-1.0))


        if False:
            with torch.no_grad():
                image, _, _, _, _ = net.forward(x = perturb.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            
            fig = plt.figure(figsize = (5,5))
            gs = fig.add_gridspec(1, 1)
            for i in range(1):
                if i == 0:
                    fig.add_subplot(gs[0,i])
                    plt.axis('off')
                    plt.title(f'perturb = {target_rad_perturb}')
                    plt.imshow(tensor2im(image.squeeze(0)))

        #interval = [0.42]
        _, images_to_plot_target_radius, _, dist_to_start = geo_interpolate_fix_r(x = x,y = perturb_current, interval = interval ,target_radius = target_radius)
        print(dist_to_start)
        dist_perturbed.append(dist_to_start[0])
        images_perturbed.append(images_to_plot_target_radius[0])

    raw_image,_,_,_ = geo_interpolate_fix_r(x = x,y = perturb_current, interval = [0] ,target_radius = target_radius)
    images_perturbed.insert(0, raw_image[0])
    return images_perturbed, dist_perturbed

def generate_perturbation_r_with_raw_inv_pick(x, y, target_radius, interval, seed):
    # 3 arguments, raw image feature, target radius and interval(actually the ratio).
    images_perturbed = []
    dist_perturbed = []
    torch.manual_seed(seed = seed)
    #perturb = torch.rand(6,512).cuda()
    perturb = y
    for i in range(len(y)):
        target_rad_perturb = 6.2126
        ratio = target_rad_perturb/dist0(perturb[i])
        perturb_current = gmath.mobius_scalar_mul(r = ratio, x = perturb[i], k = torch.tensor(-1.0))


        if False:
            with torch.no_grad():
                image, _, _, _, _ = net.forward(x = perturb_current.unsqueeze(0), codes=None, batch_size = 1, input_feature=True, input_code = False)
            
            fig = plt.figure(figsize = (5,5))
            gs = fig.add_gridspec(1, 1)
            for i in range(1):
                if i == 0:
                    fig.add_subplot(gs[0,i])
                    plt.axis('off')
                    plt.title(f'perturb = {target_rad_perturb}')
                    plt.imshow(tensor2im(image.squeeze(0)))

        #interval = [0.42]
        _, images_to_plot_target_radius, _, dist_to_start = geo_interpolate_fix_r(x = x,y = perturb_current, interval = interval ,target_radius = target_radius)
        print(dist_to_start)
        dist_perturbed.append(dist_to_start[0])
        images_perturbed.append(images_to_plot_target_radius[0])

    raw_image,_,_,_ = geo_interpolate_fix_r(x = x,y = perturb_current, interval = [0] ,target_radius = target_radius)
    images_perturbed.insert(0, raw_image[0])
    return images_perturbed, dist_perturbed