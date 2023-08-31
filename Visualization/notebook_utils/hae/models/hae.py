"""
This file defines the core research contribution
"""
from audioop import bias
import matplotlib
matplotlib.use('Agg')
import math
import geoopt.manifolds.stereographic.math as gmath

import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.stylegan2.model import Generator
from models.hyper_nets import MobiusLinear, HyperbolicMLR
from configs.paths_config import model_paths


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt

class EqualLinear_encoder(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_encoder, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim*3, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, in_dim*2)
        self.fc3 = nn.Linear(in_dim*2, in_dim)
        self.fc4 = nn.Linear(in_dim, in_dim)
        self.fc5 = nn.Linear(in_dim, out_dim*2)
        self.fc6 = nn.Linear(out_dim*2, out_dim)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        return out

class EqualLinear_decoder(nn.Module):
    def __init__(
        self, in_dim, out_dim):
        super(EqualLinear_decoder, self).__init__()
        self.out_dim=out_dim
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_dim, in_dim*2)
        self.fc2 = nn.Linear(in_dim*2, out_dim)
        self.fc3 = nn.Linear(out_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, out_dim*2)
        self.fc5 = nn.Linear(out_dim*2, out_dim*2)
        self.fc6 = nn.Linear(out_dim*2, out_dim*3)
        self.nonlinearity = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, input):
        out = self.flat(input)
        out = self.fc1(out)
        out = self.nonlinearity(out)
        out = self.fc2(out)
        out = self.nonlinearity(out)
        out = self.fc3(out)
        out = self.nonlinearity(out)
        out = self.fc4(out)
        out = self.nonlinearity(out)
        out = self.fc5(out)
        out = self.nonlinearity(out)
        out = self.fc6(out)
        return out

class MLP_encoder(nn.Module):
    def __init__(self, dim):
        super(MLP_encoder, self).__init__()
        self.encoder0=EqualLinear_encoder(512, dim)
        self.encoder1=EqualLinear_encoder(512, dim)
        
    def forward(self, dw):
        x0=self.encoder0(dw[:, :3])
        x1=self.encoder1(dw[:, 3:6])
        output_dw = torch.cat((x0, x1), dim=1)
        return output_dw

class MLP_decoder(nn.Module):
	def __init__(self, dim):
		super(MLP_decoder, self).__init__()
		self.dim = dim
		self.encoder0=EqualLinear_decoder(dim, 512)
		self.encoder1=EqualLinear_decoder(dim, 512)	
  
	def forward(self, dw):
		shape = dw[:, :self.dim].shape
		x0=self.encoder0(dw[:, :self.dim])
		x1=self.encoder1(dw[:, self.dim:])
		dw0 = x0.reshape((shape[0], 3, 512))
		dw1 = x1.reshape((shape[0], 3, 512))
		output_dw = torch.cat((dw0, dw1), dim=1)
		return output_dw
		


class hae(nn.Module):

	def __init__(self, opts):
		super(hae, self).__init__()
		self.set_opts(opts)
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		# Define architecture
		self.encoder = self.set_encoder()
		self.feature_shape = self.opts.feature_size
		if self.opts.dataset_type == 'flowers_encode':
			self.num_classes = 102 #animal_faces 151, flowers 102
		elif self.opts.dataset_type == 'animalfaces_encode':
			self.num_classes = 151
		else:
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		self.mlp_encoder = MLP_encoder(128)
		self.mlp_decoder = MLP_decoder(128)
		#self.mlp = MLP(512)
		self.hyperbolic_linear = MobiusLinear(self.feature_shape,
											  self.feature_shape,
											  # This computes an exmap0 after the operation, where the linear
											  # operation operates in the Euclidean space.
											  hyperbolic_input=False,
											  hyperbolic_bias=True,
											  nonlin=None,  # For now
											)
		self.mlr = HyperbolicMLR(ball_dim=self.feature_shape, n_classes=self.num_classes, c=1)
		self.decoder = Generator(self.opts.output_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_encoder(self):
		if self.opts.encoder_type == 'GradualStyleEncoder':
			encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
		elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
			encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se', self.opts)
		else:
			raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
		return encoder

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading HAE from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location=torch.device('cpu'))
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in ckpt['state_dict'].items():
				name = k.replace('module.','') 
				new_state_dict[name] = v
			ckpt['state_dict'] = new_state_dict
			self.hyperbolic_linear.load_state_dict(get_keys(ckpt, 'hyperbolic_linear'), strict=True)
			self.mlr.load_state_dict(get_keys(ckpt, 'mlr'), strict=True)
			self.mlp_encoder.load_state_dict(get_keys(ckpt, 'mlp_encoder'), strict=True)
			self.mlp_decoder.load_state_dict(get_keys(ckpt, 'mlp_decoder'), strict=True)
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)
		else:
			print('Loading pSp from checkpoint: {}'.format(self.opts.psp_checkpoint_path))
			ckpt = torch.load(self.opts.psp_checkpoint_path, map_location=torch.device('cpu'))
			from collections import OrderedDict
			new_state_dict = OrderedDict()
			for k, v in ckpt['state_dict'].items():
				name = k.replace('.module','') 
				new_state_dict[name] = v
			ckpt['state_dict'] = new_state_dict
			self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
			self.__load_latent_avg(ckpt)

	def forward(self, x, batch_size=4, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.encoder(x)
			# normalize with respect to the center of an average face
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

		ocodes = codes[:, :6]
		feature = self.mlp_encoder(ocodes)
		feature_reshape = torch.flatten(feature, start_dim=1)
		feature_dist = self.hyperbolic_linear(feature_reshape)
		logits = F.log_softmax(self.mlr(feature_dist, self.mlr.c), dim=-1)
		feature_euc = gmath.logmap0(feature_dist, k=torch.tensor(-1.))
		feature_euc = self.mlp_decoder(feature_euc)
		codes = torch.cat((feature_euc, codes[:, 6:]), dim=1)


		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent, logits, feature_dist, ocodes, feature_euc
		else:
			return images, logits, feature_dist, ocodes, feature_euc

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
