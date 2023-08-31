dataset_paths = {
	'celeba_train': '',
	'celeba_test': '/local/rcs/ll3504/datasets/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '/local/rcs/ll3504/datasets/FFHQ/resized',
	'flowers_train': '/local/rcs/ll3504/datasets/flowers/train',
	'flowers_valid': '/local/rcs/ll3504/datasets/flowers/valid',
	'flowers_test': '/local/rcs/ll3504/datasets/flowers/test',
	'animal_faces': '/local/rcs/ll3504/datasets/animal_faces',
	'animal_faces_10': '/local/rcs/ll3504/datasets/animal_faces_10_classes',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_flowers': 'pretrained_models/stylegan2-flowers.pt',
	'stylegan_animalfaces': 'pretrained_models/stylegan2-animalfaces.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
