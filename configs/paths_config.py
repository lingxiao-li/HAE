dataset_paths = {
	'celeba_train': '',
	'celeba_test': '/local/rcs/ll3504/datasets/CelebAMask-HQ/CelebA-HQ-img',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '/local/rcs/ll3504/datasets/FFHQ/resized',
	'flowers_train': '/proj/rcs-ssd/ll3504/datasets/flowers/train',
	'flowers_train_eva': '/proj/rcs-ssd/ll3504/datasets/flowers_eva/train',
	'flowers_valid': '/proj/rcs-ssd/ll3504/datasets/flowers/valid',
	'flowers_test': '/proj/rcs-ssd/ll3504/datasets/flowers/test',
	'flowers_test_eva': '/proj/rcs-ssd/ll3504/datasets/flowers_eva/test',
	'animal_faces': '/proj/rcs-ssd/ll3504/datasets/animal_faces',
	'animal_faces_train_eva': '/proj/rcs-ssd/ll3504/datasets/animal_faces_eva/train',
	'animal_faces_test_eva': '/proj/rcs-ssd/ll3504/datasets/animal_faces_eva/test',
	'animal_faces_10': '/proj/rcs-ssd/ll3504/datasets/animal_faces_10_classes',
	'vgg_faces_train': '/proj/rcs-ssd/ll3504/datasets/vggfaces/train',
	'vgg_faces_test': '/proj/rcs-ssd/ll3504/datasets/vggfaces/test',
}

model_paths = {
	'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_flowers': 'pretrained_models/stylegan2-flowers.pt',
	'stylegan_animalfaces': 'pretrained_models/stylegan2-animalfaces.pt',
	'stylegan_vggfaces': 'pretrained_models/psp_vggfaces.pt',
	'ir_se50': 'pretrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
