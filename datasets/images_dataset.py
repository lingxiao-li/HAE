from torch.utils.data import Dataset
from PIL import Image
import random
from utils import data_utils


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None, train_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.classes, self.class_to_idx = data_utils.find_classes(source_root)
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.train_transform = train_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		class_label = from_path.split('/')[-2]
		label = self.class_to_idx[class_label]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')
		while True:
			a = random.choice(self.source_paths)
			if class_label in a:
				postive_path = a
				break
		crop_im = Image.open(postive_path).convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)
			crop_im = self.target_transform(crop_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
			crop_im = self.source_transform(crop_im)
		else:
			from_im = to_im
			#crop_im = from_im

		return from_im, to_im, label
