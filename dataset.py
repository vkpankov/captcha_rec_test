from pathlib import Path
from torchvision import datasets
import utils
import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        captcha_str = Path(img_loc).stem
        encoded_label = utils.onehot_encode(captcha_str)
        tuple_with_path = (tensor_image, encoded_label)
        return tuple_with_path
