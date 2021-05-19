from pathlib import Path
from random import randint, choice

import PIL
import os

from torch.utils.data import Dataset
from torchvision import transforms as T

## 21.05.08 PIL.Image.DecompressionBombError: Image size (311040000 pixels) exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.
PIL.Image.MAX_IMAGE_PIXELS = 1000000000 

class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)
        
        print(f"path : {path}")

        self.text_files = {}
        self.image_files = {}

        for root, dirs, files in os.walk(folder):
            if len(dirs) == 0:
                for file in files:
                    if file.endswith(".txt"):
                        key = file.split(".")[0]
                        self.text_files[key] = root+"/"+file

                    elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"):
                        key = file.split(".")[0]
                        self.image_files[key] = root+"/"+file

#         text_files = [*path.glob('**/*.txt')]
#         image_files = [
#             *path.glob('**/*.png'), *path.glob('**/*.jpg'),
#             *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
#         ]
        
#         print(f" text_files : {text_files[0]}, len(text_files) : {len(text_files)}")
#         print(f" image_files : {image_files[0]}, len(image_files) : {len(image_files)}")

#         text_files = {text_file.stem: text_file for text_file in text_files}
#         image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (self.image_files.keys() & self.text_files.keys())

        self.keys = list(keys)
        
#         self.text_files = {k: v for k, v in text_files.items() if k in keys}
#         self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
#             T.Lambda(lambda img: img.convert('RGB')
#             if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def img_convert(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
      
    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]
        
        ## 21.05.08 descriptions = text_file.read_text().split('\n') return f.read() OSError: [Errno 61] No data available
        try:
            text_file = self.text_files[key]
            image_file = self.image_files[key]

            if text_file is None or image_file is None:
                print(f"text_file : {text_file.shape}, image_file : {image_file.shape}")
                return self.skip_sample(ind)

    #             descriptions = text_file.read_text().split('\n')
            ## faster data processing
            with open(text_file) as f:
                descriptions = f.read().split('\n')

            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        except OSError as error: 
            print(f"[OSError] An exception occurred trying to load file {text_file}.")
            print(f"[OSError] Skipping index {ind}")
            return self.skip_sample(ind)
        
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"[IndexError] An exception occurred trying to load file {text_file}.")
            print(f"[IndexError] Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            
            array_img = PIL.Image.open(image_file)
            img = self.img_convert(array_img)
            image_tensor = self.image_transform(img)
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor
