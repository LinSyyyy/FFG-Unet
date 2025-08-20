import os
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random

class RandomRotate90:
    def __call__(self, img):
        degrees = [0, 90, 180, 270]
        degree = random.choice(degrees)
        return transforms.functional.rotate(img, degree)

def mask_to_onehot(mask, palette):
    semantic_map = [ ]
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)   # 沿对应轴进行与运算0，1对应行列，-1为所有维度最后一个维度主要参考shape
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map

def make_dataset_from_list(flist):
    root = "D:/codefile/cangku"
    with open(flist) as f:
        lines = f.readlines()
        linelist = [line.split(' ') for line in lines]
        pictures = []
        masks = []
        for line in linelist:
            # line1 = "D:/codefile/cangku/"+line[0].rstrip('\n')
            # line2 = "D:/codefile/cangku/"+ line[2].rstrip('\n')
            line1 = os.path.join(root,line[0])
            line2 = os.path.join(root,line[1].rstrip('\n'))
            pictures.append(line1)
            masks.append(line2)

    return pictures,masks

mean, std = {}, {}
mean['mask'] = [0.485]
std['mask'] = [0.229]
mean['image'] = [0.485, 0.456, 0.406]
std['image'] = [0.229, 0.224, 0.225]

transform_mask = transforms.Compose([
        transforms.Normalize(mean['mask'], std['mask'])
        # transforms.Normalize(),
    ])
import torch
transform_image = transforms.Compose([
        transforms.Normalize(mean['image'], std['image'])
        # transforms.Normalize(),
    ])
import cv2
class ProstateDataset(Dataset):
    def __init__(self,one_hot_mask = False,transform = None,flist = None):
        self.one_hot_mask = one_hot_mask
        self.transform = transform
        self.pictures_path,self.mask_path = make_dataset_from_list(flist)


    def __getitem__(self, idex):

        mask_path = self.mask_path[idex]
        image_path = self.pictures_path[idex]
        # ii = Image.open(image_path)
        # image = Image.open(image_path).convert("RGB")
        # mask = Image.open(mask_path).convert("RGB")
        # image = Image.open(image_path).convert("RGB")
        # mask = Image.open(mask_path).convert('1')

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        H,W,_ = image.shape

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = np.expand_dims(mask, 0)
        #mask = mask.transpose(2, 0, 1)

        if mask.max()<1:
            mask[mask>0] = 1.0
        
        # image = transform_image(image)
        # mask = transform_mask(mask)

        return image, mask

    def __len__(self):
        return len(self.pictures_path)


if __name__=="__main__":
    sim = round(float(22/(64*7)),5)
    print(sim)
    pass
    # import random
    #
    # train_path = "D:/codefile/cangku/skin/skin_data/train.txt"
    # val_path = "D:/codefile/cangku/skin/skin_data/val.txt"
    # test_path = "D:/codefile/cangku/skin/skin_data/test.txt"
    # mean, std = {}, {}
    # mean['imagenet'] = [0.485, 0.456, 0.406]
    # std['imagenet'] = [0.229, 0.224, 0.225]
    # input_h=256
    # input_w=256
    #
    # rand_num = random.random()
    # if rand_num > 0.5:
    #     flip = transforms.RandomHorizontalFlip()
    # else:
    #     flip = transforms.RandomVerticalFlip()
    #
    # transform_train = transforms.Compose([
    #     transforms.Resize((input_h, input_w)),
    #     flip,
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean['imagenet'], std['imagenet'])
    # ])
    #
    # transform_val = transforms.Compose([
    #     transforms.Resize((input_h, input_w)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean['imagenet'], std['imagenet'])
    # ])
    #
    # train_data = ProstateDataset(flist=train_path, transforms_train=transform_train, transforms_val=None)
    # val_data = ProstateDataset(flist=val_path, transforms_train=None, transforms_val=transform_val)
    # tes_data = ProstateDataset(flist=val_path, transforms_train=None, transforms_val=transform_val)
    # trainloader = DataLoader(train_data, batch_size=4, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    # tes_loader = DataLoader(val_data, batch_size=1, shuffle=True)
    #
    # train_iter = iter(trainloader)
    # val_i = iter(val_loader)
    # tes_i = iter(tes_loader)
    # inputs_x, targets_x = tes_i.__next__()
    # m,n = val_i.__next__()
