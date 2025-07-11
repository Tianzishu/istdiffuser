import torch.utils.data as data
from PIL.Image import Resampling
from torchvision import transforms
from PIL import Image
import torch


import  os

from PIL import Image
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str_, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

class ISTNudtDBDataset(data.Dataset):
    CLASSES = ('airplane','point','ship','spot','uav') #nudt
    ENVS = ('air_field','city','cloud','field','highlight','sea') #nudt

    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader,output_anns=None, out_anns_path=None):
        imgs = make_dataset(data_root)
        self.CLASS_image_list = {cat: make_dataset(os.path.join(data_root.split('flist')[0], 'flist'+'/'+'classes',cat+'.txt')) for cat in self.CLASSES}
        self.ENV_image_list = {env: make_dataset(os.path.join(data_root.split('flist')[0], 'flist'+'/'+'envs',env+'.txt')) for env in self.ENVS}

        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])

        self.tfs0 = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor()
        ])

        self.invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
                                       transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                                            std=[1., 1., 1.]),
                                       ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size
        self.env2label = {env: i for i, env in enumerate(self.ENVS)}
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.output_anns = output_anns
        self.out_anns_path = out_anns_path
        if self.output_anns:
            if not os.path.exists(self.out_anns_path ): 
                os.makedirs(self.out_anns_path)


    def __getitem__(self, index):
        ret = {}
        ann = {}
        path = self.imgs[index]

        img = self.loader(path)
        img_size = np.array(img.size) #(w,h)
        img = self.tfs(img)
        img_tf_size = img.shape[1:] #(h,w)
        w_scale = img_tf_size[1]/img_size[0]
        h_scale = img_tf_size[0]/img_size[1]
        ann['height'] = img_tf_size[0]
        ann['weight'] = img_tf_size[1]
        ann['depth'] = 1 # for ist images
        ann['filename'] ='Out_'+path.split('/')[-1]

        if self.mask_mode == 'seg':
            #scale_factor = np.array([w_scale, h_scale],dtype=np.float32)
            ann_path = path.replace('images','masks')
            mask = self.get_mask_from_seg_pic(ann_path,self.image_size)
            if self.output_anns:
                self.gen_mask(self.out_anns_path,path.split('/')[-1],(1. - mask))

        img = self.invTrans(img)

        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        tar_cond_image = img*mask + (1. - mask)*torch.randn_like(img)
        tar_mask_img = img*mask + (1. - mask)
        tar_mask = (1. - mask)

        #ENVS:

        for env in self.ENV_image_list.keys():
            if path.split('/')[-1] in self.ENV_image_list[env]:
                image_env = self.env2label[env]
                break

        # CLASSES:
        
        for cat in self.CLASS_image_list.keys():
            if path.split('/')[-1] in self.CLASS_image_list[cat]:
                image_class = self.cat2label[cat]
                break

        ret['gt_image'] = img[0].unsqueeze(0)
        ret['cond_image'] = cond_image[0].unsqueeze(0)
        ret['mask_image'] = mask_img[0].unsqueeze(0)
        ret['mask'] = mask[0].unsqueeze(0)
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        ret['image_env'] = image_env
        ret['image_class'] = image_class

        ret['tar_cond_image'] = tar_cond_image[0].unsqueeze(0)
        ret['tar_mask_image'] = tar_mask_img[0].unsqueeze(0)
        ret['tar_mask'] = tar_mask[0].unsqueeze(0)

        return ret

    def __len__(self):
        return len(self.imgs)

    def gen_mask(self,path, name, mask_img):
        mask_numpy = mask_img[0].mul(255).byte().numpy()
        # 创建图像对象
        mask_image = Image.fromarray(mask_numpy, mode='L')  # (mask1_numpy, mode='L')灰度图像
        # 保存为图像文件
        mask_image.save(path + '/' + 'mask_' + name)

        #print("mask生成保存成功！")

    def save_image(self, name, cond_image, mask_img, mask, tar_cond_image, tar_mask_img, tar_mask, save_dir = 'noiseimage1/'):
        os.makedirs(save_dir, exist_ok=True)

        # Convert tensors to numpy arrays and then to byte format
        cond_numpy = cond_image[0].mul(255).byte().numpy() #生成黑白掩码
        #cond_numpy = cond_image.mul(255).byte().cpu().numpy().transpose(1, 2, 0) #生成彩色掩码
        mask_img_numpy = mask_img[0].mul(255).byte().numpy()
        mask_numpy = mask[0].mul(255).byte().numpy()

        tar_cond_numpy = tar_cond_image[0].mul(255).byte().numpy()
        tar_mask_img_numpy = tar_mask_img[0].mul(255).byte().numpy()
        tar_mask_numpy = tar_mask[0].mul(255).byte().numpy()

        # Create Image objects from numpy arrays
        cond_image = Image.fromarray(cond_numpy)
        mask_img_image = Image.fromarray(mask_img_numpy)
        mask_image = Image.fromarray(mask_numpy)

        tar_cond_image = Image.fromarray(tar_cond_numpy)
        tar_mask_img_image = Image.fromarray(tar_mask_img_numpy)
        tar_mask_image = Image.fromarray(tar_mask_numpy)

        # Save images in the specified directory
        cond_image.save(os.path.join(save_dir, 'cond_' + name))
        mask_img_image.save(os.path.join(save_dir, 'mask_img_' + name))
        mask_image.save(os.path.join(save_dir, 'mask_' + name))

        tar_cond_image.save(os.path.join(save_dir, 'tar_cond_' + name))
        tar_mask_img_image.save(os.path.join(save_dir, 'tar_mask_img_' + name))
        tar_mask_image.save(os.path.join(save_dir, 'tar_mask_' + name))



    def get_mask_from_seg_pic(self,pic_path, image_size):
        mask_image = Image.open(pic_path).convert("L")

        mask = mask_image.resize(image_size, Resampling.NEAREST)
        mask = np.array(mask, dtype=np.float32)

        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return  (1. - torch.from_numpy(mask))


    def resize_bboxes(self, bboxes, scale_factor):
        """Resize bounding boxes with ``results['scale_factor']``."""
        bboxes = bboxes * scale_factor
        return np.fix(bboxes).astype(np.int) 

