import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class ISTD(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers,num_env,  num_class,image_size, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(ISTD, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        self.netG.set_condition(image_size,image_size, num_env,num_class)

        ##tar
        self.loss_fn_tar = losses[0]
        self.netG_tar = networks[0]
        self.netG_tar.set_condition(image_size, image_size, num_env, num_class)

        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            ##tar
            self.netG_EMA_tar = copy.deepcopy(self.netG_tar)

            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        ##tar
        self.netG_tar = self.set_device(self.netG_tar, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
            ##tar
            self.netG_EMA_tar = self.set_device(self.netG_EMA_tar, distributed=self.opt['distributed'])
        self.load_networks()


        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        ##tar
        self.optG_tar = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG_tar.parameters())), **optimizers[0])
        #self.optG = torch.optim.Adam(list(lambda p: p.requires_grad, self.netG.parameters()), **optimizers[0])

        self.optimizers.append(self.optG)
        self.optimizers.append(self.optG_tar)
        self.resume_training()


        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
            ##tar
            self.netG_tar.module.set_loss(self.loss_fn_tar)
            self.netG_tar.module.set_new_noise_schedule(phase=self.phase)

        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)
            ##tar
            self.netG_tar.set_loss(self.loss_fn_tar)
            self.netG_tar.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.tar_cond_image = self.set_device(data.get('tar_cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.tar_mask = self.set_device(data.get('tar_mask'))
        self.mask_image = data.get('mask_image')
        self.tar_mask_image = data.get('tar_mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
        self.image_env = data.get('image_env')
        self.image_class = data.get('image_class')


    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
            ##tar
            'tar_cond_image': (self.tar_cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
                ##tar
                'tar_mask': self.tar_mask.detach()[:].float().cpu(),
                'tar_mask_image': (self.tar_mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2,
                ##tar
                'output_tar': (self.output_tar.detach()[:].float().cpu()+1)/2,

            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_scene_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(torch.add(self.visuals_t[idx::self.batch_size],self.visuals_tar_t[idx::self.batch_size]).detach().float().cpu())


            ret_path.append('Process_tar_{}'.format(self.path[idx]))
            ret_result.append(self.visuals_tar[idx::self.batch_size].detach().float().cpu())

            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(torch.add(self.visuals_t[idx-self.batch_size],self.visuals_tar_t[idx-self.batch_size]).detach().float().cpu())

            ret_path.append('Out_scene_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx - self.batch_size].detach().float().cpu())

            ret_path.append('Out_scene_t_{}'.format(self.path[idx]))
            ret_result.append(self.visuals_t[idx - self.batch_size].detach().float().cpu())

            ret_path.append('Out_tar_{}'.format(self.path[idx]))
            ret_result.append(self.visuals_tar[idx-self.batch_size].detach().float().cpu())

            ret_path.append('Out_tar_t_{}'.format(self.path[idx]))
            ret_result.append((self.mask_tar_t[idx - self.batch_size].int()*self.visuals_tar[idx-self.batch_size]).detach().float().cpu())

        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

            ##tar
            ret_path.extend(['Mask_tar_{}'.format(name) for name in self.path])
            ret_result.extend(self.tar_mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        ##tar
        self.netG_tar.train()

        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):

            self.set_input(train_data)
            self.optG.zero_grad()
            ##tar
            self.optG_tar.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask, image_env = self.image_env,image_class = self.image_class)
            loss.backward()
            ##tar
            loss_tar = self.netG_tar(self.gt_image, self.tar_cond_image, mask=self.tar_mask, image_env = self.image_env,image_class = self.image_class)
            loss_tar.backward()

            self.optG.step()
            ##tar
            self.optG_tar.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            #tar
            self.train_metrics.update(self.loss_fn_tar.__name__, loss_tar.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
                    ##tar
                    self.EMA.update_model_average(self.netG_EMA_tar, self.netG_tar)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        ##tar
        self.netG_tar.eval()

        self.val_metrics.reset()
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals, self.visuals_t, self.mask_t = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, image_env = self.image_env,image_class = self.image_class)
                        ##tar
                        self.output_tar, self.visuals_tar,self.visuals_tar_t,self.mask_tar_t = self.netG_tar.module.restoration(self.tar_cond_image, y_t=self.tar_cond_image,
                            y_0=self.gt_image, mask=self.tar_mask, sample_num=self.sample_num, image_env = self.image_env,image_class = self.image_class)
                    else:
                        self.output, self.visuals, self.visuals_t, self.mask_t = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                        ##tar
                        self.output_tar, self.visuals_tar,self.visuals_tar_t,self.mask_tar_t = self.netG_tar.module.restoration(self.tar_cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals, self.visuals_t, self.mask_t = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, image_env = self.image_env,image_class = self.image_class)
                        ##tar
                        self.output_tar, self.visuals_tar, self.visuals_tar_t,self.mask_tar_t = self.netG_tar.restoration(self.tar_cond_image, y_t=self.tar_cond_image,
                            y_0=self.gt_image, mask=self.tar_mask, sample_num=self.sample_num, image_env = self.image_env,image_class = self.image_class)
                    else:
                        self.output, self.visuals, self.visuals_t,self.mask_t = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        ##tar
                        self.output_tar, self.visuals_tar, self.visuals_tar_t,self.mask_tar_t = self.netG_tar.restoration(self.tar_cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    ##tar
                    value_tar = met(self.gt_image, self.output_tar)

                    self.val_metrics.update(key, value)
                    ##tar
                    self.val_metrics.update(key, value_tar)

                    self.writer.add_scalar(key, value)
                    ##tar
                    self.writer.add_scalar(key, value_tar)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                ##tar
                for key, value_tar in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value_tar)

                self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        ##tar
        self.netG_tar.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals, self.visuals_t,self.mask_t = self.netG.module.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num,image_env = self.image_env,image_class = self.image_class)
                        ##tar
                        self.output_tar, self.visuals_tar, self.visuals_tar_t,self.mask_tar_t = self.netG_tar.module.restoration(self.tar_cond_image, y_t=self.tar_cond_image,
                            y_0=self.gt_image, mask=self.tar_mask, sample_num=self.sample_num,image_env = self.image_env,image_class = self.image_class)
                    else:
                        self.output, self.visuals, self.visuals_t,self.mask_t = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                        ##tar
                        self.output_tar, self.visuals_tar, self.visuals_tar_t,self.mask_tar_t = self.netG_tar.module.restoration(self.tar_cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals, self.visuals_t,self.mask_t  = self.netG.restoration(self.cond_image, y_t=self.cond_image,
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, image_env = self.image_env,image_class = self.image_class)
                        ##tar
                        self.output_tar, self.visuals_tar, self.visuals_tar_t, self.mask_tar_t = self.netG_tar.restoration(self.tar_cond_image, y_t=self.tar_cond_image,
                            y_0=self.gt_image, mask=self.tar_mask, sample_num=self.sample_num, image_env = self.image_env,image_class = self.image_class)
                    else:
                        self.output, self.visuals, self.visuals_t,self.mask_t = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        ##tar
                        self.output_tar, self.visuals_tar, self.visuals_tar_t,self.mask_tar_t = self.netG_tar.restoration(self.tar_cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    ##tar
                    value_tar = met(self.gt_image, self.output_tar)
                    self.test_metrics.update(key, value)
                    ##tar
                    self.test_metrics.update(key, value_tar)

                    self.writer.add_scalar(key, value)
                    ##tar
                    self.writer.add_scalar(key, value_tar)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                ##tar
                for key, value_tar in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value_tar)

                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
            ##tar
            netG_tar_label = self.netG_tar.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
            ##tar
            netG_tar_label = self.netG_tar.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        ##tar
        self.load_network(network=self.netG_tar, network_label=netG_tar_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)
            ##tar
            self.load_network(network=self.netG_EMA_tar, network_label=netG_tar_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
            ##tar
            netG_tar_label = self.netG_tar.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
            ##tar
            netG_tar_label = self.netG_tar.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        ##tar
        self.save_network(network=self.netG_tar, network_label=netG_tar_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
            ##tar
            self.save_network(network=self.netG_EMA_tar, network_label=netG_tar_label+'_ema')
        self.save_training_state()
