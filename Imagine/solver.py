import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import os
from tqdm import tqdm
from models import Generator, Discriminator, weights_init_normal
import cv2 as cv 


SAVE_EPOCH = 10
PRINT_ITER = 20

class Solver:

    def __init__(self, config, loaders):

        # Parameters
        self.config = config
        self.loaders = loaders
        self.save_images_path = os.path.join(self.config.output_path, 'images/')
        self.save_models_path = os.path.join(self.config.output_path, 'models/')
        if self.config.mismatch:
            self.save_images_path_test = os.path.join(self.config.output_path, 'result_mismatch/')
            # self.save_images_path_test = os.path.join(self.config.output_path, 'images_test_mismatch/')
        else:
            self.save_images_path_test = os.path.join(self.config.output_path, 'images_test/')


        # Set Devices
        if self.config.cuda is not '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Initialize
        self._init_models()
        self._init_losses()
        self._init_optimizers()

        # Resume Model
        if self.config.resume_epoch == -1:
            self.start_epoch = 0
        elif self.config.resume_epoch >=0:
            self.start_epoch = self.config.resume_epoch
            self._restore_model(self.config.resume_epoch)

        if self.config.mode == "test":
            self._restore_model(self.config.test_epoch)


    def _init_models(self):

        # Init Model
        self.generator = Generator()
        self.discriminator = Discriminator(self.config.conv_dim, self.config.layer_num)
        # Init Weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        # Move model to device (GPU or CPU)
        self.generator = torch.nn.DataParallel(self.generator).to(self.device)
        self.discriminator = torch.nn.DataParallel(self.discriminator).to(self.device)


    def _init_losses(self):
        # Init GAN loss and Reconstruction Loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_recon = torch.nn.L1Loss()


    def _init_optimizers(self):
        # Init Optimizer. Use Hyper-Parameters as DCGAN
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=2e-4, betas=[0.5, 0.999])
        # self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=[0.5, 0.999])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=[0.5, 0.999])
        # Set learning-rate decay
        self.g_lr_decay = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=100, gamma=0.1)
        self.d_lr_decay = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=100, gamma=0.1)


    def _lr_decay_step(self, current_epoch):
        self.g_lr_decay.step(current_epoch)
        self.d_lr_decay.step(current_epoch)


    def _save_model(self, current_epoch):
        # Save generator and discriminator
        torch.save(self.generator.state_dict(), os.path.join(self.save_models_path, 'G_{}.pkl'.format(current_epoch)))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_models_path, 'D_{}.pkl'.format(current_epoch)))
        print('Note: Successfully save model as {}'.format(current_epoch))


    def _restore_model(self, resume_epoch):
        # Resume generator and discriminator
        print("Strict?", self.config.strict_load)
        strict_load = self.config.strict_load
        # strict_load = False
        # print(self.generator.state_dict().keys())
        self.discriminator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'D_{}.pkl'.format(resume_epoch))), strict = strict_load)
        self.generator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'G_{}.pkl'.format(resume_epoch))), strict= strict_load)
        
        print('Note: Successfully resume model from {}'.format(resume_epoch))


    def train(self):

        # Load 16 images as fixed image for displaying and debugging
        fixed_sample_shape, fixed_sample_texture, fixed_sample_color, fixed_target_images, _, _, _ = next(iter(self.loaders.test_loader))

        ones = torch.ones_like(self.discriminator(fixed_target_images, fixed_sample_shape, fixed_sample_texture, fixed_sample_color))
        zeros = torch.zeros_like(self.discriminator(fixed_target_images, fixed_sample_shape, fixed_sample_texture, fixed_sample_color))
        
        for ii in range(int(16/self.config.batch_size-1)):
            fixed_sample_shape_, fixed_sample_texture_, fixed_sample_color_, fixed_target_images_ , _, _, _ = next(iter(self.loaders.test_loader))
            fixed_sample_shape = torch.cat([fixed_sample_shape, fixed_sample_shape_], dim=0)
            fixed_sample_texture = torch.cat([fixed_sample_texture, fixed_sample_texture_], dim=0)
            fixed_sample_color = torch.cat([fixed_sample_color, fixed_sample_color_], dim=0)
            fixed_target_images = torch.cat([fixed_target_images, fixed_target_images_], dim=0)

        fixed_sample_shape = fixed_sample_shape.to(self.device)
        fixed_sample_texture = fixed_sample_texture.to(self.device)
        fixed_sample_color = fixed_sample_color.to(self.device) 
        fixed_target_images = fixed_target_images.to(self.device)

        # Train 200 epoches
        for epoch in range(self.start_epoch, 500):
            # Save Images for debugging
            with torch.no_grad():
                self.generator = self.generator.eval()
                fake_images = self.generator(fixed_sample_shape, fixed_sample_texture, fixed_sample_color) # 
                all = torch.cat([torch.cat([fixed_sample_texture] * 3, dim = 1), 
                                torch.cat([fixed_sample_shape] * 3, dim = 1), 
                                fixed_sample_color, 
                                fake_images, 
                                fixed_target_images], dim=0)
                
                save_image((all.cpu()+1.0)/2.0,
                           os.path.join(self.save_images_path, 'images_{}.jpg'.format(epoch)), 16)

            # Train
            self.generator = self.generator.train()
            
            for iteration, data in enumerate(self.loaders.train_loader):
                #########################################################################################################
                #                                            load a batch data                                          #
                #########################################################################################################
                sample_shape, sample_texture, sample_color, target_images, target_shape, target_texture, target_color = data
                sample_shape = sample_shape.to(self.device)
                sample_texture = sample_texture.to(self.device)
                sample_color = sample_color.to(self.device)
                target_images = target_images.to(self.device)

                target_shape = target_shape.to(self.device)
                target_texture = target_texture.to(self.device)
                target_color = target_color.to(self.device)

                #########################################################################################################
                #                                                     Generator                                         #
                #########################################################################################################
                fake_images = self.generator(sample_shape, sample_texture, sample_color)

                # old: 
                gan_loss = self.criterion_GAN(self.discriminator(fake_images, sample_shape, sample_texture, sample_color), ones)
                recon_loss = self.criterion_recon(fake_images, target_images)
                g_loss = gan_loss + 10 * recon_loss

                # # new
                # gan_loss = self.criterion_GAN(self.discriminator(fake_images, sample_shape, sample_texture, sample_color), ones)
                # recon_loss = -1
                # g_loss = gan_loss

                self.g_optimizer.zero_grad()
                g_loss.backward(retain_graph=True)
                self.g_optimizer.step()

                #########################################################################################################
                #                                                     Discriminator                                     #
                #########################################################################################################
                if iteration % self.config.G_iter == 0:
                    loss_real = self.criterion_GAN(self.discriminator(target_images, target_shape, target_texture, target_color), ones)
                    loss_fake = self.criterion_GAN(self.discriminator(fake_images.detach(), sample_shape, sample_texture, sample_color), zeros)
                    d_loss = (loss_real + loss_fake) / 2.0

                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()
                if (iteration+1) % PRINT_ITER == 0:
                    print('[EPOCH:{}/{}]  [ITER:{}/{}]  [D_GAN:{}]  [G_GAN:{}]  [RECON:{}] [LR:{}]'.
                          format(epoch, 500, iteration, len(self.loaders.train_loader), d_loss, gan_loss, recon_loss, self.g_optimizer.param_groups[0]['lr']))
            self._lr_decay_step(epoch - self.start_epoch)
            # Save model
            if (epoch+1) % SAVE_EPOCH == 0:
                self._save_model(epoch)


    def test(self):
        with torch.no_grad():
            self.generator = self.generator.eval()
            # Load 16 images as fixed image for displaying and debugging
            fixed_sample_shape, fixed_sample_texture, fixed_sample_color, fixed_target_images, _, _, _  = next(iter(self.loaders.test_loader))

            # ones = torch.ones_like(self.discriminator(fixed_target_images, fixed_sample_shape, fixed_sample_texture, fixed_sample_color))
            # zeros = torch.zeros_like(self.discriminator(fixed_target_images, fixed_sample_shape, fixed_sample_texture, fixed_sample_color))

            if self.config.mismatch:
                test_loader = self.loaders.mismatch_test_loader
            else:
                test_loader = self.loaders.test_loader

            for iteration, data in tqdm(enumerate(test_loader)):
                #########################################################################################################
                #                                            load a batch data                                          #
                #########################################################################################################
                sample_shape, sample_texture, sample_color, target_images,  _, _, _  = data
                sample_shape = sample_shape.to(self.device)
                sample_texture = sample_texture.to(self.device)
                sample_color = sample_color.to(self.device)
                target_images = target_images.to(self.device)

                #########################################################################################################
                #                                                     Generator                                         #
                #########################################################################################################
                fake_images = self.generator(sample_shape, sample_texture, sample_color)


                all = torch.cat([sample_color, 
                    torch.cat([sample_shape] * 3, dim = 1), 
                    torch.cat([sample_texture]  * 3, dim = 1),
                    
                    fake_images, 
                    target_images], dim=0)
                
                save_image((all.cpu()+1.0)/2.0,
                           os.path.join(self.save_images_path_test, 'images_{}.jpg'.format(iteration)), self.config.batch_size)

    def predict(self):
            with torch.no_grad():
                self.generator = self.generator.eval()
                # Load 16 images as fixed image for displaying and debugging
                fixed_sample_shape, fixed_sample_texture, fixed_sample_color, fixed_target_images, _, _, _  = next(iter(self.loaders.test_loader))

                # ones = torch.ones_like(self.discriminator(fixed_target_images, fixed_sample_shape, fixed_sample_texture, fixed_sample_color))
                # zeros = torch.zeros_like(self.discriminator(fixed_target_images, fixed_sample_shape, fixed_sample_texture, fixed_sample_color))

                if self.config.mismatch:
                    test_loader = self.loaders.mismatch_test_loader
                else:
                    test_loader = self.loaders.test_loader

                for iteration, data in tqdm(enumerate(test_loader)):
                    #########################################################################################################
                    #                                            load a batch data                                          #
                    #########################################################################################################
                    sample_shape, sample_texture, sample_color, target_images,  _, _, _  = data
                    sample_shape = sample_shape.to(self.device)
                    sample_texture = sample_texture.to(self.device)
                    sample_color = sample_color.to(self.device)
                    target_images = target_images.to(self.device)

                    #########################################################################################################
                    #                                                     Generator                                         #
                    #########################################################################################################
                    fake_images = self.generator(sample_shape, sample_texture, sample_color).cpu().numpy().transpose((0, 2, 3, 1))
                    
                    fake_images = (fake_images+1.0)/2.0*255
                    


                    for i in range(self.config.batch_size):
                        cv.imwrite(os.path.join(self.save_images_path_test, 'images_{}.jpg'.format(iteration*self.config.batch_size+i)), 
                                   fake_images[i, :, :, ::-1])
                            