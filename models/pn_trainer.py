from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os
import datetime
import re
import matplotlib.pyplot as plt
import time
from transforms3d.quaternions import *
from transforms3d.euler import *
from transforms3d.axangles import *
import scipy
from ycb_render.ycb_renderer import *
from decimal import *
import cv2
from shutil import copyfile

from models.pointnet2_msg import *
from config.config import cfg
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d, Axes3D
import gc

from utils.deepsdf_utils import *

class PointNet_Trainer(nn.Module):
    def __init__(self, cfg_path, model_category, config_new=None,
                 aae_capacity=1, aae_code_dim=128, ckpt_path=None):
        super(PointNet_Trainer, self).__init__()

        self.cfg_path = cfg_path

        if config_new != None:
            self.cfg_all = config_new
        else:
            self.cfg_all = cfg

        self.category = model_category

        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        self.ckpt_dir = './checkpoints'

        self.model = get_model(input_channels=0)

        self.use_GPU = (torch.cuda.device_count() > 0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)

        self.mseloss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.l1_recon_loss = nn.L1Loss(reduction='mean')

        if self.use_GPU:
            self.mseloss = self.mseloss.cuda()
            self.l1_loss = self.l1_loss.cuda()
            self.l1_recon_loss = self.l1_recon_loss.cuda()

        self.loss_history_recon = []
        self.val_loss_history_recon = []

        self.batch_size_train = self.cfg_all.TRAIN.BATCH_SIZE
        self.batch_size_val = self.cfg_all.TRAIN.VAL_BATCH_SIZE
        self.start_epoch = 1

        # self.codebook_dir = None
        self.log_dir = None
        self.checkpoint_path = None

        experiment_directory = './checkpoints/deepsdf_ckpts/{}s/'.format(self.category)
        print(experiment_directory)
        self.decoder = load_decoder(experiment_directory, 2000)
        self.decoder = self.decoder.module.cuda()
        self.evaluator = Evaluator(self.decoder)

        if ckpt_path is not None:
            self.load_ckpt(ckpt_path=ckpt_path)
        
    def set_log_dir(self, model_path=None, now=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        if now == None:
            now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/trans\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)), int(m.group(6)))

        # Directory for training logs
        self.log_dir = os.path.join(self.ckpt_dir, "{}{:%Y%m%dT%H%M%S}_{}".format(
            self.category, now, self.cfg_all.EXP_NAME))
        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "ckpt_{}_*epoch*.pth".format(
            self.category))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def save_ckpt(self, epoch):
        print('=> Saving checkpoint to {} ...'.format(self.checkpoint_path.format(epoch)))
        torch.save({
            'epoch': epoch,
            'log_dir': self.log_dir,
            'checkpoint_path': self.checkpoint_path,
            'aae_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss_history_recon': self.loss_history_recon,
            'val_loss_history_recon': self.val_loss_history_recon,
        }, self.checkpoint_path.format(epoch))
        print('=> Finished saving checkpoint to {} ! '.format(self.checkpoint_path.format(epoch)))

    def load_ckpt(self, ckpt_path):
        if os.path.isfile(ckpt_path):
            print("=> Loading checkpoint from {} ...".format(ckpt_path))
            checkpoint = torch.load(ckpt_path)
            self.start_epoch = checkpoint['epoch'] + 1
            self.log_dir = checkpoint['log_dir']
            self.checkpoint_path = checkpoint['checkpoint_path']
            self.model.load_ckpt_weights(checkpoint['aae_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            self.loss_history_recon = checkpoint['loss_history_recon']
            self.val_loss_history_recon = checkpoint['val_loss_history_recon']
            print("=> Finished loading checkpoint from {} (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']))
        else:
            print('=> Cannot find checkpoint file in {} !'.format(ckpt_path))

    def plot_loss(self, loss, val_loss, title, save=True, log_dir=None):
        loss = np.array(loss)
        plt.figure(title)
        plt.gcf().clear()
        plt.plot(loss, label='train')
        plt.plot(val_loss, label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        if save:
            save_path = os.path.join(log_dir, "{}.png".format(title))
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)

    def train_model(self, train_dataset, val_dataset, epochs, save_frequency=20):
        self.model.cuda()

        train_set = train_dataset
        print('train set size {}'.format(len(train_set)))

        if self.log_dir == None:
            self.set_log_dir()
            if not os.path.exists(self.log_dir) and save_frequency > 0:
                print('Create folder at {}'.format(self.log_dir))
                os.makedirs(self.log_dir)
                copyfile(self.cfg_path, self.log_dir + '/config.yml')

        train_generator = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size_train,
                                                      shuffle=True, num_workers=0)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size_train,
                                                      shuffle=True, num_workers=0)

        train_steps = np.floor(len(train_set)/self.batch_size_train)
        val_steps = 30

        if self.cfg_all.TRAIN.ONLINE_RENDERING:
            train_steps = np.floor(train_steps/4)
        
        print("train steps = ", train_steps)
        save_frequency = 30
        for epoch in np.arange(start=self.start_epoch, stop=(self.start_epoch+epochs)):
            print("Epoch {}/{}.".format(epoch, (self.start_epoch+epochs)-1))

            recon_loss_train = self.train_epoch(train_generator, self.optimizer,
                                                                  train_steps, epoch, self.start_epoch+epochs-1)

            recon_loss_val = self.val_epoch(val_generator, self.optimizer,
                                                              val_steps, epoch, self.start_epoch + epochs - 1)

            self.loss_history_recon.append(recon_loss_train)
            self.val_loss_history_recon.append(recon_loss_val)

            self.plot_loss(self.loss_history_recon, self.val_loss_history_recon, 'recon loss', save=True, log_dir=self.log_dir)
            if save_frequency > 0 and epoch % save_frequency == 0:
                self.save_ckpt(epoch)
                
    def train_epoch(self, datagenerator, optimizer, steps, epoch, total_epoch):
        self.model.train()
        loss_sum_recon = 0
        
        step = 0
        optimizer.zero_grad()
        for inputs in datagenerator:

            # receiving data from the renderer
            depths, depths_target, label, points_c, affine, shift, scale, mask = inputs
            if self.use_GPU:
                depths = depths.cuda()
                depths_target = depths_target.cuda()
                label = label.cuda()
                points_c = points_c.cuda()
            
            # forward pass
            pred_label = self.model(points_c)
            loss_label = self.mseloss(pred_label, label)
            loss = loss_label
            loss_pn_data = loss_label.data.cpu().item()
            # backward pass
            optimizer.zero_grad()
            try:
                loss.backward()
            except:
                pass

            optimizer.step()

            printProgressBar(step + 1, steps, prefix="\t{}/{}: {}/{}".format(epoch, total_epoch, step + 1, steps),
                             suffix="Complete [Training] - loss_recon: {:.4f}". \
                             format(loss_pn_data), length=10)

            # save results
            # plot_n_comparison = 5
            # if step < plot_n_comparison:
            #     self.plot_latent_comparison(label[0].detach(), pred_label[0].detach(), self.evaluator, str(step)+'_train')

            loss_sum_recon += loss_pn_data / steps

            if step==steps-1:
                break
            step += 1

        return loss_sum_recon

    def val_epoch(self, datagenerator, optimizer, steps, epoch, total_epoch, dstrgenerator=None):
        self.model.eval()
        loss_func = BootstrapedMSEloss(self.cfg_all.TRAIN.BOOTSTRAP_CONST)
        loss_sum_recon = 0
        step = 0
        optimizer.zero_grad()

        for inputs in datagenerator:
            # receiving data from the renderer
            depths, depths_target, label, points_c, affine, shift, scale, mask = inputs
            if self.use_GPU:
                depths = depths.cuda()
                depths_target = depths_target.cuda()
                label = label.cuda()
                points_c = points_c.cuda()

            # forward pass
            pred_label = self.model(points_c)
            pred = pred_label.detach()
            loss_label = self.mseloss(pred, label)
            loss = loss_label
            loss_pn_data = loss_label.data.cpu().item()

            printProgressBar(step + 1, steps, prefix="\t{}/{}: {}/{}".format(epoch, total_epoch, step + 1, steps),
                             suffix="Complete [Validation] - loss_recon: {:.4f}". \
                             format(loss_pn_data), length=10)

            loss_sum_recon += loss_pn_data / steps

            # # display
            # plot_n_comparison = 20
            # if step < plot_n_comparison:
            #     depth = depths[0, 0].detach().cpu().numpy()
            #     depth_target = depths_target[0, 0].cpu().numpy()
            #     depth_recon = depth_recnst[0, 0].detach().cpu().numpy()
            #     disp = (depth, depth_target, depth_recon)
            #     self.plot_comparison(disp, str(step)+'_val')

            if step==steps-1:
                break
            step += 1
        return loss_sum_recon

    # visualization
    def plot_comparison(self, images, name, save=True):
        if len(images) == 3:
            comparison_row = np.concatenate(images, 1)
            plt.figure("compare"+name)
            plt.gcf().clear()
            plt.imshow(comparison_row)
        else:
            comparison_row = np.concatenate(images[:3], 1)
            comparison_row2 = np.concatenate(images[3:], 1)
            plt.figure("compare" + name)
            plt.gcf().clear()
            plt.subplot(2, 1, 1)
            plt.imshow(comparison_row)
            plt.subplot(2, 1, 2)
            plt.imshow(comparison_row2)

        if save:
            save_path = os.path.join(self.log_dir, "compare_{}.png".format(name))
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)

    def plot_latent_comparison(self, label_gt, label_pred, evaluator, name):
        fname = os.path.join('./vis/', 'mesh.ply')
        points_from_label_gt = evaluator.latent_vec_to_points(label_gt, num_points=10000, silent=True)
        points_from_label_pred = evaluator.latent_vec_to_points(label_pred, num_points=10000, silent=True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        display_interval = 20
        if points_from_label_pred is not None:
            ax.scatter(points_from_label_pred[::display_interval, 0], points_from_label_pred[::display_interval, 1],
                       points_from_label_pred[::display_interval, 2], color='blue')
        else:
            print('predicted points is none!')
        ax.scatter(points_from_label_gt[::display_interval, 0], points_from_label_gt[::display_interval, 1],
                   points_from_label_gt[::display_interval, 2], color='green')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        min_coor = -1
        max_coor = 1
        ax.set_xlim(min_coor, max_coor)
        ax.set_ylim(min_coor, max_coor)
        ax.set_zlim(min_coor, max_coor)
        ax.view_init(elev=0., azim=0)
        save_path = os.path.join(self.log_dir, "compare_{}.png".format(name))
        plt.savefig(save_path)
        plt.close()

    def plot_latent(self, label, evaluator, name):
        fname = os.path.join('./vis/', 'mesh.ply')
        points_from_label = evaluator.latent_vec_to_points(label, num_points=10000, silent=True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        display_interval = 20
        ax.scatter(points_from_label[::display_interval, 0],
                   points_from_label[::display_interval, 1],
                   points_from_label[::display_interval, 2], color='green')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        min_coor = -1
        max_coor = 1
        ax.set_xlim(min_coor, max_coor)
        ax.set_ylim(min_coor, max_coor)
        ax.set_zlim(min_coor, max_coor)
        save_path = os.path.join(self.log_dir, "compare_{}.png".format(name))
        plt.savefig(save_path)
        plt.close()