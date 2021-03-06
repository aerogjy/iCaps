from models.aae_models import *
from utils.se3 import *
import numpy.ma as ma
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pdb
import glob
import copy
from config.config import cfg, cfg_from_file, get_output_dir, write_selected_class_file
import pprint
from transforms3d.axangles import *
from pose_rbpf.particle_filter import *
from pose_rbpf.render_wrapper import *
from datasets.render_shapenet_dataset import *
import matplotlib.patches as patches
from utils.deepsdf_utils import *
from models.pointnet2_msg import *
import pointnet2.pointnet2_utils as pointnet2_utils
from transforms3d import euler
from utils.compute_farthest_distance_numba import get_max_dist, get_bbox_dist
from collections import deque
from scipy import ndimage
import open3d as o3d
NPOINTS = 4873
NP_THRESHOLD = 0  
class PoseRBPF:
    def __init__(self, obj_list, cfg_list, ckpt_dir, test_instance,
                 model_dir='../category-level_models/',
                 test_model_dir='../obj_models/real_test/',
                 visualize=True):

        self.visualize = visualize

        self.obj_list = obj_list

        # load the object information
        self.cfg_list = cfg_list

        # load encoders and poses
        self.aae_list = []
        self.codebook_list = []
        self.rbpf_list = []
        self.rbpf_ok_list = []
        for obj in self.obj_list:
            ckpt_file = './checkpoints/aae_ckpts/{}/ckpt_{}_0300.pth'.format(ckpt_dir, obj)
            print(ckpt_file)
            codebook_file = './checkpoints/aae_ckpts/{}/codebook_{}_0300.pth'.format(ckpt_dir, obj)
            print(codebook_file)
            self.aae_full = AAE([obj], capacity=1, code_dim=128)
            self.aae_full.encoder.eval()
            self.aae_full.decoder.eval()
            for param in self.aae_full.encoder.parameters():
                param.requires_grad = False
            for param in self.aae_full.decoder.parameters():
                param.requires_grad = False
            checkpoint = torch.load(ckpt_file)
            self.aae_full.load_ckpt_weights(checkpoint['aae_state_dict'])
            self.aae_list.append(copy.deepcopy(self.aae_full.encoder))
            if not os.path.exists(codebook_file):
                print('Cannot find codebook in : ' + codebook_file)
                print('Start computing codebook ...')
                dataset_code = shapenet_codebook_online_generator(model_dir, obj,
                                                                  gpu_id=cfg.GPU_ID)
                self.aae_full.compute_codebook(dataset_code, codebook_file, save=True)
            else:
                print('Found codebook in : ' + codebook_file)

            self.codebook_list.append(torch.load(codebook_file)[0])
            self.rbpf_codepose = torch.load(codebook_file)[1].cpu().numpy()  # all are identical
            idx_obj = self.obj_list.index(obj)
            self.rbpf_list.append(particle_filter(self.cfg_list[idx_obj].PF, n_particles=self.cfg_list[idx_obj].PF.N_PROCESS))
            self.rbpf_ok_list.append(False)
        
        # renderer
        self.intrinsics = np.array([[self.cfg_list[0].PF.FU, 0, self.cfg_list[0].PF.U0],
                               [0, self.cfg_list[0].PF.FV, self.cfg_list[0].PF.V0],
                               [0, 0, 1.]], dtype=np.float32)
        
        self.renderer = render_wrapper(test_instance, test_model_dir, 0)
        
        # target object property
        self.target_obj = None
        self.target_obj_idx = None
        self.target_obj_encoder = None
        self.target_obj_codebook = None
        self.target_obj_cfg = None
        
        # initialize the particle filters
        self.rbpf = particle_filter(self.cfg_list[0].PF, n_particles=self.cfg_list[0].PF.N_PROCESS)
        self.rbpf_ok = False

        # pose rbpf for initialization
        self.rbpf_init_max_sim = 0

        # data properties
        self.data_with_gt = False
        self.data_with_est_bbox = False
        self.data_with_est_center = False
        self.data_intrinsics = np.ones((3, 3), dtype=np.float32)

        # initialize the PoseRBPF variables
        # ground truth information
        self.gt_available = False
        self.gt_t = [0, 0, 0]
        self.gt_rotm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        self.gt_bbox_center = np.zeros((3,))
        self.gt_bbox_size = 0
        self.gt_uv = np.array([0, 0, 1], dtype=np.float32)
        self.gt_z = 0
        self.gt_scale = 1.0

        # estimated states
        self.est_bbox_center = np.zeros((2, self.cfg_list[0].PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.cfg_list[0].PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.cfg_list[0].PF.N_PROCESS,))

        # for logging
        self.log_err_t = []
        self.log_err_tx = []
        self.log_err_ty = []
        self.log_err_tz = []
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []
        self.log_err_r = []
        self.log_err_t_star = []
        self.log_err_r_star = []
        self.log_max_sim = []
        self.log_dir = './'
        self.log_created = False
        self.log_shape_created = False
        self.log_pose = None
        self.log_shape = None
        self.log_error = None
        self.log_err_uv = []
        # posecnn prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # flags for experiments
        self.exp_with_mask = False
        self.step = 0
        self.iskf = False
        self.init_step = False
        self.save_uncertainty = False
        self.show_prior = False

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)
        self.embeddings_prev = None

        # deepsdf
        experiment_directory = './checkpoints/deepsdf_ckpts/{}s/'.format(obj_list[0])
        print(experiment_directory)
        self.decoder = load_decoder(experiment_directory, 2000)
        self.decoder = self.decoder.module.cuda()
        self.evaluator = Evaluator(self.decoder)
        latent_size = 256
        std_ = 0.01
        self.rand_tensor = torch.ones(1, latent_size).normal_(mean=0, std=std_)
        self.latent_tensor = self.rand_tensor.float().cuda()
        self.latent_tensor.requires_grad = False
        self.mask = None
        self.sdf_optim = deepsdf_optimizer(self.decoder, optimize_shape=False)
        test_model_dir = '../obj_models/real_test/'
        fn = '{}{}_vertices.txt'.format(test_model_dir, test_instance)
        points = np.loadtxt(fn, dtype=np.float32)   # n x 3
        self.size_gt = np.max(np.linalg.norm(points, axis=1))
        self.points_gt = torch.from_numpy(points)
        self.latent_tensor_initialized = False
        self.size_gt_pn = get_bbox_dist(points)
        self.size_est = self.size_gt_pn
        self.ratio = self.size_gt_pn/self.size_gt
        self.sdf_optim.ratio = self.ratio*1.0

        points_obj_norm = self.points_gt    
        # Transform from Nocs object frame to ShapeNet object frame
        rotm_obj2shapenet = euler.euler2mat(0.0, np.pi/2.0, 0.0)
        points_obj_shapenet = np.dot(rotm_obj2shapenet, points_obj_norm.T).T
        points_obj_shapenet = np.float32(points_obj_shapenet)
        self.points_c = torch.from_numpy(points_obj_shapenet)
        
        # partial point cloud observation
        self.points_o_partial = None 

        # pointnet++
        ckpt_file_pn = './checkpoints/latentnet_ckpts/{}/ckpt_{}_0300.pth'.format(self.obj_list[0], self.obj_list[0])
        self.label_gt_path = './latent_gt/{}_sfs.pth'.format(test_instance)
        pn_checkpoint = torch.load(ckpt_file_pn)
        self.model = get_model(input_channels=0)
        self.model.load_ckpt_weights(pn_checkpoint['aae_state_dict'])
        self.model.cuda()
        self.model.eval()
        for param in self.model.parameters():
                param.requires_grad = False
        self.label_gt = torch.load(self.label_gt_path)[0,:,:].detach()
        self.label_gt.requires_grad = False
        self.loss_shape_refine = 10000
        self.loss_last = 10000
        self.dist_init = 0
        self.dist_opt = 0

        # for debugging
        self.T_filter = None
        self.T_refine = None
        self.latent_vec_pointnet = None
        self.latent_vec_refine = None
        self.points_o_est_vis = None
        self.points_o_refine_vis = None
        self.points_gt_aug = torch.ones(self.points_gt.shape[0], self.points_gt.shape[1] + 1)
        self.points_gt_aug[:,:3] = self.points_gt
        self.points_gt_aug = self.points_gt_aug.cuda()
        self.fps = []
    # specify the target object for tracking
    def set_target_obj(self, target_object):
        assert target_object in self.obj_list, "target object {} is not in the list of test objects".format(target_object)
        # set target object property
        self.target_obj = target_object
        self.target_obj_idx = self.obj_list.index(target_object)
        self.target_obj_encoder = self.aae_list[self.target_obj_idx]
        self.target_obj_codebook = self.codebook_list[self.target_obj_idx]
        self.target_obj_cfg = self.cfg_list[self.target_obj_idx]

        self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
        self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
        self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
        self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]

        # reset particle filter
        self.rbpf = self.rbpf_list[self.target_obj_idx]
        self.rbpf_ok = self.rbpf_ok_list[self.target_obj_idx]
        self.rbpf_init_max_sim = 0

        # estimated states
        self.est_bbox_center = np.zeros((2, self.target_obj_cfg.PF.N_PROCESS))
        self.est_bbox_size = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))
        self.est_bbox_weights = np.zeros((self.target_obj_cfg.PF.N_PROCESS,))

        # for logging
        self.log_err_t = []
        self.log_err_tx = []
        self.log_err_ty = []
        self.log_err_tz = []
        self.log_err_rx = []
        self.log_err_ry = []
        self.log_err_rz = []
        self.log_err_r = []
        self.log_err_t_star = []
        self.log_err_r_star = []
        self.log_max_sim = []
        self.log_dir = './'
        self.log_created = False
        self.log_shape_created = False
        self.log_pose = None
        self.log_shape = None
        self.log_error = None

        # posecnn prior
        self.prior_uv = [0, 0, 1]
        self.prior_z = 0
        self.prior_t = np.array([0, 0, 0], dtype=np.float32)
        self.prior_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # motion model
        self.T_c1c0 = np.eye(4, dtype=np.float32)
        self.T_o0o1 = np.eye(4, dtype=np.float32)
        self.T_c0o = np.eye(4, dtype=np.float32)
        self.T_c1o = np.eye(4, dtype=np.float32)
        self.Tbr1 = np.eye(4, dtype=np.float32)
        self.Tbr0 = np.eye(4, dtype=np.float32)
        self.Trc = np.eye(4, dtype=np.float32)

        # print
        print('target object class is set to {}'.format(self.target_obj_cfg.PF.TRACK_OBJ))

    def visualize_roi(self, image, uv, z, scale, step, phase='tracking', show_gt=False, error=True, uncertainty=False, show=False, color='g', skip=False):
        image_disp = image

        # plt.figure()
        fig, ax = plt.subplots(1)

        ax.imshow(image_disp)

        plt.gca().set_axis_off()

        if skip:
            plt.axis('off')
            save_name = self.log_dir + '/{:06}_{}.png'.format(step, phase)
            plt.savefig(save_name)
            return

        if error == False:
            # set the margin to 0
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if show_gt:
            gt_bbox_center = self.gt_uv
            gt_bbox_size = 128 * self.target_obj_cfg.TRAIN.RENDER_DIST[0] / self.gt_z * self.target_obj_cfg.PF.FU / self.target_obj_cfg.TRAIN.FU * self.size_gt_pn

        est_bbox_center = uv
        est_bbox_size = 128 * self.target_obj_cfg.TRAIN.RENDER_DIST[0] * np.ones_like(z) / z * self.target_obj_cfg.PF.FU / self.target_obj_cfg.TRAIN.FU * scale

        if error:
            plt.plot(uv[:, 0], uv[:, 1], 'co', markersize=2)
            plt.plot(gt_bbox_center[0], gt_bbox_center[1], 'ro', markersize=5)
            plt.axis('off')

        t_v = self.rbpf.trans_bar
        center = project(t_v, self.data_intrinsics)
        plt.plot(center[0], center[1], 'co', markersize=5)

        if show_gt:
            rect = patches.Rectangle((gt_bbox_center[0] - 0.5*gt_bbox_size,
                                      gt_bbox_center[1] - 0.5*gt_bbox_size),
                                      gt_bbox_size,
                                      gt_bbox_size, linewidth=5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        for bbox_center, bbox_size in zip(est_bbox_center, est_bbox_size):
            rect = patches.Rectangle((bbox_center[0] - 0.5 * bbox_size,
                                      bbox_center[1] - 0.5 * bbox_size),
                                      bbox_size,
                                      bbox_size, linewidth=0.5, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        if error:
            z_show = [- 0.035, -0.07, -0.105, -0.14]
            plt.annotate('step {} - t err:        {:.3f} cm'.format(step, self.log_err_t[-1] * 100), xy=(0.15, z_show[0]),
                         xycoords='axes fraction')
            plt.annotate('step {} - R err:        {:.3f} deg'.format(step, self.log_err_r[-1]), xy=(0.15, z_show[1]),
                         xycoords='axes fraction')
            plt.annotate('step {} - max similarity:   {:.2f}'.format(step, self.log_max_sim[-1]), xy=(0.15, z_show[2]),
                         xycoords='axes fraction')

        plt.axis('off')

        save_name = self.log_dir + '/{:06}_{}.png'.format(step, phase)
        plt.savefig(save_name)

        if show:
            plt.show()
            # raw_input('show image')

        if uncertainty:
            ### save for uncertainty visualization ###
            rot_max = torch.sum(self.rbpf.rot.view(self.rbpf.n_particles, -1), dim=0) / self.rbpf.n_particles
            rot_max_c = rot_max.clone()
            rot_max = rot_max.view(1, 37, 72, 72)
            rot_max, _ = torch.max(rot_max, dim=3)
            np.savez(self.log_dir + '/rot_{:06}.npz'.format(step), rot=rot_max.cpu().numpy(),
                     rot_gt=mat2quat(self.gt_rotm))

    # logging
    def display_result(self, step, steps, refine=False):
        qco = mat2quat(self.gt_rotm)
        if not refine:
            self.T_filter = np.eye(4, dtype=np.float32)
            self.T_filter[:3, 3] = self.rbpf.trans_bar
            self.T_filter[:3, :3] = self.rbpf.rot_bar
        trans_bar_f = self.T_filter[:3, 3]
        rot_bar_f = self.T_filter[:3, :3]
        filter_rot_error_bar = abs(single_orientation_error(qco, mat2quat(rot_bar_f)))
        filter_trans_error_bar = np.linalg.norm(trans_bar_f - self.gt_t)

        trans_star_f = self.rbpf.trans_star
        rot_star_f = self.rbpf.rot_star

        filter_trans_error_star = np.linalg.norm(trans_star_f - self.gt_t)
        filter_rot_error_star = abs(single_orientation_error(qco, mat2quat(rot_star_f)))

        self.log_err_t_star.append(filter_trans_error_star)
        self.log_err_t.append(filter_trans_error_bar)
        self.log_err_r_star.append(filter_rot_error_star * 57.3)
        self.log_err_r.append(filter_rot_error_bar * 57.3)

        self.log_err_tx.append(np.abs(trans_bar_f[0] - self.gt_t[0]))
        self.log_err_ty.append(np.abs(trans_bar_f[1] - self.gt_t[1]))
        self.log_err_tz.append(np.abs(trans_bar_f[2] - self.gt_t[2]))

        rot_err_axis = single_orientation_error_axes(qco, mat2quat(rot_bar_f))
        self.log_err_rx.append(np.abs(rot_err_axis[0]))
        self.log_err_ry.append(np.abs(rot_err_axis[1]))
        self.log_err_rz.append(np.abs(rot_err_axis[2]))

        print('     step {}/{}: translation error (filter)   = {:.4f} cm'.format(step+1, int(steps),
                                                                                filter_trans_error_bar * 100))
        print('     step {}/{}: xyz error (filter)           = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                self.log_err_tx[-1
                                                                                                ] * 100,
                                                                                                self.log_err_ty[
                                                                                                    -1] * 100,
                                                                                                self.log_err_tz[
                                                                                                    -1] * 100))
        print('     step {}/{}: xyz rotation err (filter)    = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                self.log_err_rx[-1
                                                                                                ] * 57.3,
                                                                                                self.log_err_ry[
                                                                                                    -1] * 57.3,
                                                                                                self.log_err_rz[
                                                                                                    -1] * 57.3))
        print('     step {}/{}: rotation error (filter)      = {:.4f} deg'.format(step+1, int(steps),
                                                                                filter_rot_error_bar * 57.3))                                                                                            
        if not refine:
            return

        trans_refine_f = self.T_refine[:3, 3]
        rot_refine_f = self.T_refine[:3, :3]
        filter_trans_error_refine = np.linalg.norm(trans_refine_f - self.gt_t)
        filter_rot_error_refine = abs(single_orientation_error(qco, mat2quat(rot_refine_f)))

        log_err_tx_refine = []
        log_err_ty_refine = []
        log_err_tz_refine = []
        log_err_rx_refine = []
        log_err_ry_refine = []
        log_err_rz_refine = []
        log_err_tx_refine.append(np.abs(trans_refine_f[0] - self.gt_t[0]))
        log_err_ty_refine.append(np.abs(trans_refine_f[1] - self.gt_t[1]))
        log_err_tz_refine.append(np.abs(trans_refine_f[2] - self.gt_t[2]))

        rot_err_axis_refine = single_orientation_error_axes(qco, mat2quat(rot_refine_f))
        log_err_rx_refine.append(np.abs(rot_err_axis_refine[0]))
        log_err_ry_refine.append(np.abs(rot_err_axis_refine[1]))
        log_err_rz_refine.append(np.abs(rot_err_axis_refine[2]))

        print('     step {}/{}: translation error (refine)     = {:.4f} cm'.format(step+1, int(steps),
                                                                                filter_trans_error_refine * 100))

        print('     step {}/{}: xyz error (refine)           = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                log_err_tx_refine[-1
                                                                                                ] * 100,
                                                                                                log_err_ty_refine[
                                                                                                    -1] * 100,
                                                                                                log_err_tz_refine[
                                                                                                    -1] * 100))
        print('     step {}/{}: xyz rotation err (refine)    = ({:.4f}, {:.4f}, {:.4f})'.format(step + 1, int(steps),
                                                                                                log_err_rx_refine[-1
                                                                                                ] * 57.3,
                                                                                                log_err_ry_refine[
                                                                                                    -1] * 57.3,
                                                                                                log_err_rz_refine[
                                                                                                    -1] * 57.3))
        print('     step {}/{}: rotation error (refine)        = {:.4f} deg'.format(step+1, int(steps),
                                                                                filter_rot_error_refine * 57.3))

    def save_log(self, sequence, filename, with_gt=True):
        if not self.log_created:
            self.log_pose = open(self.log_dir + "/Pose_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence), "w+")
            if with_gt:
                self.log_pose_gt = open(self.log_dir + "/Pose_GT_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence),
                                     "w+")
                self.log_error = open(self.log_dir + "/Error_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence), "w+")
            self.log_created = True

        q_log = mat2quat(self.rbpf.rot_bar)
        self.log_pose.write('{} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                                               filename[0],
                                                                                               self.rbpf.trans_bar[0],
                                                                                               self.rbpf.trans_bar[1],
                                                                                               self.rbpf.trans_bar[2],
                                                                                               q_log[0],
                                                                                               q_log[1],
                                                                                               q_log[2],
                                                                                               q_log[3],
                                                                                               self.size_est,
                                                                                               self.fps))

        if with_gt:
            q_log_gt = mat2quat(self.gt_rotm)
            self.log_pose_gt.write(
                '{} {} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                                   filename[0],
                                                                                   self.gt_t[0],
                                                                                   self.gt_t[1],
                                                                                   self.gt_t[2],
                                                                                   q_log_gt[0],
                                                                                   q_log_gt[1],
                                                                                   q_log_gt[2],
                                                                                   q_log_gt[3],
                                                                                   self.size_gt_pn,
                                                                                   self.fps))
    def save_shape_log(self, sequence, filename):
        if not self.log_shape_created:
            self.log_shape = open(self.log_dir + "/Shape_{}_seq{}.txt".format(self.target_obj_cfg.PF.TRACK_OBJ, sequence), "w+")
  
            self.log_shape_created = True
        self.log_shape.write('{} {} {:.5f} {:.5f} \n'.format(self.target_obj_cfg.PF.TRACK_OBJ,
                                                                                               filename[0],
                                                                                               self.dist_init,
                                                                                               self.dist_opt))

    def display_overall_result(self):
        print('filter trans closest error = ', np.mean(np.asarray(self.log_err_t_star)))
        print('filter trans mean error = ', np.mean(np.asarray(self.log_err_t)))
        print('filter trans RMSE (x) = ', np.sqrt(np.mean(np.asarray(self.log_err_tx) ** 2)) * 1000)
        print('filter trans RMSE (y) = ', np.sqrt(np.mean(np.asarray(self.log_err_ty) ** 2)) * 1000)
        print('filter trans RMSE (z) = ', np.sqrt(np.mean(np.asarray(self.log_err_tz) ** 2)) * 1000)
        print('filter rot RMSE (x) = ', np.sqrt(np.mean(np.asarray(self.log_err_rx) ** 2)) * 57.3)
        print('filter rot RMSE (y) = ', np.sqrt(np.mean(np.asarray(self.log_err_ry) ** 2)) * 57.3)
        print('filter rot RMSE (z) = ', np.sqrt(np.mean(np.asarray(self.log_err_rz) ** 2)) * 57.3)
        print('filter rot closest error = ', np.mean(np.asarray(self.log_err_r_star)))
        print('filter rot mean error = ', np.mean(np.asarray(self.log_err_r)))

    def use_detection_priors(self, n_particles):
        self.rbpf.uv[-n_particles:] = np.repeat([self.prior_uv], n_particles, axis=0)
        self.rbpf.uv[-n_particles:, :2] += np.random.uniform(-self.target_obj_cfg.PF.UV_NOISE_PRIOR,
                                                             self.target_obj_cfg.PF.UV_NOISE_PRIOR,
                                                             (n_particles, 2))

    # initialize PoseRBPF
    def initialize_poserbpf(self, image, intrinsics, uv_init, n_init_samples, scale_prior, depth=None):
        # sample around the center of bounding box
        uv_h = np.array([uv_init[0], uv_init[1], 1])
        uv_h = np.repeat(np.expand_dims(uv_h, axis=0), n_init_samples, axis=0)
        uv_h[:, :2] += np.random.uniform(-self.target_obj_cfg.PF.INIT_UV_NOISE, self.target_obj_cfg.PF.INIT_UV_NOISE,
                                         (n_init_samples, 2))
        uv_h[:, 0] = np.clip(uv_h[:, 0], 0, image.shape[1])
        uv_h[:, 1] = np.clip(uv_h[:, 1], 0, image.shape[0])
        self.uv_init = uv_h.copy()

        uv_h_int = uv_h.astype(int)
        uv_h_int[:, 0] = np.clip(uv_h_int[:, 0], 0, image.shape[1] - 1)
        uv_h_int[:, 1] = np.clip(uv_h_int[:, 1], 0, image.shape[0] - 1)
        depth_np = depth.numpy()
        z = depth_np[uv_h_int[:, 1], uv_h_int[:, 0], 0]
        z = np.expand_dims(z, axis=1)
        z[z > 0] += np.random.uniform(-0.25, 0.05, z[z > 0].shape)
        z[z == 0] = np.random.uniform(0.5, 1.5, z[z == 0].shape)
        self.z_init = z.copy()

        scale_h = np.array([scale_prior])
        scale_h = np.repeat(np.expand_dims(scale_h, axis=0), n_init_samples, axis=0)
        scale_h += np.random.randn(n_init_samples, 1) * 0.05
        
        # evaluate translation
        distribution, _ = self.evaluate_particles(depth, uv_h, z, scale_h, self.target_obj_cfg.TRAIN.RENDER_DIST[0],
                                               0.1, depth,
                                               debug_mode=False)
        
        # find the max pdf from the distribution matrix
        self.index_star = my_arg_max(distribution)
        uv_star = uv_h[self.index_star[0], :]  # .copy()
        z_star = z[self.index_star[0], :]  # .copy()
        scale_star = scale_h[self.index_star[0], :]
        self.rbpf.update_trans_star_uvz(uv_star, z_star, scale_star, intrinsics)
        distribution[self.index_star[0], :] /= torch.sum(distribution[self.index_star[0], :])
        self.rbpf.rot = distribution[self.index_star[0], :].view(1, 1, 37, 72, 72).repeat(self.rbpf.n_particles, 1, 1, 1, 1)

        self.rbpf.update_rot_star_R(quat2mat(self.rbpf_codepose[self.index_star[1]][3:]))
        self.rbpf.rot_bar = self.rbpf.rot_star
        self.rbpf.trans_bar = self.rbpf.trans_star
        self.rbpf.uv_bar = uv_star
        self.rbpf.z_bar = z_star
        self.rbpf.scale_star = scale_star
        self.rbpf.scale_bar = scale_star
        self.rbpf_init_max_sim = self.log_max_sim[-1]
        
        # initialize shape latent vector
        depth_np = depth_np[:, :, 0] * self.mask
        depth_vis = depth_np*1.0
        points_np = depth2pc(depth_np, depth_np.shape[0], depth_np.shape[1], intrinsics)    # n x 4
        points_c = torch.from_numpy(points_np).cuda()
        T_init = np.eye(4, dtype=np.float32)
        T_init[:3, :3] = self.rbpf.rot_bar
        T_init[:3, 3] = self.rbpf.trans_bar
        self.size_est = self.rbpf.scale_bar[0]

        if len(points_np) > 0:
            self.sdf_optim.size_est = torch.tensor(self.size_est/self.ratio).cuda()
            # with this 
            points_choice = self.sample_points(points_np[:,:3])
            self.latent_vector_prediction_pn(points_choice, self.rbpf.rot_bar, self.rbpf.trans_bar)
            self.latent_vec_optim = self.latent_tensor.clone()
        else:
            print('*** NOTHING ON THE DEPTH IMAGE! ... USING ORIGINAL BACKPROGATION METHOD ***')
            self.initialize_latent_vector(points_c, T_init)

    def latent_vector_prediction_pn(self, points_choice, rot, trans):

        # Transform from camera frame to object frame
        points_obj = np.dot(rot.T, (points_choice.T - np.tile(trans, (len(points_choice),1)).T)).T

        # Normalization  
        points_obj_norm = points_obj/self.size_est

        # Transform from Nocs object frame to ShapeNet object frame
        rotm_obj2shapenet = euler.euler2mat(0.0, np.pi/2.0, 0.0)
        points_obj_shapenet = np.dot(rotm_obj2shapenet, points_obj_norm.T).T
        points_obj_shapenet = np.float32(points_obj_shapenet)
        # visualize_depth_pc(points_obj_norm, points_obj_shapenet)
        self.points_o_partial = points_obj_shapenet * self.size_est

        points_c = torch.from_numpy(points_obj_shapenet).cuda().unsqueeze(0)
        pred_label = self.model(points_c)
        self.latent_tensor = pred_label.clone()
        
    # evaluate particles according to the RGB(D) images
    def evaluate_particles(self, image,
                           uv, z, scale,
                           render_dist, gaussian_std,
                           depth,
                           debug_mode=False,
                           run_deep_sdf=False):

        # crop the rois from input depth image
        images_roi_cuda = trans_zoom_uvz_cuda(depth.detach(), uv, z, scale,
                                                self.target_obj_cfg.PF.FU,
                                                self.target_obj_cfg.PF.FV,
                                                render_dist).detach()
        
        # normalize the depth
        n_particles = z.shape[0]
        z_cuda = torch.from_numpy(z).float().cuda().unsqueeze(2).unsqueeze(3)
        scale_cuda = torch.from_numpy(scale).float().cuda().unsqueeze(2).unsqueeze(3)
        images_roi_cuda = (images_roi_cuda - z_cuda) / scale_cuda + 0.5
        images_roi_cuda = torch.clamp(images_roi_cuda, 0, 1)
        
        # forward passing
        codes = self.target_obj_encoder.forward(images_roi_cuda).view(n_particles, -1).detach()
        self.inputs = images_roi_cuda
        # self.recon, self.code_rec = self.aae_full.forward(images_roi_cuda)

        # compute the similarity between particles' codes and the codebook
        cosine_distance_matrix = self.aae_full.compute_distance_matrix(codes, self.target_obj_codebook)

        # prior distance
        if self.embeddings_prev is not None:
            prior_dists = self.aae_full.compute_distance_matrix(codes, self.embeddings_prev)
            if torch.max(prior_dists) < 0.8:
                prior_dists = None
        else:
            prior_dists = None


        if prior_dists is not None:
            cosine_distance_matrix = cosine_distance_matrix + prior_dists.repeat(1, cosine_distance_matrix.size(1))

        # get the maximum similarity for each particle
        v_sims, i_sims = torch.max(cosine_distance_matrix, dim=1)
        self.cos_dist_mat = v_sims

        # compute distribution from similarity
        max_sim_all, i_sim_all = torch.max(v_sims, dim=0)
        self.log_max_sim.append(max_sim_all)
        pdf_matrix = mat2pdf(cosine_distance_matrix/max_sim_all, 1, gaussian_std)

        # evaluate with deepsdf
        sdf_scores = torch.from_numpy(np.ones_like(z)).cuda().float()

        if run_deep_sdf:
            distances = np.ones_like(z)
            depth_np = depth.numpy()[:, :, 0] * self.mask
            points_np = depth2pc(depth_np, depth_np.shape[0], depth_np.shape[1], self.intrinsics)
            points_c = torch.from_numpy(points_np).cuda()
            if len(points_c) > NP_THRESHOLD:
                for i in range(self.rbpf.n_particles):
                    R = quat2mat(self.rbpf_codepose[i_sims[i]][3:])
                    t = back_project(uv[i], self.intrinsics, z[i])
                    R = add_trans_R(t, R)
                    T_co = np.eye(4, dtype=np.float32)
                    T_co[:3, :3] = R
                    T_co[:3, 3] = t
                    distances[i] = self.sdf_optim.eval_latent_vector(T_co, points_c, self.latent_tensor)
                
                distances_temp = distances/np.min(distances)
                distances_torch = torch.from_numpy(distances_temp).cuda().float()
                sdf_scores = mat2pdf(distances_torch, 1.0, 0.5)
                pdf_matrix = torch.mul(pdf_matrix, sdf_scores)

        # visualize reconstruction
        if debug_mode:
            depth_recon = self.aae_full.decoder.forward(codes).detach()
            depth_recon_disp = depth_recon[i_sim_all][0].cpu().numpy()
            depth_input_disp = images_roi_cuda[i_sim_all][0].cpu().numpy()
            depth_disp = depth[:, :, 0].cpu().numpy()
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(depth_input_disp)
            plt.subplot(1, 3, 2)
            plt.imshow(depth_recon_disp)
            plt.subplot(1, 3, 3)
            plt.imshow(depth_disp)
            plt.plot(self.prior_uv[0], self.prior_uv[1], 'co', markersize=2)
            plt.show()
        

        return pdf_matrix, prior_dists

    def debug_gt_particles(self, n_gt_particles):
        self.rbpf.uv[-n_gt_particles:] = np.repeat([self.gt_uv], n_gt_particles, axis=0)
        self.rbpf.z[-n_gt_particles:] = np.ones_like(self.rbpf.z[-n_gt_particles:]) * self.gt_z
        self.rbpf.scale[-n_gt_particles:] = np.ones_like(self.rbpf.z[-n_gt_particles:]) * 0.2

    # filtering
    def process_poserbpf(self, image, intrinsics, use_detection_prior=True, depth=None, debug_mode=False, run_deep_sdf=False):

        # propagation
        uv_noise = self.target_obj_cfg.PF.UV_NOISE
        z_noise = self.target_obj_cfg.PF.Z_NOISE
        self.rbpf.add_noise_r3(uv_noise, z_noise)
        self.rbpf.add_noise_rot()

        # poserbpf++
        if use_detection_prior and self.prior_uv[0] > 0 and self.prior_uv[1] > 0:
            self.use_detection_priors(int(self.rbpf.n_particles / 2))

        # compute pdf matrix for each particle
        est_pdf_matrix, prior_dists = self.evaluate_particles(depth, self.rbpf.uv, self.rbpf.z, self.rbpf.scale,
                                                   self.target_obj_cfg.TRAIN.RENDER_DIST[0],
                                                   self.target_obj_cfg.PF.WT_RESHAPE_VAR,
                                                   depth, debug_mode=debug_mode, run_deep_sdf=run_deep_sdf)

        # most likely particle
        temp_indext_star = my_arg_max(est_pdf_matrix)
        if temp_indext_star is not None:
            self.index_star = temp_indext_star
        else:
            title = "Exception debug"
            plt.figure(title)
            plt.subplot(1,2,1)
            plt.imshow(image.cpu().numpy() / 255.0)
            plt.subplot(1,2,2)
            plt.imshow(depth[:,:,0].cpu().numpy())
            plt.show()
            import pdb;pdb.set_trace()
        uv_star = self.rbpf.uv[self.index_star[0], :].copy()
        z_star = self.rbpf.z[self.index_star[0], :].copy()
        # scale_star = self.rbpf.scale[self.index_star[0], :].copy()
        self.rbpf.update_trans_star(uv_star, z_star, intrinsics)
        self.rbpf.update_rot_star_R(quat2mat(self.rbpf_codepose[self.index_star[1]][3:]))

        # match rotation distribution
        self.rbpf.rot = torch.clamp(self.rbpf.rot, 1e-6, 1)
        rot_dist = torch.exp(torch.add(torch.log(est_pdf_matrix), torch.log(self.rbpf.rot.view(self.rbpf.n_particles, -1))))
        normalizers = torch.sum(rot_dist, dim=1)

        normalizers_cpu = normalizers.cpu().numpy()
        self.rbpf.weights = normalizers_cpu / np.sum(normalizers_cpu)

        if prior_dists is not None:
            self.rbpf.weights = self.rbpf.weights
            self.rbpf.weights /= np.sum(self.rbpf.weights)
        
        rot_dist /= normalizers.unsqueeze(1).repeat(1, self.target_obj_codebook.size(0))

        # matched distributions
        self.rbpf.rot = rot_dist.view(self.rbpf.n_particles, 1, 37, 72, 72)
        
        # resample
        self.rbpf.resample_ddpf(self.rbpf_codepose, intrinsics, self.target_obj_cfg.PF)
        self.size_est = self.rbpf.scale_bar[0]

        # compute previous embeddings
        images_roi_cuda = trans_zoom_uvz_cuda(depth.detach(), np.expand_dims(self.rbpf.uv_bar, 0), np.expand_dims(self.rbpf.z_bar, 0), np.expand_dims(self.rbpf.scale_bar, 0),
                                                self.target_obj_cfg.PF.FU,
                                                self.target_obj_cfg.PF.FV,
                                                self.target_obj_cfg.TRAIN.RENDER_DIST[0]).detach()
        
        # normalize the depth
        z = np.expand_dims(self.rbpf.z_bar, 0)
        z_cuda = torch.from_numpy(z).float().cuda().unsqueeze(2).unsqueeze(3)
        scale_cuda = torch.from_numpy(np.expand_dims(self.rbpf.scale_bar, 0)).float().cuda().unsqueeze(2).unsqueeze(3)
        images_roi_cuda = (images_roi_cuda - z_cuda) / scale_cuda + 0.5
        images_roi_cuda = torch.clamp(images_roi_cuda, 0, 1)
        
        # forward passing
        codes = self.target_obj_encoder.forward(images_roi_cuda).view(1, -1).detach()
        self.embeddings_prev = codes

        return 0

    def refine_pose_and_shape(self, depth, intrinsics, refine_steps=50):

        # initialize shape latent vector
        depth_np = depth.numpy()[:, :, 0] * self.mask
        points_np = depth2pc(depth_np, depth_np.shape[0], depth_np.shape[1], intrinsics.numpy()[0])
        points_c = torch.from_numpy(points_np).cuda()

        if len(points_np) > NP_THRESHOLD:
            points_choice = self.sample_points(points_np[:,:3])
            torch.cuda.synchronize()
            time_start = time.time()
            self.latent_vector_prediction_pn(points_choice, self.rbpf.rot_bar, self.rbpf.trans_bar)
            torch.cuda.synchronize()
            time_elapse = time.time() - time_start
            print("LatentNet inference time = ", time_elapse)
            if len(points_np) > NPOINTS:
                points_c = torch.from_numpy(np.hstack((points_choice,np.ones((points_choice.shape[0], 1), dtype=np.float32)))).cuda()

        else:
            print('*** NOT ENOUGH POINTS ON THE DEPTH IMAGE! ... SKIP REFINEMENT FOR THIS FRAME ***')
            return
        # debug information
        self.latent_vec_pointnet = self.latent_tensor.clone().detach()

        T_co = np.eye(4, dtype=np.float32)
        T_co[:3, :3] = self.rbpf.rot_bar
        T_co[:3, 3] = self.rbpf.trans_bar

        self.sdf_optim.size_est = torch.tensor(self.size_est/self.ratio).cuda()

        # ******************************************************************************************************* 
        T_co_opt, dist, _, size_est, loss_optim = self.sdf_optim.refine_pose(T_co, points_c, self.latent_tensor,
                                                                                  steps=refine_steps, shape_only=False)
        # Let's do this multiple times
        for idx in range(2):
            self.latent_vector_prediction_pn(points_choice, T_co_opt[:3, :3], T_co_opt[:3, 3])
            T_co_opt, dist, _, size_est, loss_optim = self.sdf_optim.refine_pose(T_co_opt, points_c, self.latent_tensor,
                                                                            steps=refine_steps, shape_only=False)
        # *******************************************************************************************************
        
        # debug information
        self.latent_vec_refine = self.latent_tensor.clone().detach() 
        self.T_filter = T_co
        self.T_refine = T_co_opt

        self.rbpf.rot_bar = T_co_opt[:3, :3]
        self.rbpf.trans_bar = T_co_opt[:3, 3]

        return 0

    def initialize_latent_vector(self, points_c, T_co_init):
        dist_min = 100000000
        for _ in range(500):
            latent_size = 256
            std_ = 0.1
            rand_tensor = torch.ones(1, latent_size).normal_(mean=0, std=std_)
            latent_tensor_rand = rand_tensor.float().cuda()
            latent_tensor_rand.requires_grad = False
            dist = self.sdf_optim.eval_latent_vector(T_co_init, points_c, latent_tensor_rand)
            if dist < dist_min:
                self.latent_tensor = latent_tensor_rand.clone()
                dist_min = dist

    # run dataset
    def run_nocs_dataset(self, val_dataset, sequence):

        if self.aae_full.angle_diff.shape[0] != 0:
            self.aae_full.angle_diff = np.array([])

        self.log_err_r = []
        self.log_err_r_star = []
        self.log_err_t = []
        self.log_err_t_star = []
        self.log_dir = self.target_obj_cfg.PF.SAVE_DIR+'seq_{}'.format(sequence)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                    shuffle=False, num_workers=0)
        steps = len(val_dataset)

        step = 0
        detect_flag = False

        for inputs in val_generator:

            if len(inputs) == 8:
                if step == 0:
                    print('RUNNING NOCS Real DATASET ! ')
                images, poses_gt, intrinsics, depths, masks, file_names, scale_gt, bbox = inputs
                
                self.data_intrinsics = intrinsics[0].numpy()
                self.intrinsics = intrinsics[0].numpy()
                self.target_obj_cfg.PF.FU = self.intrinsics[0, 0]
                self.target_obj_cfg.PF.FV = self.intrinsics[1, 1]
                self.target_obj_cfg.PF.U0 = self.intrinsics[0, 2]
                self.target_obj_cfg.PF.V0 = self.intrinsics[1, 2]
                
                self.data_with_est_center = False
                self.data_with_gt = True

                self.mask_raw = masks[0, :, :, 0].cpu().numpy()
                self.mask = ndimage.binary_erosion(self.mask_raw, iterations=2).astype(self.mask_raw.dtype)    
                # ground truth for visualization
                pose_gt = poses_gt.numpy()[0, :, :]
                self.gt_t = pose_gt[:3, 3]
                self.gt_rotm = pose_gt[:3, :3]
                gt_center = np.matmul(intrinsics, self.gt_t)
                if gt_center.shape[0] == 1:
                    gt_center = gt_center[0]
                gt_center = gt_center / gt_center[2]
                self.gt_uv[:2] = gt_center[:2]
                self.gt_z = self.gt_t[2]
                self.gt_scale = scale_gt[0]

                self.prior_uv[0] = (bbox[0, 2] + bbox[0, 3])/2
                self.prior_uv[1] = (bbox[0, 0] + bbox[0, 1])/2
                self.log_err_uv.append(np.linalg.norm(self.prior_uv[:2] - self.gt_uv[:2]))

                if self.prior_uv[0] > 0 and self.prior_uv[1]:
                    detect_flag = True
                else:
                    detect_flag = False
                    print('NOT detected!')
            else:
                print('*** INCORRECT DATASET SETTING! ***')
                break
            
            # skip the unlabeled frames
            if self.gt_z == 0:
                self.data_with_gt = False

            self.step = step

            # initialization
            if step == 0:
                while not self.rbpf_ok:    
                    if self.target_obj_cfg.PF.USE_DEPTH:
                        depth_data = depths[0]
                    else:
                        depth_data = None

                    if self.prior_uv[0] > 0 and self.prior_uv[1] > 0:
                        print('[Initialization] Initialize PoseRBPF with detection center ... ')
                        self.initialize_poserbpf(images[0].detach(),
                                                self.data_intrinsics,
                                                self.prior_uv[:2],
                                                self.target_obj_cfg.PF.N_INIT,
                                                scale_prior=self.target_obj_cfg.PF.SCALE_PRIOR,
                                                depth=depth_data)

                        if self.data_with_gt:
                            init_error = np.linalg.norm(self.rbpf.trans_star - self.gt_t)
                            print('     Initial translation error = {:.4} cm'.format(init_error * 100))
                            init_rot_error = abs(single_orientation_error(mat2quat(self.gt_rotm), mat2quat(self.rbpf.rot_star)))
                            print('     Initial rotation error    = {:.4} deg'.format(init_rot_error * 57.3))

                        self.rbpf_ok = True

                        # to avoid initialization to symmetric view and cause abnormal results
                        if init_rot_error * 57.3 > 60 and self.target_obj != 'bowl':
                            self.rbpf_ok = False
                            self.embeddings_prev = None
            
            # filtering
            if self.rbpf_ok:
                torch.cuda.synchronize()
                time_start = time.time()

                if self.target_obj_cfg.PF.USE_DEPTH:
                    depth_data = depths[0]
                else:
                    depth_data = None
                refine_start = 5    #1000
                self.process_poserbpf(images[0], intrinsics, depth=depth_data, run_deep_sdf= False) #(step > refine_start and detect_flag)
                #kf_refine = step % 10 == 0
                if step > refine_start and detect_flag:
                    self.refine_pose_and_shape(depth_data, intrinsics)

                torch.cuda.synchronize()
                time_elapse = time.time() - time_start
                self.fps = 1.0/time_elapse
                print(
                    '[Filtering: {}] {}/{} fps = {:.2f}, max sim = {:.2f}, scale est = {:.5f}'.format(
                        self.target_obj,
                        step + 1, steps,
                        1 / time_elapse,
                        self.log_max_sim[-1],
                        self.size_est))
                print('size_gt_pn = ', self.size_gt_pn)

                # logging
                if self.data_with_gt:
                    self.display_result(step, steps, step>refine_start and detect_flag)  #(step>refine_start and detect_flag)
                    self.save_log(sequence, file_names, with_gt=self.data_with_gt)

                    # visualization
                    is_kf = step % 20 == 0
                    if is_kf:

                        image_disp = images[0].cpu().numpy() / 255.0

                        image_est_render, _ = self.renderer.render_pose(self.rbpf.trans_bar,
                                                                         self.rbpf.rot_bar)

                        image_est_disp = image_est_render[0].permute(1, 2, 0).cpu().numpy()
                        image_est_disp[:, :, 0] *= 0
                        image_est_disp[:, :, 2] *= 0
                        
                        image_disp = 0.4 * image_disp + 0.6 * image_est_disp
                        self.visualize_roi(image_disp, self.rbpf.uv, self.rbpf.z, self.rbpf.scale, step,
                                           show_gt=True, error=True, uncertainty=self.show_prior, show=False,
                                           skip=False)
                        plt.close()

                        # evaluate chamfer distance and save point cloud for visualization
                        # if step > refine_start and detect_flag:
                        #     self.dist_init, self.dist_opt, points_init, points_opt = visualize_shape_comparison(self.points_c, self.label_gt, self.latent_vec_pointnet, self.latent_vec_refine, self.evaluator, self.decoder, self.size_est/self.ratio)
                        #     self.save_shape_log(sequence, file_names)
                            
                        #     shape_output_name = self.log_dir + "/shape_pcd_est_{}_{}.ply".format(self.target_obj_cfg.PF.TRACK_OBJ, step)
                        #     pcd = o3d.geometry.PointCloud()
                        #     pcd.points = o3d.utility.Vector3dVector(points_opt[:,:3])
                        #     o3d.io.write_point_cloud(shape_output_name, pcd)

                        #     shape_output_name = self.log_dir + "/shape_pcd_partial_{}_{}.ply".format(self.target_obj_cfg.PF.TRACK_OBJ, step)
                        #     pcd_ob = o3d.geometry.PointCloud()
                        #     pcd_ob.points = o3d.utility.Vector3dVector(self.points_o_partial)
                        #     o3d.io.write_point_cloud(shape_output_name, pcd_ob)
                        
            if step == steps-1:
                break
            step += 1

        self.log_pose.close()
        self.display_overall_result()

    def sample_points(self, points_np, npoints = NPOINTS):
        if npoints < len(points_np):
            choice = pointnet2_utils.furthest_point_sample(torch.from_numpy(points_np).cuda().unsqueeze(0).contiguous(), npoints).cpu().numpy()[0]
            points_choice = points_np[choice,:]
        else:
            choice = np.arange(0, len(points_np), dtype=np.int32)
            #print('length_point=', len(points_np))
            if npoints > len(points_np):
                lc = len(choice)
                le = npoints - len(points_np)
                if lc < le:
                    extra_choice = []
                    for i in range(le):
                        x = np.random.randint(lc)
                        extra_choice.append(x)
                    extra_choice = np.asarray(extra_choice)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                else:
                    extra_choice = np.random.choice(choice, le, replace = False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
            points_choice = points_np[choice,:]  

        return points_choice