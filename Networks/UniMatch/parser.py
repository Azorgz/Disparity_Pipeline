import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='checkpoints_stereo/StereoHybrid', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--path_checkpoint', default='checkpoints_stereo/StereoHybrid', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='sceneflow', type=str,
                        help='training stage on different datasets')
    parser.add_argument('--val_dataset', default=['kitti15'], type=str, nargs='+')
    parser.add_argument('--max_disp', default=400, type=int,
                        help='exclude very large disparity in the loss function')
    parser.add_argument('--img_height', default=288, type=int)
    parser.add_argument('--img_width', default=512, type=int)
    parser.add_argument('--padding_factor', default=16, type=int)

    # # training
    # parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--num_workers', default=8, type=int)
    # parser.add_argument('--lr', default=1e-3, type=float)
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    # parser.add_argument('--seed', default=326, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default='checkpoints_stereo/gmstereo-scale2-regrefine3-resumeflowthings-eth3dft-a807cb16.pth', type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    parser.add_argument('--strict_resume', action='store_true',
                        help='strict resume while loading pretrained weights')
    parser.add_argument('--no_resume_optimizer', action='store_true')
    parser.add_argument('--resume_exclude_upsampler', action='store_true')

    # model: learnable parameters
    parser.add_argument('--task', default='stereo', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    # model: parameter-free
    parser.add_argument('--attn_type', default='self_swin2d_cross_1d', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--count_time', action='store_true')
    parser.add_argument('--save_vis_disp', action='store_true')
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--middlebury_resolution', default='F', choices=['Q', 'H', 'F'])

    # submission
    parser.add_argument('--submission', action='store_true')
    parser.add_argument('--eth_submission_mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--middlebury_submission_mode', default='training', type=str, choices=['training', 'test'])
    parser.add_argument('--output_path', default='output/result', type=str)

    # log
    parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=1000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--val_freq', default=1000, type=int, help='validation frequency in terms of training steps')
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=100000, type=int)

    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', default='none', type=str)
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    # inference
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_dir_left', default=None, type=str)
    parser.add_argument('--inference_dir_right', default=None, type=str)
    parser.add_argument('--pred_bidir_disp', action='store_true',
                        help='predict both left and right disparities')
    parser.add_argument('--pred_right_disp', action='store_true',
                        help='predict right disparity')
    parser.add_argument('--save_pfm_disp', action='store_true',
                        help='save predicted disparity as .pfm format')

    parser.add_argument('--debug', action='store_true')

    return parser