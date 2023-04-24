#!/usr/bin/env bash


# gmstereo-scale2-regrefine3 model
CUDA_VISIBLE_DEVICES=0  \
python3.9 main_stereo.py \
--inference_size 1024 1536 \
--output_path /home/godeta/PycharmProjects/LYNRED/Stereo_matching/NeuralNetwork/UniMatch/output/result \
--resume checkpoints_stereo/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3 \
--inference_dir demo/dataset \


#--inference_dir_left /home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/visible/left \
#--inference_dir_right /home/godeta/PycharmProjects/LYNRED/LynredDataset/Day/visible/right \


# optionally predict both left and right disparities
#--pred_bidir_disp
#--inference_dir_left /home/godeta/PycharmProjects/LYNRED/Video_frame/Day/visible/left \
#--inference_dir_right /home/godeta/PycharmProjects/LYNRED/Video_frame/Day/visible/right \



