#!/usr/bin/env bash


## gmflow-scale2-regrefine6, inference on image dir
#CUDA_VISIBLE_DEVICES=0 python3.9 main_flow.py \
#--inference_dir /home/godeta/PycharmProjects/LYNRED/Stereo_matching/NeuralNetwork/UniMatch/demo/flow-davis \
#--resume pretrained/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth \
#--output_path output/test_flow \
#--padding_factor 32 \
#--upsample_factor 4 \
#--num_scales 2 \
#--attn_splits_list 2 8 \
#--corr_radius_list -1 4 \
#--prop_radius_list -1 1 \
#--reg_refine \
#--num_reg_refine 3 \
##--print_info

# gmflow-scale2-regrefine6, inference on image visible
CUDA_VISIBLE_DEVICES=0 python3.9 main_flow.py \
--inference_dir /home/godeta/PycharmProjects/LYNRED/Stereo_matching/NeuralNetwork/UniMatch/demo/flow-davis \
--resume pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth \
--output_path output/test_flow \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 6 \
#--print_info

# gmflow-scale2-regrefine6, inference on image infrared
#CUDA_VISIBLE_DEVICES=0 python3.9 main_flow.py \
#--inference_dir /home/godeta/PycharmProjects/LYNRED/Video_frame/Day/infrared/left \
#--resume pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth \
#--output_path output/dataset/flow_ir \
#--padding_factor 32 \
#--upsample_factor 4 \
#--num_scales 2 \
#--attn_splits_list 2 8 \
#--corr_radius_list -1 4 \
#--prop_radius_list -1 1 \
#--reg_refine \
#--num_reg_refine 6 \
#--print_info

# gmflow-scale2-regrefine6, inference on video, save as video
#CUDA_VISIBLE_DEVICES=0 python3.9 main_flow.py \
#--inference_video demo/kitti.mp4 \
#--resume pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth \
#--output_path output/kitti \
#--padding_factor 32 \
#--upsample_factor 4 \
#--num_scales 2 \
#--attn_splits_list 2 8 \
#--corr_radius_list -1 4 \
#--prop_radius_list -1 1 \
#--reg_refine \
#--num_reg_refine 6 \
#--save_video \
#--concat_flow_img



# gmflow-scale1, inference on image dir
#CUDA_VISIBLE_DEVICES=0 python3.9 main_flow.py \
#--inference_dir demo/flow-davis \
#--resume pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth \
#--output_path output/gmflow-scale1-davis

# optional predict bidirection flow and forward-backward consistency check
#--pred_bidir_flow
#--fwd_bwd_check


# gmflow-scale2, inference on image dir
#CUDA_VISIBLE_DEVICES=0 python3.9 main_flow.py \
#--inference_dir /home/godeta/PycharmProjects/LYNRED/Images/test/ \
#--resume pretrained/gmflow-scale2-regrefine6-sintelft-6e39e2b9.pth \
#--output_path output/test_flow \
#--padding_factor 32 \
#--upsample_factor 4 \
#--num_scales 2 \
#--attn_splits_list 2 8 \
#--corr_radius_list -1 4 \
#--prop_radius_list -1 1



