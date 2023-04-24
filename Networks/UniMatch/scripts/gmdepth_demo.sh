#!/usr/bin/env bash


# gmdepth-scale1-regrefine1
CUDA_VISIBLE_DEVICES=0 python3.9 main_depth.py \
--inference_dir demo/depth_test \
--output_path output/gmdepth-test \
--resume pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth \
--reg_refine \
--num_reg_refine 1

# --pred_bidir_depth

