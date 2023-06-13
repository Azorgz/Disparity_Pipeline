import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
    parser.add_argument('--path_checkpoint', default=None, type=str,
                        help='where to save the training log and models')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+')
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--padding_factor', default=16, type=int)
    parser.add_argument('--attention_weights_only', default=False, type=str, help='only train attention weights')
    parser.add_argument('--freeze_attention_weights', default=True, type=str, help='freeze attention weights parameters')
    parser.add_argument('--launcher', default=None, type=str)
    parser.add_argument('--half_resolution', action='store_true')
    parser.add_argument('--mode', default='test', help='select train or test mode', choices=['test', 'train'])
    # # parser.add_argument('--test_batch_size', type=int, default=16, help='testing batch size')

    return parser