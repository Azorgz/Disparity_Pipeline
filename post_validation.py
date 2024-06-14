import glob
import os

import oyaml
from tqdm import tqdm

from config.Config import ConfigPipe
from module.Validation import Validation
from utils.ImagesCameras import CameraSetup, ImageTensor

config = ConfigPipe()
config['validation']['activated'] = True


def post_validation(path):
    with open(path + '/CumMask.yaml', 'r') as file:
        MaskCum = oyaml.safe_load(file)
        roi = MaskCum["Cumulative Mask"]
    with open(path + '/Summary_experiment.yaml', 'r') as file:
        summary = oyaml.safe_load(file)
    cam_src = summary["Wrap"]["cam_src"]
    cam_dst = summary["Wrap"]["cam_dst"]
    with open(path + '/dataset.yaml', 'r') as file:
        dataset = oyaml.safe_load(file)
    nb_sample = int(dataset["Number of sample"])
    setup = CameraSetup(from_file=os.getcwd() + '/' + dataset["Setup"])
    validation = Validation(config)
    validation.activated = True
    validation.post_validation = False

    def valid(s, **kwargs):
        name = f'{cam_src}_to_{cam_dst}'
        im_ref = s[cam_dst]
        image_reg = s['image_reg'].match_shape(im_ref)
        im_old = s[cam_src].match_shape(im_ref)
        validation(image_reg, im_ref, im_old, name, path.split('/')[-1], roi=roi)

    with tqdm(total=nb_sample,
              desc=f"Nombre d'itérations for {path.split('/')[-1]}: ", leave=True, position=0) as bar:
        for im_src, im_dst, im_reg in tqdm(zip(setup.cameras[cam_src],
                                               setup.cameras[cam_dst],
                                               sorted(glob.glob(path + f'/image_reg/{cam_src}_to_{cam_dst}/*')))):
            sample = {cam_dst: ImageTensor(im_dst, device=validation.device),
                      cam_src: ImageTensor(im_src, device=validation.device),
                      'image_reg': ImageTensor(im_reg, device=validation.device)}
            valid(sample)
            bar.update(1)
        validation.statistic()
        validation.save(path)


if __name__ == '__main__':
    experiment = 'Test'
    for p in sorted(glob.glob(os.getcwd() + f'/results/{experiment}/*')):
        post_validation(p)
