import os
from pathlib import Path
from typing import Union
import cv2 as cv
import numpy as np
import oyaml as yaml
from kornia.utils import get_cuda_device_if_available
from tqdm import tqdm

from utils.classes import ImageTensor, DepthTensor
from utils.classes.Metrics import Metric_nec_tensor, Metric_ssim_tensor
from utils.classes.Vizualisation.VideoGenerator import VideoGenerator
from utils.gradient_tools import grad_image


class Visualizer:
    show_validation = False
    show_grad_im = 0
    show_occlusion = False
    show_disp_overlay = False
    show_idx = True
    font = cv.FONT_HERSHEY_TRIPLEX
    color = (255, 255, 255)
    org_idx = (10, 20)
    thickness = 1
    fontScale = 0.5
    idx = 0
    key = 0
    tensor = False
    window = 0

    def __init__(self, path: Union[str, Path, list] = None, search_exp=False):
        """
        :param path: Path to the result folder
        :return: None
        path ___|input|target
                |reg_images
                |disp_target
                |disp_ref
                |occlusion
                |Validation.yaml
        To navigate use the arrows or +/- or specify an index using the num pad and validate with Enter.
                |     |ref
        To quit press Escape
        To show/hide the current index press i
        To show/hide the overlay of disparity press d
        To show/hide the validation indexes (only available with the validation done) press v
        """
        self.multi_setup = False
        if path is None or search_exp:
            if path is None:
                p = "/home/godeta/PycharmProjects/Disparity_Pipeline/results"
            else:
                p = path
            path = [p + f'/{d}' for d in os.listdir(p) if os.path.isdir(p + f'/{d}')]
        if isinstance(path, list):
            self.exp_list = []
            self.path = []
            for pa in path:
                self.exp_list.append(*os.listdir(pa + '/image_reg'))
                self.path.append(pa)
        else:
            self.exp_list = os.listdir(path + '/image_reg')
            self.path = [path]
        self.experiment = {}
        for idx, (p, P) in enumerate(zip(self.exp_list, self.path)):
            ref, target = p.split('_to_')
            exp_name = os.path.split(P)[1]
            p = f'{exp_name} - {p}'
            self.exp_list[idx] = p
            self.experiment[p] = {}
            new_path, _, new_list = os.walk(f'{P}/image_reg/{ref}_to_{target}').__next__()
            if os.path.exists(f'{P}/inputs'):
                target_path, _, target_list = os.walk(f'{P}/inputs/{target}').__next__()
                ref_path, _, ref_list = os.walk(f'{P}/inputs/{ref}').__next__()
                target_list = sorted(target_list)
                ref_list = sorted(ref_list)

            elif os.path.exists(f'{P}/dataset.yaml'):
                with open(f'{P}/dataset.yaml', "r") as file:
                    dataset = yaml.safe_load(file)
                target_path, target_list = '', dataset['Files'][target]
                ref_path, ref_list = '', dataset['Files'][ref]
                self.multi_setup = True if len(dataset['Setup']) > 1 else False
            else:
                target_path, ref_path = None, None
            try:
                self.experiment[p]['occlusion_path'], _, occlusion_list = (
                    os.walk(f'{P}/occlusion/{ref}_to_{target}').__next__())
                self.experiment[p]['occlusion_mask'] = sorted(occlusion_list)
                self.experiment[p]['occlusion_ok'] = True
            except StopIteration:
                self.experiment[p]['occlusion_ok'] = False
                print(f'Occlusion masks wont be available for the {p} couple')
            if target_path is not None:
                self.experiment[p]['target_list'] = [target_path + '/' + n for n in target_list]
                self.experiment[p]['ref_list'] = [ref_path + '/' + n for n in ref_list]
                self.experiment[p]['inputs_available'] = True

            else:
                self.experiment[p]['target_list'], self.experiment[p]['ref_list'] = None, None
                print(f'Inputs images wont be available for experiment {p}')
                self.experiment[p]['inputs_available'] = False
            self.experiment[p]['new_list'] = [new_path + '/' + n for n in sorted(new_list)]
            if self.multi_setup:
                self.dx_max, self.dy_max, self.dz_max, self.da_max, self.f_max, self.px_max = 0, 0, 0, 0, 0, 0
                self.dx, self.dy, self.dz, self.da, self.f, self.px = 0, 0, 0, 0, 0, 0
                self.define_delta(self.experiment[p]['new_list'][-1], init_max=True)
            if os.path.exists(f'{P}/Summary_experiment.yaml'):
                with open(f'{P}/Summary_experiment.yaml', "r") as file:
                    summary = yaml.safe_load(file)
                warp_inverted = summary['Wrap']['reverse']
            else:
                warp_inverted = False
            try:
                self.experiment[p]['target_depth_path'], _, target_depth_list = os.walk(
                    f'{P}/pred_depth/{target if not warp_inverted else ref}').__next__()
                self.experiment[p]['ref_depth_path'], _, ref_depth_list = os.walk(
                    f'{P}/depth_reg/{ref if not warp_inverted else target}').__next__()
                self.experiment[p]['target_depth_list'] = sorted(target_depth_list)
                self.experiment[p]['ref_depth_list'] = sorted(ref_depth_list)
                self.experiment[p]['depth_ok'] = True
            except StopIteration:
                self.experiment[p]['depth_ok'] = False
                print(f'Depth images wont be available for the {p} couple')

            if os.path.exists(f'{P}/Validation.yaml'):
                self.experiment[p]['validation_available'] = True
                with open(f'{P}/Validation.yaml', "r") as file:
                    self.experiment[p]['val'] = yaml.safe_load(file)
            else:
                self.experiment[p]['validation_available'] = False
            self.experiment[p]['idx_max'] = len(new_list)
            self.experiment[p]['videoWritter'] = VideoGenerator(30, P)
        self.device = get_cuda_device_if_available()
        self.tensor = True
        self.idx = 0
        self.show_disp_overlay = 0
        self.video_array = []

    def run(self):
        exp = self.exp_list[0]
        exp_idx = 0
        # experiment = self.experiment[exp]
        while self.key != 27:
            visu = self._create_visual(exp)
            if self.key == ord('w') or len(self.video_array) > 0:
                visu = self._video_creation(exp, visu)

            cv.imshow(f'Experience {exp}', visu)
            cv.setWindowProperty(f'Experience {exp}', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            cv.setWindowProperty(f'Experience {exp}', cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
            self.key = cv.waitKey(0)

            self._execute_cmd(exp)
            if self.key == ord('\t'):
                exp_idx += 1
                exp_idx = exp_idx % len(self.exp_list)
                exp = self.exp_list[exp_idx]
                experiment = self.experiment[exp]
                cv.setWindowProperty(f'Experience {exp}', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
                cv.setWindowProperty(f'Experience {exp}', cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
                if self.show_disp_overlay > 1:
                    self.show_disp_overlay = 0
            self.idx = self.idx % self.experiment[exp]['idx_max']
        cv.destroyAllWindows()

    def define_delta(self, name, init_max=False):
        name = name.split('_')
        if init_max:
            self.dx_max, self.dy_max, self.dz_max, self.da_max, self.f_max, self.px_max = 0, 0, 0, 0, 0, 0
            for i, mot in enumerate(name):
                if mot == 'dx':
                    self.dx_max = int(name[i + 1]) + 1
                elif mot == 'dy':
                    self.dy_max = int(name[i + 1]) + 1
                elif mot == 'dz':
                    self.dz_max = int(name[i + 1]) + 1
                elif mot == 'da':
                    self.da_max = int(name[i + 1]) + 1
                elif mot == 'f':
                    self.f_max = int(name[i + 1]) + 1
                elif mot == 'px':
                    self.px_max = int(name[i + 1]) + 1
        else:
            self.dx, self.dy, self.dz, self.da, self.f, self.px = 0, 0, 0, 0, 0, 0
            for i, mot in enumerate(name):
                if mot == 'dx':
                    self.dx = int(name[i + 1])
                elif mot == 'dy':
                    self.dy = int(name[i + 1])
                elif mot == 'dz':
                    self.dz = int(name[i + 1])
                elif mot == 'da':
                    self.da = int(name[i + 1])
                elif mot == 'f':
                    self.f = int(name[i + 1])
                elif mot == 'px':
                    self.px = int(name[i + 1])

    def _execute_cmd(self, exp):
        i = 0
        experiment = self.experiment[exp]
        while self.key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'),
                           ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:  # selection from idx + enter
            i = i * 10 + int(chr(self.key))
            self.key = cv.waitKey(0)
            if self.key == ord('\r'):
                self.idx = i
                self.key = -1
        if self.key == ord('-'):
            self.idx -= 1
        if self.key == ord('+'):
            self.idx += 1
        if self.key == ord('d'):
            self.show_disp_overlay += 1
            if self.show_disp_overlay == 1:
                if not experiment['depth_ok']:
                    self.show_disp_overlay = 0
            if self.show_disp_overlay > 1:
                self.show_disp_overlay = 0
        if self.key == ord('i'):
            self.show_idx = not self.show_idx
        if self.key == ord('v'):
            self.show_validation = not self.show_validation
        if self.key == ord('g'):
            self.show_grad_im += 1
            if self.show_grad_im > 2:
                self.show_grad_im = 0
        if self.key == ord('t') and self.device:
            self.tensor = not self.tensor
        if self.key == ord('o') and experiment['occlusion_ok']:
            self.show_occlusion = not self.show_occlusion
        if self.multi_setup:
            if self.key == ord('x'):
                self.dx = (self.dx + 1) % max(self.dx_max, 1)
                self.idx = (self.dx * self.dy_max * self.dz_max * self.da_max +
                            self.dy * self.dz_max * self.da_max +
                            self.dz * self.da_max + self.da +
                            self.f * self.px_max + self.px)
            elif self.key == ord('y'):
                self.dy = (self.dy + 1) % max(self.dy_max, 1)
                self.idx = (self.dx * self.dy_max * self.dz_max * self.da_max +
                            self.dy * self.dz_max * self.da_max +
                            self.dz * self.da_max + self.da +
                            self.f * self.px_max + self.px)
            elif self.key == ord('z'):
                self.dz = (self.dz + 1) % max(self.dz_max, 1)
                self.idx = (self.dx * self.dy_max * self.dz_max * self.da_max +
                            self.dy * self.dz_max * self.da_max +
                            self.dz * self.da_max + self.da +
                            self.f * self.px_max + self.px)
            elif self.key == ord('a'):
                self.da = (self.da + 1) % max(self.da_max, 1)
                self.idx = (self.dx * self.dy_max * self.dz_max * self.da_max +
                            self.dy * self.dz_max * self.da_max +
                            self.dz * self.da_max + self.da +
                            self.f * self.px_max + self.px)
            elif self.key == ord('f'):
                self.f = (self.f + 1) % max(self.f_max, 1)
                self.idx = (self.dx * self.dy_max * self.dz_max * self.da_max +
                            self.dy * self.dz_max * self.da_max +
                            self.dz * self.da_max + self.da +
                            self.f * self.px_max + self.px)
            elif self.key == ord('p'):
                self.px = (self.px + 1) % max(self.px_max, 1)
                self.idx = (self.dx * self.dy_max * self.dz_max * self.da_max +
                            self.dy * self.dz_max * self.da_max +
                            self.dz * self.da_max + self.da +
                            self.f * self.px_max + self.px)

    def _create_visual(self, exp):
        experiment = self.experiment[exp]
        new_im = ImageTensor(f'{experiment["new_list"][self.idx]}').RGB()
        if experiment["inputs_available"]:
            if self.multi_setup:
                self.define_delta(experiment["new_list"][self.idx])
            target_im = ImageTensor(f'{experiment["target_list"][self.idx]}').RGB()
            ref_im = ImageTensor(f'{experiment["ref_list"][self.idx]}').RGB().match_shape(target_im, keep_ratio=True)
            new_im = new_im.match_shape(target_im, keep_ratio=True)
        else:
            target_im = new_im.clone()
            ref_im = new_im.clone()
        if self.show_occlusion:
            mask = 1 - ImageTensor(
                f'{experiment["occlusion_path"]}/{experiment["occlusion_mask"][self.idx]}').match_shape(
                target_im, keep_ratio=True)
        else:
            mask = 1
        visu = (target_im / 2 + new_im * mask / 2).vstack(target_im / 2 + ref_im / 2)

        if self.show_grad_im:
            grad_im = self._create_grad_im(new_im, ref_im, target_im, mask, self.show_grad_im - 1)
            visu = visu.hstack(grad_im)

        if self.show_disp_overlay:
            depth_overlay = self._create_depth_overlay(experiment, ref_im, target_im, mask)
            visu = visu.hstack(depth_overlay)

        h, w = ref_im.shape[-2:]

        while visu.shape[3] > 1920 or visu.shape[2] > 1080:
            visu = visu.pyrDown()
            h, w = h // 2, w // 2

        visu = visu.opencv()
        if self.show_idx:
            visu = cv.putText(visu, f'idx : {self.idx}', self.org_idx, self.font, self.fontScale, self.color,
                              self.thickness, cv.LINE_AA)
        if self.multi_setup:
            org = self.org_idx[0], self.org_idx[1] + 15
            visu = cv.putText(visu, f'{f"dx : {self.dx}," if self.dx_max > 0 else ""}'
                                    f'{f"dy : {self.dy}," if self.dy_max > 0 else ""}'
                                    f'{f"dz : {self.dz}," if self.dz_max > 0 else ""}'
                                    f'{f"da : {self.da}," if self.da_max > 0 else ""}'
                                    f'{f"df : {self.f}," if self.f_max > 0 else ""}'
                                    f'{f"dpx : {self.px}," if self.px_max > 0 else ""}',
                              org, self.font, self.fontScale, self.color, self.thickness, cv.LINE_AA)
        if self.show_grad_im:
            org = self.org_idx[0] + w, self.org_idx[1]
            if self.show_grad_im == 1 or not self.tensor:
                text = f'Image grad : {"with tensor" if self.tensor else "with numpy"}'
            else:
                text = f'Image SSIM'
            visu = cv.putText(visu, text, org,
                              self.font,
                              self.fontScale, self.color,
                              self.thickness, cv.LINE_AA)

        if self.show_validation and experiment['validation_available']:
            org_val = 10, visu.shape[0] - 65
            for key, value in experiment['val']['2. results'].items():
                if key in exp:
                    occ = 'new_occ' if self.show_occlusion else 'new'
                    for key_stat, stat in value.items():
                        stats = f'{key_stat} : {stat[occ][self.idx]} / {stat["ref"][self.idx]}'
                        if key_stat == 'rmse':
                            color_val = (0, 0, 255) if stat[occ][self.idx] >= stat['ref'][self.idx] else (0, 255, 0)
                        else:
                            color_val = (0, 255, 0) if stat[occ][self.idx] >= stat['ref'][self.idx] else (0, 0, 255)
                        visu = cv.putText(visu, stats, org_val, self.font, self.fontScale, color_val,
                                          self.thickness,
                                          cv.LINE_AA)
                        org_val = org_val[0], org_val[1] + 15
        return visu

    def _video_creation(self, exp, visu):
        experiment = self.experiment[exp]
        if self.key == ord('w'):
            self.video_array.append(self.idx)
        if len(self.video_array) > 2:
            self.video_array = self.video_array[2:]
        self.video_array = sorted(self.video_array)
        delta = 20 if self.multi_setup else 0
        org = self.org_idx[0], self.org_idx[1] + 20 + delta
        visu = cv.putText(visu, f'Starting video frame : {self.video_array[0]}', org, self.font, self.fontScale,
                          self.color,
                          self.thickness, cv.LINE_AA)
        if len(self.video_array) == 2:
            org = self.org_idx[0], self.org_idx[1] + 40 + delta
            visu = cv.putText(visu, f'Ending video frame : {self.video_array[1]}', org, self.font,
                              self.fontScale, self.color,
                              self.thickness, cv.LINE_AA)
            org = self.org_idx[0], self.org_idx[1] + 60 + delta
            visu = cv.putText(visu, f'Choose your format and press "Enter"', org, self.font,
                              self.fontScale, self.color,
                              self.thickness, cv.LINE_AA)
            if self.key == ord('\r'):
                frames_idx = np.arange(self.video_array[0], self.video_array[1] + 1)
                visu = self._create_visual(exp)
                for idx in tqdm(frames_idx, total=len(frames_idx), desc=f"Frames encoding..."):
                    self.idx = idx
                    visu = self._create_visual(exp)
                    experiment['videoWritter'].append(visu)
                name = input('Enter the Name of the video')
                experiment['videoWritter'].write(name)
                self.key = -1
                self.video_array = []
        return visu

    def _create_grad_im(self, new_im, ref_im, target_im, mask, idx=0):
        metrics = [Metric_nec_tensor, Metric_ssim_tensor]
        if self.tensor:
            metric = metrics[idx](self.device)
            new_target = metric(new_im, target_im, return_image=True).match_shape(new_im)
            grad_ref_target = metric(ref_im, target_im, return_image=True).match_shape(new_im)
            im = new_target.vstack(grad_ref_target)
        else:
            grad_new = grad_image(new_im)
            grad_ref = grad_image(ref_im)
            grad_target = grad_image(target_im)
            im = (grad_new * mask / 2 + grad_target * mask / 2).vstack(grad_ref / 2 + grad_target / 2)
        return im

    def _create_depth_overlay(self, experiment, ref_im, target_im, mask):
        depth_target = ImageTensor(DepthTensor(ImageTensor(
            f'{experiment["target_depth_path"]}/{experiment["target_depth_list"][self.idx]}')).inverse_depth()).RGB()
        depth_ref = ImageTensor(DepthTensor(ImageTensor(f'{experiment["ref_depth_path"]}/{experiment["ref_depth_list"][self.idx]}')).inverse_depth()).RGB()
        depth_overlay_ref = depth_ref.match_shape(ref_im)
        depth_overlay_target = depth_target.match_shape(ref_im) * mask
        return (depth_overlay_ref / depth_overlay_ref.max()).vstack(depth_overlay_target / depth_overlay_target.max())

    # def _write_video(self):


if __name__ == '__main__':
    pro = '/home/godeta/PycharmProjects/Disparity_Pipeline/results/'
    perso = '/home/aurelien/PycharmProjects/Disparity_Pipeline/results/'
    p = pro if 'godeta' in os.getcwd() else perso
    path = p + "/Dataset_Lynred/"
    # path = p + "/Test/Disparity-Depth_night"
    Visualizer(path, search_exp=True).run()
