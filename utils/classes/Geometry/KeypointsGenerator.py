import torch
from kornia.feature import SIFTFeatureScaleSpace, SIFTFeature, GFTTAffNetHardNet, match_adalam, match_fginn, match_smnn, \
    match_snn, match_mnn, match_nn, get_laf_center
from torch import Tensor
from utils.classes import ImageTensor
import cv2 as cv
import numpy as np


class KeypointsGenerator:
    DETECTOR = {'SIFT_SCALE': SIFTFeatureScaleSpace, 'SIFT': SIFTFeature, 'DISK': GFTTAffNetHardNet}

    def __init__(self, device: torch.device, detector: str = 'sift_scale', matcher: str = 'snn',
                 num_feature=8000, th=0.8, spatial_th=10.0, mutual=False):
        detector = detector.upper()
        matcher = matcher.upper()
        assert detector in self.DETECTOR.keys()
        self.device = device
        self.detector = self.DETECTOR[detector](num_features=num_feature, device=device)
        self.detector_name = self.detector.__class__.__name__
        self.matcher = DescriptorMatcher(matcher=matcher, th=th, spatial_th=spatial_th, mutual=mutual)
        self.matcher_name = self.matcher.name

    @torch.no_grad()
    def __call__(self, img_src: ImageTensor, img_dst: ImageTensor, *args,
                 method: str = 'auto', pts_ref=None, min_kpt=-1, th=0, draw_result=False, draw_result_inplace=False,
                 max_drawn=200,
                 **kwargs):
        if method == 'manual' or method == 'm':
            if pts_ref is not None:
                pts_ref = pts_ref if len(pts_ref) <= 20 else pts_ref[np.randint(0, len(pts_ref), size=20)]
            return self.manual_keypoints_selection(img_src, img_dst, pts_ref=pts_ref, nb_point=min_kpt)
        else:
            laf_src, r_func_src, desc_src = self.detector(Tensor(img_src.GRAYSCALE()))
            laf_dst, r_func_dst, desc_dst = self.detector(Tensor(img_dst.GRAYSCALE()))
            m_distance, index_tensor = self.matcher(desc_src, desc_dst, laf_src, laf_dst)
            if th == 0 and min_kpt == -1:
                idx_src, idx_dst = index_tensor.T
            elif th > 0 and min_kpt > 0:
                idx_src, idx_dst = index_tensor[m_distance.squeeze() > th].T
                STOP = 0
                while len(idx_src) < min_kpt and not STOP:
                    th -= 0.01
                    idx_src, idx_dst = index_tensor[m_distance.squeeze() > th].T
                    STOP = 1 if th <= 0 else 0
                    if STOP:
                        print(f'The number of keypoints required was not reached...{len(idx_src)}/{min_kpt}')
            elif min_kpt > 0:
                th, STOP = 1, 0
                idx_src, idx_dst = index_tensor[m_distance.squeeze() > th].T
                while len(idx_src) < min_kpt and not STOP:
                    th -= 0.01
                    STOP = 1 if th <= 0 else 0
                    idx_src, idx_dst = index_tensor[m_distance.squeeze() > th].T
                if STOP:
                    print(f'The number of keypoints required was not reached...{len(idx_src)}/{min_kpt}')
            elif 1 > th >= 0:
                idx_src, idx_dst = index_tensor[m_distance.squeeze() > th].T
            else:
                print(f'The given value of th and min_kpt are not meaningful, the keypoints won"t be sorted')
                idx_src, idx_dst = index_tensor.T
            keypoints_src = get_laf_center(laf_src[:, idx_src, ...])
            keypoints_dst = get_laf_center(laf_dst[:, idx_dst, ...])

            if draw_result:
                self.draw_keypoints(img_src, keypoints_src, img_dst, keypoints_dst, max_drawn=max_drawn)
            if draw_result_inplace:
                self.draw_keypoints_inplace(img_src, img_dst, keypoints_src, keypoints_dst, max_drawn=max_drawn)
            return keypoints_src, keypoints_dst

    def draw_keypoints(self, img_src, keypoints_src, img_dst, keypoints_dst, max_drawn=200):
        if img_src.im_type == 'RGB':
            img_src_ = img_src.opencv()
        else:
            img_src_ = img_src.RGB(cmap='gray').opencv()
        if img_dst.im_type == 'RGB':
            img_dst_ = img_dst.opencv()
        else:
            img_dst_ = img_dst.RGB(cmap='gray').opencv()
        keypoints_src_ = keypoints_src.squeeze().cpu().numpy()
        keypoints_dst_ = keypoints_dst.squeeze().cpu().numpy()
        draw_params = dict(matchColor=-1,  # draw matches in random colors
                           singlePointColor=(0, 0, 255),  # draw single points in red color
                           matchesMask=None,  # draw only inliers
                           flags=2)
        max_drawn = min(max_drawn, keypoints_src_.shape[0])
        keypoints_src_ = tuple([cv.KeyPoint(*kpts_src, 1) for kpts_src in keypoints_src_[:max_drawn]])
        keypoints_dst_ = tuple([cv.KeyPoint(*kpts_dst, 1) for kpts_dst in keypoints_dst_[:max_drawn]])
        crsp = tuple([cv.DMatch(i, i, 0) for i in range(len(keypoints_dst_))])
        img_ = ImageTensor(
            cv.drawMatches(img_dst_, keypoints_dst_, img_src_, keypoints_src_, crsp, None, **draw_params)[
                ..., [2, 1, 0]])
        name = f'Detector : {self.detector_name} & Matcher : {self.matcher_name}'
        img_.show(num=name)

    def draw_keypoints_inplace(self, img_src, img_dst, keypoints_src, keypoints_dst, max_drawn=200):
        if img_src.im_type == 'RGB':
            img_src_ = img_src.opencv()
        else:
            img_src_ = img_src.RGB(cmap='gray').opencv()
        if img_dst.im_type == 'RGB':
            img_dst_ = img_dst.opencv()
        else:
            img_dst_ = img_dst.RGB(cmap='gray').opencv()
        img = np.uint8(img_dst_ / 2 + img_src_ / 2)
        keypoints_src_ = keypoints_src.squeeze().cpu().numpy()
        keypoints_dst_ = keypoints_dst.squeeze().cpu().numpy()
        max_drawn = min(max_drawn, keypoints_src_.shape[0])
        keypoints_ = [(cv.KeyPoint(*kpts_src, 1), cv.KeyPoint(*kpts_dst, 1)) for kpts_src, kpts_dst in
                      zip(keypoints_src_[:max_drawn], keypoints_dst_[:max_drawn])]
        for kpts in keypoints_:
            color = tuple(np.round(np.random.rand(3, ) * 255, 0).tolist())
            cv.drawKeypoints(img, kpts, img, color=color)
            img = cv.line(img, (int(kpts[0].pt[0]), int(kpts[0].pt[1])),
                               (int(kpts[1].pt[0]), int(kpts[1].pt[1])), color=color, thickness=2)
        name = f'Detector : {self.detector_name} & Matcher : {self.matcher_name}'
        ImageTensor(img[..., [2, 1, 0]]).show(num=name)

    @staticmethod
    def manual_keypoints_selection(im_src: ImageTensor, im_dst: ImageTensor, pts_ref=None, nb_point=50) -> tuple:
        global im_temp, pts_temp, rightclick

        def mouseHandler(event, x, y, flags, param):
            global pts_temp, rightclick
            if event == cv.EVENT_LBUTTONDOWN:
                cv.circle(im_temp, (x, y), 2, (0, 255, 255), 2, cv.LINE_AA)
                cv.putText(im_temp, str(len(pts_temp) + 1), (x + 3, y + 3),
                           cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 200, 250), 1, cv.LINE_AA)
                if len(pts_temp) < nb_point:
                    pts_temp = np.append(pts_temp, [(x, y)], axis=0)
                if rightclick == 0:
                    cv.imshow('Ref selection window', im_temp)
                elif rightclick == 1:
                    cv.imshow('Image to position selection window', im_temp)
            if (event == cv.EVENT_RBUTTONDOWN or event == cv.EVENT_MOUSEWHEEL) and len(pts_temp) >= 8:
                rightclick = 2
                print('end of selection')
            # Image REF

        if im_src.im_type == 'RGB':
            im_src_ = np.ascontiguousarray(im_src.opencv(), dtype=np.uint8)
        else:
            im_src_ = np.ascontiguousarray(im_src.RGB(cmap='gray').opencv(), dtype=np.uint8)
        if im_dst.im_type == 'RGB':
            im_dst_ = np.ascontiguousarray(im_dst.opencv(), dtype=np.uint8)
        else:
            im_dst_ = np.ascontiguousarray(im_dst.RGB(cmap='gray').opencv(), dtype=np.uint8)

        # Vector temp
        pts_temp = np.empty((0, 2), dtype=np.int32)
        # Create a window
        im_temp = im_src_.copy()
        cv.namedWindow('Image to position selection window')
        cv.namedWindow('Ref selection window')

        if pts_ref is not None:
            try:
                assert isinstance(pts_ref, Tensor)
                pts_ref = pts_ref.squeeze().cpu().numpy()
                assert len(pts_ref) >= nb_point
            except AssertionError:
                rightclick = 0
            rightclick = 1
            for idx, (x, y) in enumerate(pts_ref):
                cv.circle(im_temp, (int(x), int(y)), 2, (0, 255, 255), 2, cv.LINE_AA)
                cv.putText(im_temp, str(idx + 1), (int(x) + 3, int(y) + 3),
                           cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (200, 200, 250), 1, cv.LINE_AA)
            pts_src = pts_ref
        else:
            rightclick = 0

        if rightclick == 0:
            cv.imshow('Image to position selection window', im_dst_)
            cv.imshow('Ref selection window', im_temp)
            cv.setMouseCallback('Ref selection window', mouseHandler)
            while True:
                print(len(pts_temp), nb_point)
                if len(pts_temp) >= nb_point or (cv.waitKey(10) == 27 or rightclick == 2):
                    rightclick = 1
                    break
            pts_src = pts_temp.copy()
        if rightclick == 1:
            # Destination image
            im_src_ = im_temp.copy()
            im_temp = im_dst_.copy()
            pts_temp = np.empty((0, 2), dtype=np.int32)
            # Create a window
            cv.imshow('Ref selection window', im_src_)
            cv.imshow('Image to position selection window', im_temp)
            cv.setMouseCallback('Image to position selection window', mouseHandler)
            while True:
                if len(pts_temp) == len(pts_src) or cv.waitKey(10) == 27:
                    break
        cv.destroyAllWindows()
        pts_dst = pts_temp
        # if len(pts_ref) >= 4:
        #     tform, _ = cv.findHomography(pts_src, pts_dst)
        # else:
        #     tform = None
        del im_temp, pts_temp, rightclick
        return Tensor(pts_src).to(dtype=torch.float32).unsqueeze(0), \
            Tensor(pts_dst).to(dtype=torch.float32).unsqueeze(0)


class DescriptorMatcher:
    MATCHER = {'NN': match_nn, 'MNN': match_mnn, 'SNN': match_snn, 'SMNN': match_smnn, 'FGINN': match_fginn,
               'ADALAM': match_adalam}

    def __init__(self, matcher='snn', th=0.8, spatial_th=10.0, mutual=False):
        assert matcher in self.MATCHER.keys()
        self.matcher = self.MATCHER[matcher]
        self.name = matcher
        self.th = th
        self.spatial_th = spatial_th
        self.mutual = mutual

    def __call__(self, desc1, desc2, *args, **kwargs):
        desc1, desc2 = desc1.squeeze(), desc2.squeeze()
        if self.name == 'NN' or self.name == 'MNN':
            m_distance, index_tensor = self.matcher(desc1, desc2)
            m_distance = (m_distance - m_distance.max()) / (m_distance.min() - m_distance.max())
            return m_distance, index_tensor
        elif self.name == 'SNN' or self.name == 'SMNN':
            m_distance, index_tensor = self.matcher(desc1, desc2, th=self.th)
            m_distance = (m_distance - m_distance.min()) / (m_distance.max() - m_distance.min())
            return m_distance, index_tensor
        elif self.name == 'FGINN':
            try:
                lafs1, lafs2 = args[0], args[1]
                m_distance, index_tensor = self.matcher(desc1, desc2, lafs1, lafs2, th=self.th,
                                                        spatial_th=self.spatial_th,
                                                        mutual=self.mutual)
                m_distance = (m_distance - m_distance.min()) / (m_distance.max() - m_distance.min())
                return m_distance, index_tensor
            except IndexError:
                print(f'The LAFs are required for the {self.name} matcher')
        else:
            try:
                lafs1, lafs2 = args[0], args[1]
                m_distance, index_tensor = self.matcher(desc1, desc2, lafs1, lafs2)
                m_distance = (m_distance - m_distance.min()) / (m_distance.max() - m_distance.min())
                return m_distance, index_tensor
            except IndexError:
                print(f'The LAFs are required for the {self.name} matcher')
