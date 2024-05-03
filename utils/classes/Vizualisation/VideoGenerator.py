import os.path
from typing import Union

import cv2 as cv
from tqdm import tqdm


class VideoGenerator(list):
    """
    A simple class to generate video from a list of image
    """
    def __init__(self, fps: int, path: str):
        super().__init__()
        self.framerate = fps
        self.path = path
        self.size = (0, 0)
        self.format = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

    def append(self, obj):
        assert obj is not None
        if not isinstance(obj, list):
            obj = [obj]
        for o in obj:
            if isinstance(o, str):
                assert os.path.exists(o)
                assert os.path.isfile(o)
                frame = cv.imread(o, 1)
            else:
                frame = o
            if self.size == (0, 0):
                self.size = frame.shape[:2][1], frame.shape[:2][0]
            else:
                frame = cv.resize(frame, self.size)
            super().append(frame)

    def write(self, namevideo):
        path = self.path + f'/{namevideo}.mp4'
        assert len(self) > 0
        out = cv.VideoWriter(path, self.format, self.framerate, self.size)
        print(f"\n  Video writing...")
        for frame in tqdm(self):
            out.write(frame)
        out.release()
        self.clear()

