
BORDER_FLAG = {"BORDER_CONSTANT": 0,
               "BORDER_REPLICATE": 1,
               "BORDER_REFLECT": 2,
               "BORDER_WRAP": 3,
               "BORDER_REFLECT_101": 4,
               "BORDER_TRANSPARENT": 5,
               "BORDER_ISOLATED": 16}

INTERPOLATION_FLAG = {"INTER_NEAREST": 0,
                      "INTER_LINEAR": 1,
                      "INTER_CUBIC": 2,
                      "INTER_AREA": 3,
                      "INTER_LANCZOS4": 4,
                      "INTER_LINEAR_EXACT": 5,
                      "INTER_NEAREST_EXACT": 6,
                      "INTER_MAX": 7,
                      "WARP_FILL_OUTLIERS": 8,
                      "WARP_INVERSE_MAP": 16}


# The projection process treat each case like : input left, input right, 0/1 is the index of the projected image
Projection_process = {0: ['left', 'right', 1, None],              # L-Vis--R, L-VIS--R, Vis-L---R, VIS-L---R
                      1: ['left', 'right', 0, None],              # L--Vis-R, L--VIS-R, L---R-Vis, L---R-VIS
                      98: ['left', 'right', 1, None],             # ?
                      10: ['other', 'left', 0, "Vis-L---R"],             # Vis-L---R
                      11: ['other', 'left', 1, "VIS-L---R"],             # VIS-L---R
                      12: ['left', 'other', 0, "L-VIS---R"],             # L-VIS---R
                      13: ['left', 'other', 1, "L-Vis---R"],             # L-Vis---R
                      14: ['other', 'right', 0, "L---Vis-R"],            # L---Vis-R
                      15: ['other', 'right', 1, "L---VIS-R"],            # L---VIS-R
                      16: ['right', 'other', 0, "L---R-VIS"],            # L---R-VIS
                      17: ['right', 'other', 1, "L---R-Vis"],            # L---R-Vis
                      99: ['left', 'other', 1, None]}             # ?
