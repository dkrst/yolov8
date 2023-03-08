from pathlib import Path
import shutil
from tqdm import tqdm
import cv2

MAT_FORMATS = 'npy'  # include matrix format for multichannel images
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm' # include image suffixes

def extract_boxes(path='.'):
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classification') if (path / 'classification').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            im = cv2.imread(str(im_file))[..., ::-1]
            print(im_file)
            print(im.shape)
