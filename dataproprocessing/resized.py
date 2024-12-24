# #!/usr/bin/python
#adjusted from 'https://github.com/nyuad-cai/MedFuse/blob/main/resize.py'
# import thread
import time
from PIL import Image
import glob
from tqdm import tqdm
import os

print('starting')

data_dir ='/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/files'

# paths_done = glob.glob(f'{data_dir}/{version}/resized/**/*.jpg', recursive = True)
# print('done', len(paths_done))
paths_done = glob.glob(f'/home/mimic/MIMIC_subset/MIMIC_subset/resized/*.jpg', recursive = True)
print('done', len(paths_done))

# paths_all = glob.glob(f'{data_dir}/*.jpg', recursive = True)
# # print(paths_all)
# print('all', len(paths_all))


paths_all = glob.glob(f'{data_dir}/**/*.jpg', recursive = True)
print('all', len(paths_all))

done_files = [os.path.basename(path) for path in paths_done]
# print(done_files)
paths = [path for path in paths_all if os.path.basename(path) not in done_files ]
print('left', len(paths))

def resize_images(path):
    try:
        basewidth = 512
        filename = path.split('/')[-1]
        img = Image.open(path)

        wpercent = (basewidth/float(img.size[0]))
        
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize))
        print(f'')
        
        img.save(f'/home/mimic/MIMIC_subset/MIMIC_subset/resized/{filename}')

    except OSError as e:
        print(f"Error processing file {path}: {e}")



from multiprocessing.dummy import Pool as ThreadPool

threads = 10

for i in tqdm(range(0, len(paths), threads)):
    paths_subset = paths[i: i+threads]
    pool = ThreadPool(len(paths_subset))
    pool.map(resize_images, paths_subset)
    pool.close()
    pool.join()
