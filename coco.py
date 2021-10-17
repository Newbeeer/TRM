import os, sys, time, io, subprocess, requests
import numpy as np
import random

from PIL import Image
from pycocotools.coco import COCO
from skimage.transform import resize

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import h5py
################ Paths and other configs - Set these #################################
CLASSES = [
        'boat',
        'airplane',
        'truck',
        'dog',
        'zebra',
        'horse',
        'bird',
        'train',
        'bus',
        'motorcycle'
        ]


output_dir = os.path.join('../data', 'coco')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

NUM_CLASSES = len(CLASSES)
ANOMALY = 0

h5pyfname = output_dir
print(h5pyfname,os.path.exists(h5pyfname))
if not os.path.exists(h5pyfname):
    os.makedirs(h5pyfname)
# we dont use this function below
def getClassName(cID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == cID:
            return cats[i]['name']
    return 'None'

###########################################################################################
biased_colours = [[0,100,0],
                  [188, 143, 143],
                  [255, 0, 0],
                  [255, 215, 0],
                  [0, 255, 0],
                  [65, 105, 225],
                  [0, 225, 225],
                  [0, 0, 255],
                  [255, 20, 147]]
biased_colours = np.array(biased_colours)

_D = 2500
def random_different_enough_colour():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D:
            break
    return list(x)

# unbiased_color: distance at least 50 from biased color
unbiased_colours = np.array([random_different_enough_colour() for _ in range(10)])

def test_colours():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D and np.min(np.sum((x - unbiased_colours)**2, 1)) > _D:
            break
    return x
test_unbiased_colours = np.array([test_colours() for _ in range(10)])

def validation_colours():
    while True:
        x = np.random.choice(255, size=3)
        if np.min(np.sum((x - biased_colours)**2, 1)) > _D and np.min(np.sum((x - unbiased_colours)**2, 1)) > _D and np.min(np.sum((x - test_unbiased_colours)**2, 1)) > _D:
            break
    return x
validation_unbiased_colours = np.array([validation_colours() for _ in range(10)])

###########################################################################################

######################################################################################

tr_i = 800*NUM_CLASSES
val_i = 100*NUM_CLASSES
te_i = 100*NUM_CLASSES

train_fname = os.path.join(h5pyfname,'train.h5py')

val_id_fname = os.path.join(h5pyfname,'validtest.h5py')
val_ood_fname = os.path.join(h5pyfname,'valoodtest.h5py')
val_sg_fname = os.path.join(h5pyfname,'valsgtest.h5py')

id_fname = os.path.join(h5pyfname,'idtest.h5py')
sg_fname = os.path.join(h5pyfname,'sgtest.h5py')
ood_fname =os.path.join( h5pyfname,'oodtest.h5py')

ano_fname =os.path.join( h5pyfname,'anotest.h5py')

# if os.path.exists(train_fname): subprocess.call(['rm', train_fname])
# if os.path.exists(val_id_fname): subprocess.call(['rm', val_id_fname])
# if os.path.exists(val_ood_fname): subprocess.call(['rm', val_ood_fname])
# if os.path.exists(val_sg_fname): subprocess.call(['rm', val_sg_fname])
# if os.path.exists(id_fname): subprocess.call(['rm', id_fname])
# if os.path.exists(sg_fname): subprocess.call(['rm', sg_fname])
# if os.path.exists(ood_fname): subprocess.call(['rm', ood_fname])
# if os.path.exists(ano_fname): subprocess.call(['rm', ano_fname])

train_file = h5py.File(train_fname, mode='w')
val_id_file = h5py.File(val_id_fname, mode='w')
id_test_file = h5py.File(id_fname, mode='w')

# train_file.create_dataset('images', (tr_i,3,64,64), dtype=np.dtype('float32'))
# val_id_file.create_dataset('images', (val_i,3,64,64), dtype=np.dtype('float32'))
train_file.create_dataset('resized_images', (tr_i,3,64,64), dtype=np.dtype('float32'))
val_id_file.create_dataset('resized_images', (val_i,3,64,64), dtype=np.dtype('float32'))
id_test_file.create_dataset('resized_images', (te_i,3,64,64), dtype=np.dtype('float32'))
train_file.create_dataset('resized_mask', (tr_i,3,64,64), dtype=np.dtype('float32'))
val_id_file.create_dataset('resized_mask', (val_i,3,64,64), dtype=np.dtype('float32'))
id_test_file.create_dataset('resized_mask', (te_i,3,64,64), dtype=np.dtype('float32'))
# g stands for 'group'
train_file.create_dataset('y', (tr_i,), dtype='int32')
val_id_file.create_dataset('y', (val_i,), dtype='int32')
id_test_file.create_dataset('y', (te_i,), dtype='int32')


coco = COCO('../data/coco/annotations/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())


tr_s, val_s, te_s = 0, 0, 0
for c in range(NUM_CLASSES):
    catIds = coco.getCatIds(catNms=[CLASSES[c]])
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)

    i = -1
    tr_si = 0
    print('Class {} (train) : #images = {}'.format(c, len(images)))
    while tr_si < tr_i//NUM_CLASSES:
        i += 1

        # get the image
        im = images[i]
        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']

        if max_ann < 10000: continue;
        img_path = os.path.join('../data/coco/train2017', im['file_name'])
        I = np.asarray(Image.open(img_path))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        # that's the one:
        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)
        resized_image = resize(I, (64, 64), anti_aliasing=True)
        train_file['resized_images'][tr_s, ...] = np.transpose(resized_image, (2,0,1))
        train_file['resized_mask'][tr_s, ...] = np.transpose(resized_mask, (2, 0, 1))
        train_file['y'][tr_s] = c

        tr_s += 1
        tr_si += 1
        if tr_si % 100 == 0:
            print('>'.format(c), end='')
            time.sleep(1)
    print(' ')

    val_si = 0
    while val_si < val_i//NUM_CLASSES:
        i += 1

        # get the image
        im = images[i]

        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']

        if max_ann < 10000: continue;

        img_path = os.path.join('../data/coco/train2017', im['file_name'])
        I = np.asarray(Image.open(img_path))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)
        resized_image = resize(I, (64, 64), anti_aliasing=True)

        # val_id:
        val_id_file['resized_images'][val_s, ...] = np.transpose(resized_image, (2,0,1))
        val_id_file['resized_mask'][val_s, ...] = np.transpose(resized_mask, (2, 0, 1))
        val_id_file['y'][val_s] = c

        val_s += 1
        val_si += 1
        if val_si % 100 == 0: print('>'.format(c), end='')
    print('')

    te_si = 0
    print('Class {} (test) : '.format(c), end=' ')
    while te_si < te_i//NUM_CLASSES:
        i += 1
        # In-dist test:
        ########################################
        # get the image
        im = images[i]

        # get the annoatations
        annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # pick largest area object
        max_ann = -1
        for _pos in range(len(anns)):
            if anns[_pos]['area'] > max_ann:
                pos = _pos
                max_ann = anns[_pos]['area']
        if max_ann < 10000: continue;

        img_path = os.path.join('../data/coco/train2017', im['file_name'])
        I = np.asarray(Image.open(img_path))
        if len(I.shape) == 2:
            I = np.tile(I[:,:,None], [1,1,3])

        mask = np.tile(255*coco.annToMask(anns[pos]).astype('uint8')[:,:,None], [1,1,3])
        resized_mask = resize(mask, (64, 64), anti_aliasing=True)
        resized_image = resize(I, (64, 64), anti_aliasing=True)

        id_test_file['resized_images'][te_s, ...] = np.transpose(resized_image, (2,0,1))
        id_test_file['resized_mask'][te_s, ...] = np.transpose(resized_mask, (2, 0, 1))
        id_test_file['y'][te_s] = c
        te_s += 1
        te_si += 1
        if te_si % 100 == 0: print('>'.format(c), end='')
        ########################################
    print('')

train_file.close()
val_id_file.close()
id_test_file.close()

