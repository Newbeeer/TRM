import os, subprocess
import numpy as np
import random
from PIL import Image
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')

import h5py
################ Paths and other configs - Set these #################################
output_dir = os.path.join('../data', 'places')
places_dir = os.path.join('../data', 'places', 'data_256')

biased_places = ['b/beach',
                 'c/canyon',
                 'b/building_facade',
                 's/staircase',
                 'd/desert/sand',
                 'c/crevasse',
                 'b/bamboo_forest',
                 'f/forest/broadleaf',
                 'b/ball_pit',
                 ]

unbiased_places = [
                 'k/kasbah',
                 'l/lighthouse',
                 'p/pagoda',
                 'r/rock_arch']

validation_unbiased_places = [
        'o/oast_house',
        'o/orchard',
        'v/viaduct']

test_unbiased_places = [
                 'w/water_tower',
                 'w/waterfall',
                 'z/zen_garden']


biased_places = biased_places + unbiased_places + validation_unbiased_places + test_unbiased_places
NUM_CLASSES = len(biased_places)


print('----- Bias places ------')
print(biased_places)

dataset_name = 'cocoplaces'
h5pyfname = os.path.join(output_dir, dataset_name)
print('---------------- FNAME --------------')
print(h5pyfname)
if not os.path.exists(h5pyfname):
    os.makedirs(h5pyfname)
######################################################################################
biased_place_fnames = {}
for i, target_place in enumerate(biased_places):
    L = [f'{target_place}/{filename}' for filename in os.listdir(os.path.join(places_dir, target_place)) if filename.endswith('.jpg')]
    random.shuffle(L)
    biased_place_fnames[i] = L


tr_i = 1000 * NUM_CLASSES
train_fname = os.path.join(h5pyfname,'places.h5py')
if os.path.exists(train_fname): subprocess.call(['rm', train_fname])
train_file = h5py.File(train_fname, mode='w')
train_file.create_dataset('resized_place', (NUM_CLASSES,1000,3,64,64), dtype=np.dtype('float32'))

tr_s, val_s, te_s = 0, 0, 0

for c in range(NUM_CLASSES):

    tr_si = 0
    print('Class {} (train) : '.format(c), biased_places[c], end=' ')
    while tr_si < 1000:
        place_path = biased_place_fnames[c][tr_si]
        place_img = np.asarray(Image.open(os.path.join(places_dir, place_path)).convert('RGB'))
        # that's the one:
        resized_place = resize(place_img, (64, 64))
        train_file['resized_place'][c, tr_si, ...] = np.transpose(resized_place, (2,0,1))
        tr_si += 1
        if tr_si % 100 == 0: print('>'.format(c), end='')
        ########################################
    print('')

train_file.close()

