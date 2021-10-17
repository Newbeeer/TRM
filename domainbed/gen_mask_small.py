import numpy as np
import random
import IPython
import argparse

parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--rate', type=float, default=0.7)
args = parser.parse_args()

mask1 = np.ones([10, 64, 28, 28])
mask2 = np.ones([10, 128, 14, 14])
mask3 = np.ones([10, 128, 14, 14])
mask4 = np.ones([10, 128, 14, 14])


mask1.reshape(-1)[random.sample(range(np.prod(mask1.shape)), (np.sum(mask1)*args.rate).astype(int))]=0
mask2.reshape(-1)[random.sample(range(np.prod(mask2.shape)), (np.sum(mask2)*args.rate).astype(int))]=0
mask3.reshape(-1)[random.sample(range(np.prod(mask3.shape)), (np.sum(mask3)*args.rate).astype(int))]=0
mask4.reshape(-1)[random.sample(range(np.prod(mask4.shape)), (np.sum(mask4)*args.rate).astype(int))]=0


mask1 = np.expand_dims(mask1, 1)
mask2 = np.expand_dims(mask2, 1)
mask3 = np.expand_dims(mask3, 1)
mask4 = np.expand_dims(mask4, 1)

print('mask num', np.sum(1-mask1), 'total num', np.prod(mask1.shape))
print('mask num', np.sum(1-mask2), 'total num', np.prod(mask2.shape))
print('mask num', np.sum(1-mask3), 'total num', np.prod(mask3.shape))
print('mask num', np.sum(1-mask4), 'total num', np.prod(mask4.shape))

np.save('./arr/mask_small_'+str(args.rate)+'_1.npy', mask1)
np.save('./arr/mask_small_'+str(args.rate)+'_2.npy', mask2)
np.save('./arr/mask_small_'+str(args.rate)+'_3.npy', mask3)
np.save('./arr/mask_small_'+str(args.rate)+'_4.npy', mask4)

