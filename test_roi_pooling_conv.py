import pdb

import keras.backend as K
import numpy as np
from keras.layers import Input
from keras.models import Model

from RoiPoolingConv import RoiPoolingConv

dim_ordering = K.image_dim_ordering()
assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

pooling_regions = 2
num_rois = 4
num_channels = 12

if dim_ordering == 'tf':
    in_img = Input(shape=(None, None, num_channels))
elif dim_ordering == 'th':
    in_img = Input(shape=(num_channels, None, None))

in_roi = Input(shape=(num_rois, 4))

out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([in_img, in_roi])

model = Model([in_img, in_roi], out_roi_pool)
model.summary()
batch_size = 2
model.compile(loss='mse', optimizer='sgd')

for img_size in [32]:
    if dim_ordering == 'th':
        X_img = np.random.rand(batch_size, num_channels, img_size, img_size)
        row_length = [float(X_img.shape[2]) / pooling_regions]
        col_length = [float(X_img.shape[3]) / pooling_regions]
    elif dim_ordering == 'tf':
        X_img = np.random.rand(batch_size, img_size, img_size, num_channels)
        row_length = [float(X_img.shape[1]) / pooling_regions]
        col_length = [float(X_img.shape[2]) / pooling_regions]

    X_roi = np.array([[0, 0, img_size / 2, img_size / 2],
                      [0, img_size / 2, img_size / 2, img_size / 2],
                      [img_size / 2, 0, img_size / 2, img_size / 2],
                      [img_size / 2, img_size / 2, img_size / 2, img_size / 2]],dtype=np.int32)

    X_roi = np.tile(X_roi, (batch_size, 1))

    X_roi = np.reshape(X_roi, (batch_size, num_rois, 4))

    Y = model.predict([X_img, X_roi])
    for bs in range(batch_size):
        for roi in range(num_rois):

            if dim_ordering == 'th':
                X_curr = X_img[bs, :, X_roi[bs, roi, 1]:X_roi[bs, roi, 1] + X_roi[bs, roi, 3],
                         X_roi[bs, roi, 0]:X_roi[bs, roi, 0] + X_roi[bs, roi, 2]]
                row_length = float(X_curr.shape[1]) / pooling_regions
                col_length = float(X_curr.shape[2]) / pooling_regions
            elif dim_ordering == 'tf':
                X_curr = X_img[bs, X_roi[bs, roi, 1]:X_roi[bs, roi, 1] + X_roi[bs, roi, 3],
                         X_roi[bs, roi, 0]:X_roi[bs, roi, 0] + X_roi[bs, roi, 2], :]
                row_length = float(X_curr.shape[0]) / pooling_regions
                col_length = float(X_curr.shape[1]) / pooling_regions

            idx = 0

            for ix in range(pooling_regions):
                for jy in range(pooling_regions):
                    for cn in range(num_channels):

                        x1 = int((ix * col_length))
                        x2 = int((ix * col_length + col_length))
                        y1 = int((jy * row_length))
                        y2 = int((jy * row_length + row_length))
                        dx = max(1, x2 - x1)
                        dy = max(1, y2 - y1)
                        x2 = x1 + dx
                        y2 = y1 + dy

                        if dim_ordering == 'th':
                            m_val = np.max(X_curr[cn, y1:y2, x1:x2])
                            if abs(m_val - Y[bs, roi, cn, jy, ix]) > 0.01:
                                pdb.set_trace()
                            np.testing.assert_almost_equal(
                                m_val, Y[bs, roi, cn, jy, ix], decimal=6)
                            idx += 1
                        elif dim_ordering == 'tf':
                            m_val = np.max(X_curr[y1:y2, x1:x2, cn])
                            if abs(m_val - Y[bs, roi, jy, ix, cn]) > 0.01:
                                pdb.set_trace()
                            np.testing.assert_almost_equal(
                                m_val, Y[bs, roi, jy, ix, cn], decimal=6)
                            idx += 1

print('Passed roi pooling test')
