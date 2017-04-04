import numpy as np
from keras.layers import Input
from keras.models import Model
from RoiPooling import RoiPooling
import keras.backend as K

dim_ordering = K.image_dim_ordering()
assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

pooling_regions = [1, 2, 4]
num_rois = 2
num_channels = 3

if dim_ordering == 'tf':
    in_img = Input(shape=(None, None, num_channels))
elif dim_ordering == 'th':
    in_img = Input(shape=(num_channels, None, None))

in_roi = Input(shape=(num_rois, 4))

out_roi_pool = RoiPooling(pooling_regions, num_rois)([in_img, in_roi])

model = Model([in_img, in_roi], out_roi_pool)
model.summary()

model.compile(loss='mse', optimizer='sgd')
batch_size = 2

for img_size in [8, 16, 32]:

    if dim_ordering == 'th':
        X_img = np.random.rand(batch_size, num_channels, img_size, img_size)
        row_length = [float(X_img.shape[2]) / i for i in pooling_regions]
        col_length = [float(X_img.shape[3]) / i for i in pooling_regions]
    elif dim_ordering == 'tf':
        X_img = np.random.rand(batch_size, img_size, img_size, num_channels)
        row_length = [float(X_img.shape[1]) / i for i in pooling_regions]
        col_length = [float(X_img.shape[2]) / i for i in pooling_regions]

    X_roi = np.array([[0, 0, img_size // 1, img_size // 1],
                      [0, 0, img_size // 2, img_size // 2]],dtype=np.int32)

    X_roi = np.tile(X_roi,(batch_size,1))

    X_roi = np.reshape(X_roi, (batch_size, num_rois, 4))

    Y = model.predict([X_img, X_roi])
    for batch in range(batch_size):
        for roi in range(num_rois):

            if dim_ordering == 'th':
                X_curr = X_img[batch, :, X_roi[batch, roi, 0]:X_roi[batch, roi, 2], X_roi[batch, roi, 1]:X_roi[batch, roi, 3]]
                row_length = [float(X_curr.shape[1]) / i for i in pooling_regions]
                col_length = [float(X_curr.shape[2]) / i for i in pooling_regions]
            elif dim_ordering == 'tf':
                X_curr = X_img[batch, X_roi[batch, roi, 0]:X_roi[batch, roi, 2], X_roi[batch, roi, 1]:X_roi[batch, roi, 3], :]
                row_length = [float(X_curr.shape[0]) / i for i in pooling_regions]
                col_length = [float(X_curr.shape[1]) / i for i in pooling_regions]

            idx = 0

            for pool_num, num_pool_regions in enumerate(pooling_regions):
                print("POOL NUM:",pool_num)
                for ix in range(num_pool_regions):
                    for jy in range(num_pool_regions):
                        for cn in range(num_channels):

                            x1 = int(round(ix * col_length[pool_num]))
                            x2 = int(round(ix * col_length[pool_num] + col_length[pool_num]))
                            y1 = int(round(jy * row_length[pool_num]))
                            y2 = int(round(jy * row_length[pool_num] + row_length[pool_num]))

                            if dim_ordering == 'th':
                                m_val = np.max(X_curr[cn, y1:y2, x1:x2])
                            elif dim_ordering == 'tf':
                                m_val = np.max(X_curr[y1:y2, x1:x2, cn])

                            np.testing.assert_almost_equal(
                                m_val, Y[batch, roi, idx], decimal=6)
                            idx += 1
                        
print('Passed roi pooling test')
