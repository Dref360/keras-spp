import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = K.reshape(rois[:, roi_idx, 0], [-1, 1])
            y = K.reshape(rois[:, roi_idx, 1], [-1, 1])
            w = K.reshape(rois[:, roi_idx, 2], [-1, 1])
            h = K.reshape(rois[:, roi_idx, 3], [-1, 1])

            row_length = K.reshape(w / float(self.pool_size), [-1, 1])
            col_length = K.reshape(h / float(self.pool_size), [-1, 1])

            num_pool_regions = self.pool_size
            def fun(elem):
                x_crop_in, x_in, y_in, row_in, col_in = elem
                acc = []
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x_in + ix * row_in
                        x2 = x1 + row_in
                        y1 = y_in + jy * col_in
                        y2 = y1 + col_in

                        x1 = K.cast(x1, 'int32')[0]
                        x2 = K.cast(x2, 'int32')[0]
                        y1 = K.cast(y1, 'int32')[0]
                        y2 = K.cast(y2, 'int32')[0]

                        new_shape = [1, y2 - y1,
                                     x2 - x1, input_shape[3]]
                        x_crop = x_crop_in[y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        acc.append(pooled_val)
                return acc

            x = tf.map_fn(fun, [img, x, y, row_length, col_length], dtype=[tf.float32] * (num_pool_regions * 2))
            outputs.extend(x)

        final_output = K.concatenate(outputs, axis=1)
        final_output = K.reshape(final_output, (-1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
