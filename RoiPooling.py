import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer


class RoiPooling(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(batch_size, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(batch_size, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(batch_size,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(batch_size, num_rois, channels * sum([i * i for i in pool_list])`
    '''

    def __init__(self, pool_list, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list
        self.num_rois = num_rois

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            self.nb_channels = input_shape[0][1]
        elif self.dim_ordering == 'tf':
            self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return (None, self.num_rois, self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list, 'num_rois': self.num_rois}
        base_config = super(RoiPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        print(img)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = K.reshape(rois[:, roi_idx, 0], [-1, 1])
            y = K.reshape(rois[:, roi_idx, 1], [-1, 1])
            w = K.reshape(rois[:, roi_idx, 2], [-1, 1])
            h = K.reshape(rois[:, roi_idx, 3], [-1, 1])

            row_length = K.reshape(K.stack([w / i for i in self.pool_list], 1), [-1, len(self.pool_list)])
            col_length = K.reshape(K.stack([h / i for i in self.pool_list], 1), [-1, len(self.pool_list)])

            def fun(elem):
                x_crop_in, x_in, y_in, row_in, col_in = elem
                acc = []
                for pool_num, num_pool_regions in enumerate(self.pool_list):
                    for ix in range(num_pool_regions):
                        for jy in range(num_pool_regions):
                            x1 = x_in + ix * col_in[pool_num]
                            x2 = x1 + col_in[pool_num]
                            y1 = y_in + jy * row_in[pool_num]
                            y2 = y1 + row_in[pool_num]

                            x1 = K.cast(K.round(x1), 'int32')[0]
                            x2 = K.cast(K.round(x2), 'int32')[0]
                            y1 = K.cast(K.round(y1), 'int32')[0]
                            y2 = K.cast(K.round(y2), 'int32')[0]

                            new_shape = [1, y2 - y1,
                                         x2 - x1, input_shape[3]]

                            x_crop = x_crop_in[y1:y2, x1:x2, :]
                            xm = K.reshape(x_crop, new_shape)
                            pooled_val = K.max(xm, axis=(1, 2))
                            acc.append(pooled_val)
                return acc

            x = tf.map_fn(fun, [img, x, y, row_length, col_length], dtype=[tf.float32] * (sum(self.pool_list) * 3))
            outputs.extend(x)

        final_output = K.concatenate(outputs, axis=1)
        final_output = K.reshape(final_output, (-1, self.num_rois, self.nb_channels * self.num_outputs_per_channel))

        return final_output
