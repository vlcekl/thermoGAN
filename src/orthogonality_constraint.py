import numpy
import tensorflow

from keras.constraints import Constraint
from tensorflow.linalg import expm


class Orthogonal(Constraint):
    """Orthogonal weight constraint.
    Constrains the weights incident to each hidden unit
    to be orthogonal when there are more inputs than hidden units.
    When there are more hidden units than there are inputs,
    the rows of the layer's weight matrix are constrainted
    to be orthogonal.
    # Arguments
        axis: Axis or axes along which to calculate weight norms.
            `None` to use all but the last (output) axis.
            For instance, in a `Dense` layer the weight matrix
            has shape `(input_dim, output_dim)`,
            set `axis` to `0` to constrain each weight vector
            of length `(input_dim,)`.
            In a `Conv2D` layer with `data_format="channels_last"`,
            the weight tensor has shape
            `(rows, cols, input_depth, output_depth)`,
            set `axis` to `[0, 1, 2]`
            to constrain the weights of each filter tensor of size
            `(rows, cols, input_depth)`.
        orthonormal: If `True`, the weight matrix is further
            constrained to be orthonormal along the appropriate axis.
    """

    def __init__(self, axis=None, orthonormal=False):
        self.axis = axis
        self.orthonormal = orthonormal

    def __call__(self, w):
        if self.axis is None:
            self.axis = list(range(len(w.shape) - 1))
        elif type(self.axis) == int:
            self.axis = [self.axis]
        else:
            self.axis = numpy.asarray(self.axis, dtype='uint8')
        self.axis = list(self.axis)

        axis_shape = [w.shape[a] for a in self.axis]
        perm = [i for i in range(len(w.shape) - 1) if i not in self.axis]
        perm.extend(self.axis)
        perm.append(len(w.shape) - 1)
        w = tensorflow.transpose(w, perm=perm)
        shape = w.shape
        w = tensorflow.reshape(w, [-1] + axis_shape + [shape[-1]])
        w = tensorflow.map_fn(self.orthogonalize, w)
        w = tensorflow.reshape(w, shape)
        w = tensorflow.transpose(w, perm=numpy.argsort(perm))
        return w

    def orthogonalize(self, w):
        shape = w.shape
        output_shape = int(shape[-1])
        input_shape = int(numpy.prod(shape[:-1]))
        final_shape = int(max(input_shape, output_shape))
        w_matrix = tensorflow.reshape(w, (output_shape, input_shape))
        w_matrix = tensorflow.pad(w_matrix,
                           tensorflow.constant([
                               [0, final_shape - output_shape],
                               [0, final_shape - input_shape]
                           ]))
        upper_triangular = tensorflow.matrix_band_part(w_matrix, 1, -1)
        antisymmetric = upper_triangular - tensorflow.transpose(upper_triangular)
        rotation = expm(antisymmetric)
        w_final = tensorflow.slice(rotation, [0,] * 2, [output_shape, input_shape])
        if not self.orthonormal:
            if input_shape >= output_shape:
                w_final = tensorflow.matmul(w_final,
                                            tensorflow.matrix_band_part(
                                                tensorflow.slice(w_matrix,
                                                                 [0, 0],
                                                                 [input_shape, input_shape]),
                                                0, 0))
            else:
                w_final = tensorflow.matmul(tensorflow.matrix_band_part(
                                                tensorflow.slice(w_matrix,
                                                                 [0, 0],
                                                                 [output_shape, output_shape]),
                                                0, 0), w_final)
        return tensorflow.reshape(w_final, w.shape)

    def get_config(self):
        return {'axis': self.axis,
                'orthonormal': self.orthonormal}