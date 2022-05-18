import uuid
import contextlib
import functools
import numpy as np
from skimage import transform as sktransform


class ImageTransformer:
    def __init__(self, image):
        self._image = image
        self._transforms = []
        self._reshapes = []

    def set_image(self, image):
        self._image = image

    @property
    def transforms(self):
        return self._transforms

    def add_transform(self, transform, output_shape=None):
        self.transforms.append(transform)
        self._reshapes.append(output_shape)

    def add_null_transform(self, output_shape=None):
        self.transforms.append(self._null_transform())
        self._reshapes.append(output_shape)

    @staticmethod
    def _null_transform():
        return sktransform.EuclideanTransform()

    def clear_transforms(self):
        self.transforms.clear()
        self._reshapes.clear()

    def current_shape(self):
        reshapes = [r for r in self._reshapes if r is not None]
        if reshapes:
            return reshapes[-1]
        return self._image.shape

    def get_combined_transform(self):
        transform_mxs = [t.params for t in self.transforms]
        if not transform_mxs:
            transform = self._null_transform().params
        elif len(transform_mxs) >= 2:
            transform = functools.reduce(np.matmul, transform_mxs)
        else:
            transform = transform_mxs[0]
        combined = sktransform.AffineTransform(matrix=transform)
        # This combines existing transforms into a single step for speed
        # If necessary code also exists to have an undo functionality
        # by retaining a history of each transform step
        current_shape = self.current_shape()
        self.clear_transforms()
        self.add_transform(combined, output_shape=current_shape)
        return combined

    def get_transformed_image(self, preserve_range=True, cval=0.,**kwargs):
        if not self.transforms:
            return self._image
        combined_transform = self.get_combined_transform()
        output_shape = self.current_shape()
        return sktransform.warp(self._image,
                                combined_transform,
                                output_shape=output_shape,
                                preserve_range=preserve_range,
                                cval=cval,
                                **kwargs)

    def get_current_center(self):
        current_shape = np.asarray(self.current_shape())
        return current_shape / 2.

    def translate(self, xshift=0., yshift=0., output_shape=None):
        transform = sktransform.EuclideanTransform(translation=(xshift, yshift))
        self.add_transform(transform, output_shape=output_shape)

    def rotate_about_point(self, point_yx, rotation_degrees=None, rotation_rad=None):
        if rotation_degrees and rotation_rad:
            raise ValueError('Cannot specify both degrees and radians')
        elif rotation_degrees:
            rotation_rad = np.deg2rad(rotation_degrees)
        if not rotation_rad:
            if rotation_rad == 0.:
                return
            raise ValueError('Must specify one of degrees or radians')
        
        transform = sktransform.EuclideanTransform(rotation=rotation_rad)
        self._operation_with_origin(point_yx, transform)

    def rotate_about_center(self, **kwargs):
        current_center = self.get_current_center()
        return self.rotate_about_point(current_center, **kwargs)

    def uniform_scale_centered(self, scale_factor, output_shape=None):
        transform = sktransform.SimilarityTransform(scale=scale_factor)
        current_center = self.get_current_center()
        self._operation_with_origin(current_center, transform)

    def xy_scale_about_point(self, point_yx, xscale=1., yscale=1.):
        transform = sktransform.AffineTransform(scale=(xscale, yscale))
        self._operation_with_origin(point_yx, transform)

    def xy_scale_about_center(self, **kwargs):
        current_center = self.get_current_center()
        return self.xy_scale_about_point(current_center, **kwargs)

    def _operation_with_origin(self, origin_yx, transform):
        origin_xy = np.asarray(origin_yx).astype(float)[::-1]
        forward_shift = sktransform.EuclideanTransform(translation=origin_xy)
        self.add_transform(forward_shift)
        self.add_transform(transform)
        backward_shift = sktransform.EuclideanTransform(translation=-1 * origin_xy)
        self.add_transform(backward_shift)
