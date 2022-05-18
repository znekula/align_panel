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
            transform = sktransform.EuclideanTransform().params
        elif len(transform_mxs) >= 2:
            transform = functools.reduce(np.matmul, transform_mxs)
        else:
            transform = transform_mxs[0]
        return sktransform.AffineTransform(matrix=transform)

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


class ImageTransformerState(ImageTransformer):
    def __init__(self, *args, **kwargs):
        self.meta = kwargs.pop('meta', {})
        super().__init__(*args, **kwargs)
        self._history = []
        self._current_key = None

    def add_transform(self, transform, output_shape=None):
        super().add_transform(transform, output_shape=output_shape)
        self._history.append(self._current_key)

    def remove_transforms(self, key):
        self._transforms = [t for k, t in zip(self._history, self.transforms) if k != key]
        self._reshapes = [s for k, s in zip(self._history, self._reshapes) if k != key]
        self._history = [k for k in self._history if k != key]

    def clear_transforms(self):
        super().clear_transforms()
        self._history.clear()

    @staticmethod
    def get_key():
        return str(uuid.uuid4())

    def get_meta(self, key, default=None):
        return self.meta.get(key, default)

    def set_meta(self, key, value):
        self.meta[key] = value

    @contextlib.contextmanager
    def group_transforms(self, key=None):
        if not key:
            key = self.get_key()
        self._current_key = key
        yield key
        self._current_key = None