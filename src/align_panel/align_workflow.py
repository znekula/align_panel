from __future__ import annotations
import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from skimage import transform as sktransform
import numpy as np
from typing import TYPE_CHECKING

import panel as pn
import aperture as ap
from aperture.layouts.step import Step

from .imgsetlib import Imgset, H5file
from .align_panel import point_registration, fine_adjust, assure_size, array_format

if TYPE_CHECKING:
    from aperture.layouts.workflow import TemplateWorkflow
    from aperture.display.figure import BokehFigure


class HDF5Step(Step):
    # @staticmethod
    # def img_choices():
    #     return {'Amplitude': lambda x: (x.amplitude_stat, x.amplitude),
    #             'Phase': lambda x: (x.phase_stat, x.phase),
    #             'Phase_Unwrapped': lambda x: (x.unwrapped_phase_stat, x.unwrapped_phase)}

    @property
    def hdf5_path(self):
        return self.get_shared('hdf5_path')

    def load_hdf5_meta(self):
        return H5file(self.hdf5_path)

    @property
    def available_imgsets(self):
        return self.load_hdf5_meta().imageset_names

    @property
    def static_key(self):
        # Implicitly take first reference image
        return self.load_hdf5_meta().ref_imageset_name[0]

    def get_imgset(self, key) -> Imgset:
        return Imgset(self.hdf5_path, key)

    def get_static_imgset(self):
        return self.get_imgset(self.static_key)

    def get_moving_imgset(self):
        return self.get_imgset(self.moving_key)

    @property
    def moving_key(self):
        return self.get_shared('moving_key')

    @moving_key.setter
    def moving_key(self, value: str):
        self.workflow.shared['moving_key'] = value

    def imgset_viewer(self,
                      imgset: Imgset | pn.widgets.Select,
                      name_prefix=''):
        img_select = pn.widgets.Select(name=name_prefix + 'Array',
                                       options=[])

        if isinstance(imgset, pn.widgets.Select):
            imgset_select = imgset
            imgset = self.get_imgset(imgset_select.value)

            def _update_options(*events):
                nonlocal imgset
                imgset = self.get_imgset(imgset_select.value)
                img_select.options = imgset.get_2d_image_keys()
                img_select.param.trigger('value')

            imgset_select.param.watch(_update_options, 'value')
        
        img_select.options = imgset.get_2d_image_keys()

        # Display the selected image
        img = ap.image((400, 400))
        img.add_colorbar()
        fig = img.fig
        fig.set_title(name_prefix + 'Image')

        def _update_image(event):
            array_key = event.new
            try:
                array = imgset.get_data(array_key)
                if (not array.ndim == 2) or np.iscomplexobj(array.dtype):
                    raise TypeError
                img.raw_image = array
                fig.autorange_images()
                fig.set_title(name_prefix + array_key)
            except (KeyError, TypeError):
                img.clear_data()

        # Change the displayed image each time we select
        img_select.param.watch(_update_image, 'value')
        if img_select.options:
            img_select.param.trigger('value')

        return fig, img_select


class ImageSelector(HDF5Step):
    @staticmethod
    def label():
        return 'Select'

    @staticmethod
    def title():
        return 'Select images'

    def pane(self):
        md = pn.pane.Markdown(object="""
- Choose the moving image from the HDF5 file
- Inspect the contents of the file
""", min_width=400)
        static_key = self.static_key
        static_key_text = pn.widgets.StaticText(value=f'Static imgset: {static_key}', min_width=300)
        static_imgset = self.get_static_imgset()

        static_fig, static_select_box = self.imgset_viewer(static_imgset, name_prefix='Static ')
        static_fig.scale_to_frame_size(frame_height=500)

        imgset_select = pn.widgets.Select(name='Moving imgset',
                                          options=self.available_imgsets)
        fig, imgset_array_select = self.imgset_viewer(imgset_select, name_prefix='Moving ')
        fig.scale_to_frame_size(frame_height=500)

        current_transform_md = pn.pane.Markdown(object='')

        def _update_tmat_display(*event):
            imgset = self.get_imgset(imgset_select.value)
            current_tmat = imgset.get_tmat()
            current_transform_md.object = array_format(current_tmat)
        
        _update_tmat_display()

        imgset_select.param.watch(_update_tmat_display, 'value')
        clear_tmat_btn = pn.widgets.Button(name='Clear Transform')

        def _clear_transform(*event):
            imgset = self.get_imgset(imgset_select.value)
            imgset.clear_tmat()
            _update_tmat_display()

        clear_tmat_btn.on_click(_clear_transform)

        return pn.Row(pn.Column(md, static_key_text, static_select_box, static_fig),
                      pn.Column(imgset_select, imgset_array_select, fig,
                                pn.Row(current_transform_md, clear_tmat_btn)))
    
    def after(self, pane):
        self.moving_key = pane[1][0].value
        super().after(pane)


class AutoAlignStep(HDF5Step):
    @staticmethod
    def label():
        return 'Autoalign'

    def pane(self):
        md = pn.pane.Markdown(object="""
- Set autoalignment parameters
- Alignment is performed when moving to the next page
- Results will be visible on the next page
""", min_width=400)
        layout = ap.layout(mode='simple')
        layout.panes[0].append(md)

        bins_int_input = pn.widgets.IntInput(name='Bins',
                                             value=62,
                                             start=1)
        del_back_checkbox = pn.widgets.Checkbox(name='Remove background',
                                                value=True)

        static_arrays = set(self.get_static_imgset().get_2d_image_keys())
        moving_arrays = set(self.get_moving_imgset().get_2d_image_keys())
        array_choices = static_arrays.intersection(moving_arrays)

        image_choice = pn.widgets.Select(name='Image choice',
                                         options=list(array_choices))

        method_choice = pn.widgets.Select(name='Align method',
                                          options=Imgset.autoalign_methods())

        layout.panes[0].add_element_group('params',
                                          {'bins_int_input': bins_int_input,
                                           'del_back_checkbox': del_back_checkbox,
                                           'method_choice': method_choice,
                                           'image_choice': image_choice}, #
                                           container=pn.layout.Card(title='Parameters'))
        return layout
    
    def after(self, pane):
        bins = pane.panes[0]['bins_int_input'].value
        del_back = pane.panes[0]['del_back_checkbox'].value
        method_choice = pane.panes[0]['method_choice'].value
        img_choice = pane.panes[0]['image_choice'].value

        static_imgset = self.get_static_imgset()
        static_image = static_imgset.get_data(img_choice)
        moving_imgset = self.get_moving_imgset()
        moving_image = moving_imgset.get_data(img_choice)
        try:
            tmat = moving_imgset.autoalign(static_image,
                                           assure_size(moving_image, static_image.shape),
                                           transformation=method_choice,
                                           bins=bins,
                                           del_back=del_back)
        except ValueError as e:
            return self.return_error(str(e))

        self.workflow.shared['temp_tmat'] = tmat
        self.results.set(data=tmat, meta={'img_choice': img_choice})
        super().after(pane)

    @staticmethod
    def provides(results):
        return {'img_choice': results.meta['img_choice']}


class ConfirmTransformStep(HDF5Step):
    @staticmethod
    def label():
        return 'Save'

    @staticmethod
    def title():
        return 'Save transform'

    def pane(self):
        md = pn.pane.Markdown(object="""
- Inspect new transformation matrix
- Press save to store new transform
""", min_width=400)
        layout = ap.layout(mode='simple')
        layout.panes[0].append(md)

        static_imgset = self.get_static_imgset()
        fig, img_select = self.imgset_viewer(static_imgset)
        moving_imgset = self.get_moving_imgset()

        current_tmat = moving_imgset.get_tmat()
        proposed_tmat = self.get_shared('temp_tmat')[:]
        combined_tmat = current_tmat @ proposed_tmat

        overlay = fig.add_image(height=400, width=400)
        overlay_alpha_slider = overlay.get_alpha_slider(0.5)
        layout.panes[0].append(img_select)
        layout.panes[0].append(fig)
        layout.panes[0].append(overlay_alpha_slider)

        def _get_transformed(*event):
            selected = img_select.value
            moving = moving_imgset.get_data(selected)
            transformed = moving_imgset.apply_tmat(moving, tmat=combined_tmat)
            overlay.raw_image = transformed
            fig.autorange_images()

        img_select.param.watch(_get_transformed, 'value')

        save_btn = pn.widgets.Button(name='Save transform')
        save_msg = pn.widgets.StaticText(value='Not yet saved')
        layout.panes[1].append(save_btn)
        layout.panes[1].append(save_msg)

        def _apply_transform(*event):
            current_tmat = moving_imgset.get_tmat()
            tmat = self.workflow.shared.pop('temp_tmat', None)
            if tmat is None:
                # nothing to save
                return
            combined_tmat = current_tmat @ proposed_tmat
            moving_imgset.save_tmat(combined_tmat)
            save_msg.value = 'Saved transformation matrix'

        save_btn.on_click(_apply_transform)

        return layout


class ChosenImagesStep(HDF5Step):
    def before(self, **kwargs):
        static_imgset = self.get_static_imgset()
        moving_imgset = self.get_moving_imgset()
        img_choice = self.get_result('img_choice', 'unwrapped_phase')
        img_static = static_imgset.get_data(img_choice, aligned=True)
        img_moving = moving_imgset.get_data(img_choice, aligned=True)
        self.results.add_child('static', data=img_static)
        self.results.add_child('moving', data=assure_size(img_moving, img_static.shape))

    def after(self, pane):
        self.results.freeze()
        self.workflow.shared['temp_tmat'] = self.results.data['transform'].params
        super().after(pane)


class PointsAlignStep(ChosenImagesStep):
    @staticmethod
    def label():
        return 'Points'

    @staticmethod
    def title():
        return 'Point-based alignment'

    def pane(self):
        layout, getter = point_registration(self.results['static'].data, self.results['moving'].data)
        self.results.set(data=lambda **x: getter())
        return layout.finalize()


class FineAlignStep(ChosenImagesStep):
    @staticmethod
    def label():
        return 'Fine'

    @staticmethod
    def title():
        return 'Fine-adjust alignment'

    def pane(self):
        layout, getter = fine_adjust(self.results['static'].data, self.results['moving'].data)
        self.results.set(data=lambda **x: getter())
        return layout


class FinalPage(Step):
    @staticmethod
    def label():
        return 'End'

    def pane(self):
        md = pn.pane.Markdown(object=f"""
### Workflow completed

Results have been saved to {self.get_shared('hdf5_path')}
""", width=800)
        return pn.Column(md)


def build_workflow(workflow: 'TemplateWorkflow', args=None, hdf5_path=None):
    workflow.set_title('Holo-Alignment')
    workflow.set_shared({'hdf5_path': hdf5_path})
    workflow.add_step(ImageSelector())
    workflow.add_step(AutoAlignStep())
    workflow.add_step(ConfirmTransformStep(), forward_only=True)
    workflow.add_step(PointsAlignStep())
    workflow.add_step(ConfirmTransformStep(), forward_only=True)
    workflow.add_step(FineAlignStep())
    workflow.add_step(ConfirmTransformStep(), forward_only=True)
    workflow.add_step(FinalPage())
    return workflow


if __name__ == '__main__':
    from aperture.layouts.workflow import TemplateWorkflow

    filepaths = {
        'static': {'sample': 'test_data/-4-H.dm3',
                   'reference': 'test_data/-4-R2.dm3'},
        'moving': {'sample': 'test_data/+4-H.dm3',
                   'reference': 'test_data/+4-R2.dm3'},
    }
    hdf5_path = './test_data/testpath.h5'
    workflow = TemplateWorkflow(shared={
                                    'filepaths': filepaths,
                                    'hdf5_path': hdf5_path
                                })
    build_workflow(workflow).show()


