"""this script show basic operations done during image alignment from the raw data to aligned phase images"""

import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import pathlib
from functools import partial

import panel as pn
import aperture as ap
from aperture.layouts.step import Step
from aperture.layouts.template_specs import ap_spec
from aperture.layouts.workflow import TemplateWorkflow
from aperture.utils import loading_context

from phase import Imgset_new
from imgsetlib import Imgset
from align_panel import point_registration, fine_adjust

class LoadStep(Step):
    @staticmethod
    def label():
        return 'Load'

    @staticmethod
    def title():
        return 'Load data'

    def pane(self):
        md = pn.pane.Markdown(object="""
- Load two sample and reference hologram pairs to align
- Run the phase reconstruction before continuing
""", min_width=400)
        layout = ap.layout(mode='simple')
        layout.panes[0].append(md)

        static_filepaths = self.get_shared('filepaths', default={'static': {}})['static']
        moving_filepaths = self.get_shared('filepaths', default={'moving': {}})['moving']

        static_real_input = pn.widgets.TextInput(name='Sample',
                                                 value=str(static_filepaths.get('sample', '')))
        static_ref_input = pn.widgets.TextInput(name='Reference',
                                                value=str(static_filepaths.get('reference', '')))
        static_load_button = pn.widgets.Button(name='Load')
        static_recon_button = pn.widgets.Button(name='Reconstruct', disabled=True)
        static_card = pn.layout.Card(static_real_input,
                                     static_ref_input,
                                     static_load_button,
                                     static_recon_button,
                                     title='Static holograms',
                                     collapsible=False)
        moving_real_input = pn.widgets.TextInput(name='Sample',
                                                 value=str(moving_filepaths.get('sample', '')))
        moving_ref_input = pn.widgets.TextInput(name='Reference',
                                                value=str(moving_filepaths.get('reference', '')))
        moving_load_button = pn.widgets.Button(name='Load')
        moving_recon_button = pn.widgets.Button(name='Reconstruct', disabled=True)
        moving_card = pn.layout.Card(moving_real_input,
                                     moving_ref_input,
                                     moving_load_button,
                                     moving_recon_button,
                                     title='Moving holograms',
                                     collapsible=False)
        layout.panes[0].add_element('static_card', static_card)
        layout.panes[0].add_element('moving_card', moving_card)

        buttons = [static_load_button,
                   static_recon_button,
                   moving_load_button,
                   moving_recon_button]        

        def _load(event, *, key, real_input, ref_input, recon_button):
            real_path = real_input.value
            ref_path = ref_input.value
            if not pathlib.Path(real_path).resolve().is_file():
                self.display_error(f'{key} file (sample) {real_path} not found')
                recon_button.disabled = True
                return
            if not pathlib.Path(ref_path).resolve().is_file():
                self.display_error(f'{key} file (ref) {ref_path} not found')
                recon_button.disabled = True
                return
            with loading_context(*buttons):
                try:
                    imgset = Imgset_new(real_path, ref_path)
                except Exception:
                    self.display_error('Error loading {key} images')
                    return
            self.results.add_child(key, data=imgset)
            recon_button.disabled = False

        static_load_button.on_click(partial(_load,
                                            key='static',
                                            real_input=static_real_input,
                                            ref_input=static_ref_input,
                                            recon_button=static_recon_button))
        moving_load_button.on_click(partial(_load,
                                            key='moving',
                                            real_input=moving_real_input,
                                            ref_input=moving_ref_input,
                                            recon_button=moving_recon_button))

        def _run_reconstruction(event, *, key):
            if not isinstance(self.results[key].data, Imgset_new):
                self.display_error(f'No images loaded for {key}')
                return
            with loading_context(*buttons):
                self.results[key].data.phase_reconstruction()

        static_recon_button.on_click(partial(_run_reconstruction, key='static'))
        moving_recon_button.on_click(partial(_run_reconstruction, key='moving'))

        output_file_path = str(self.get_shared('hdf5_path', default='output.hdf5'))
        output_file_input = pn.widgets.TextInput(name='Sample', value=output_file_path)
        output_save_button = pn.widgets.Button(name='Save HDF5')
        output_set_button = pn.widgets.Button(name='Set HDF5 path')
        output_card = pn.layout.Card(output_file_input,
                                     output_save_button,
                                     output_set_button, title='Output HDF5', collapsible=False)
        layout.panes[1].add_element('output_card', output_card)

        initial_md = 'No HDF5 found'
        if pathlib.Path(output_file_path).is_file():
            initial_md = f'HDF5: {output_file_path}'
        output_md = pn.pane.Markdown(object=initial_md, width=500)
        layout.panes[1].append(output_md)

        def _set_hdf5_path(*args, filepath=None):
            if filepath is None:
                filepath = output_file_input.value
            try:
                filepath = pathlib.Path(filepath)
            except Exception:
                self.display_error('Invalid path format')
                return
            if not filepath.is_file():
                self.display_error('Unable to find {filepath}')
                return                
            self.workflow.shared['hdf5_path'] = filepath
            self.results.set(meta={'hdf5_path': filepath})
            output_md.object = f'HDF5: {filepath}'

        output_set_button.on_click(_set_hdf5_path)

        def save_output(event):
            try:
                filepath = pathlib.Path(output_file_input.value)
            except Exception:
                self.display_error('Invalid path format')
                return
            if not filepath.suffix == '.h5':
                self.display_error('Must use .h5 suffix')
                return
            try:
                filepath.parent.mkdir(exist_ok=True, parents=True)
                assert filepath.parent.is_dir()
            except (OSError, AssertionError):
                self.display_error('Unable to create/find output save directory')
                return
            if ('static' not in self.results.keys()
                    or not isinstance(self.results['static'].data, Imgset_new)):
                self.display_error(f'No images loaded for static')
                return
            if ('moving' not in self.results.keys() or 
                    not isinstance(self.results['moving'].data, Imgset_new)):
                self.display_error(f'No images loaded for moving')
                return
            # would be better to have a Imgset_new.is_valid() method ???
            self.results['static'].data.save(str(filepath), 0)
            self.results['moving'].data.save(str(filepath), 1)
            _set_hdf5_path(filepath=filepath)


        output_save_button.on_click(save_output)

        return layout

    def after(self, pane):
        try:
            assert pathlib.Path(self.get_shared('hdf5_path')).is_file()
        except (AssertionError, KeyError, OSError, AttributeError, TypeError):
            return self.return_error('No valid hdf5 file found, cannot continue')
        super().after(pane)


class HDF5Step(Step):
    @staticmethod
    def img_choices():
        return {'Amplitude': lambda x: (x.amplitude_stat, x.amplitude),
                'Phase': lambda x: (x.phase_stat, x.phase)}

    def load_hdf5(self):
        hdf5_path = self.get_shared('hdf5_path')
        return Imgset(hdf5_path, 1)


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

        rougness_int_input = pn.widgets.IntInput(name='Rougness',
                                                 value=1000,
                                                 start=0)
        del_back_checkbox = pn.widgets.Checkbox(name='Remove background')

        image_choice = pn.widgets.Select(name='Image choice',
                                         options=[*self.img_choices().keys()])
        layout.panes[0].add_element_group('params',
                                          {'rougness_int_input': rougness_int_input,
                                           'del_back_checkbox': del_back_checkbox,
                                           'image_choice': image_choice},
                                           container=pn.layout.Card(title='Parameters'))
        return layout
    
    def after(self, pane):
        roughness = pane.panes[0]['rougness_int_input'].value
        del_back = pane.panes[0]['del_back_checkbox'].value
        img_choice = pane.panes[0]['image_choice'].value
        self.results.set(meta={'img_choice': img_choice})

        imgset = self.load_hdf5()
        static, moving = self.img_choices()[img_choice](imgset)
        imgset.autoalign(roughness, del_back=del_back, img_stat=static, img_move=moving)
        super().after(pane)

    @staticmethod
    def provides(results):
        return {'img_choice': results.meta['img_choice']}


class PointsAlignStep(HDF5Step):
    @staticmethod
    def label():
        return 'Points'

    @staticmethod
    def title():
        return 'Point-based alignment'

    def before(self, **kwargs):
        imgset = self.load_hdf5()
        img_choice = self.get_result('img_choice', 'Amplitude')
        img_static, img_moving = self.img_choices()[img_choice](imgset)
        self.results.add_child('static', data=img_static)
        self.results.add_child('moving', data=img_moving)

    def pane(self):
        md = pn.pane.Markdown(object="""
- Draw points on each image to compute an alignment
""")
        layout, getter = point_registration(self.results['static'].data, self.results['moving'].data)
        self.results.set(data=lambda **x: getter())
        return pn.Column(md, layout.finalize())

    def after(self, pane):
        self.results.freeze()
        imgset = self.load_hdf5()
        imgset.savedata(['tmat'],[self.results.data['transform'].params])
        super().after(pane)

    @staticmethod
    def provides(results):
        return {'points_transform': results.data['transform']}


class FineAlignStep(HDF5Step):
    @staticmethod
    def label():
        return 'Fine'

    @staticmethod
    def title():
        return 'Fine-adjust alignment'

    def before(self, **kwargs):
        imgset = self.load_hdf5()
        initial_transform = imgset.tmat
        img_choice = self.get_result('img_choice', 'Amplitude')
        img_static, img_moving = self.img_choices()[img_choice](imgset)
        self.results.add_child('static', data=img_static)
        self.results.add_child('moving', data=img_moving)
        self.results.add_child('initial_transform', data=initial_transform)

    def pane(self):
        md = pn.pane.Markdown(object="""
- Adjust alignment by using the arrows and plot tools
""", min_width=400)
        layout, getter = fine_adjust(self.results['static'].data, self.results['moving'].data,
                                     initial_transform=self.get_result('points_transform'))
        self.results.set(data=lambda **x: getter())
        layout.insert(0, md)
        return layout

    def after(self, pane):
        self.results.freeze()
        imgset = self.load_hdf5()
        imgset.savedata(['tmat'],[self.results.data['transform'].params])
        super().after(pane)

    @staticmethod
    def provides(results):
        return {'fine_transform': results.data['transform']}


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


def get_workflow(filepaths=None, hdf5_path=None, panel_spec=ap_spec):
    ps = {}
    if isinstance(panel_spec, dict):
        ps = panel_spec
    elif callable(panel_spec):
        ps = panel_spec()

    workflow = TemplateWorkflow(title='Holo-Alignment',
                                panel_spec=ps,
                                shared={'hdf5_path': hdf5_path,
                                        'filepaths': filepaths})
    workflow.add_step(LoadStep())
    workflow.add_step(AutoAlignStep())
    workflow.add_step(PointsAlignStep())
    workflow.add_step(FineAlignStep())
    workflow.add_step(FinalPage())
    workflow.initialize()
    return workflow


if __name__ == '__main__':
    filepaths = {
        'static': {'sample': 'test_data/-4-H.dm3',
                   'reference': 'test_data/-4-R2.dm3'},
        'moving': {'sample': 'test_data/+4-H.dm3',
                   'reference': 'test_data/+4-R2.dm3'},
    }
    hdf5_path = './test_data/testpath.h5'

    get_workflow(filepaths=filepaths, hdf5_path=hdf5_path).show()


