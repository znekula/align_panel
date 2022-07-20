import panel as pn
import numpy as np
import aperture as ap

from .imgsetlib import Imgset, H5file
from .align_panel import array_format, point_registration, fine_adjust


def autoalign_params_panel(static_imgset, moving_imgset, layout=None):
    if layout is None:
        layout = ap.layout(mode='simple')

    bins_int_input = pn.widgets.IntInput(name='Bins',
                                         value=62,
                                         start=1)
    del_back_checkbox = pn.widgets.Checkbox(name='Remove background',
                                            value=True)

    method_choice = pn.widgets.Select(name='Align method',
                                      options=Imgset.autoalign_methods())

    param_widgets = {'transformation': method_choice,
                     'bins': bins_int_input,
                     'del_back': del_back_checkbox}

    params_card = pn.layout.Card(title='Autoalign parameters',
                                 collapsible=False)
    layout.panes[0].add_element_group('params',
                                      param_widgets,
                                      container=params_card)

    def param_getter():
        return {k: v.value for k, v in param_widgets.items()}

    return layout, param_getter


autoalign_message = '''
- Configure autoalignment parameters
- Press 'Run' to do alignment (this can take a while!)
- Inspect the alignment using the alpha slider / plot tools
- Save the transformation if satisfactory

**Note:** the autoalignment is done on the original images.
Any existing transformation matrix will effectively be reset.
'''


def autoalign_panel(static_imgset: Imgset, moving_imgset: Imgset, img_key):
    layout = ap.layout(mode='simple')

    layout, param_getter = autoalign_params_panel(static_imgset,
                                                  moving_imgset,
                                                  layout=layout)

    static_image = static_imgset.get_data(img_key, aligned=True)
    moving_image = moving_imgset.get_data(img_key, aligned=True)

    fig = ap.figure()
    static_img = fig.add_image(array=static_image, name='Static')
    static_img.add_colorbar()
    overlay = fig.add_image(array=moving_image, name='Moving')
    overlay_alpha_slider = overlay.get_alpha_slider(0.5,
                                                    name='Moving alpha',
                                                    max_width=250)
    layout.panes[1].append(overlay_alpha_slider)
    layout.panes[1].append(fig)

    spinner = pn.indicators.LoadingSpinner(height=40, width=40, align='end')
    transform_def = {}

    def _run_auto_align(*event):
        spinner.value = True
        run_button.disabled = True
        pn.io.push_notebook(spinner, run_button)

        params = param_getter()
        static_image_raw = static_imgset.get_data(img_key)
        moving_image_raw = moving_imgset.get_data(img_key)
        tmat = moving_imgset.autoalign(static_image_raw, moving_image_raw, **params)
        transform_def['transform'] = tmat
        overlay.raw_image = moving_imgset.apply_tmat(img_key, tmat=tmat)

        spinner.value = False
        run_button.disabled = False
        pn.io.push_notebook(spinner, run_button, fig)

    run_button = pn.widgets.Button(name='Run auto-alignment',
                                   max_width=150,
                                   height=40,
                                   button_type='primary')
    run_button.on_click(_run_auto_align)
    layout.panes[0].append(pn.Row(run_button, spinner))
    layout.panes[0].append(fig.get_toolbox(name='Image toolbox'))

    def res_getter():
        return transform_def

    return layout, res_getter


pointsalign_message = '''
- Draw and position points on the static image
- Move the points on the moving image to the equivalent positions
- Choose a transform type and press run to compute a transform
- Inspect the transform with the Overlay and Difference images
- Save the transformation if satisfactory
'''

def pointsalign_panel(static_imgset, moving_imgset, img_key):
    static_image = static_imgset.get_data(img_key, aligned=True)
    moving_image = moving_imgset.get_data(img_key, aligned=True)
    lo, getter = point_registration(static_image, moving_image)

    def _getter():
        params = getter()
        incremental = params['transform'].params
        current = moving_imgset.get_tmat()
        params['transform'] = current @ incremental
        return params

    return lo, _getter


finealign_message = '''
- Use the controls to adjust the moving image position and scale
- Use the free draw tool to drag the moving image directly
- Steps can be remove using the Undo button
- Inspect the transform by adjusting the alpha
- Save the transformation if satisfactory
'''

def finealign_panel(static_imgset, moving_imgset, img_key):
    static_image = static_imgset.get_data(img_key, aligned=True)
    moving_image = moving_imgset.get_data(img_key, aligned=True)
    lo, getter = fine_adjust(static_image, moving_image)

    def _getter():
        params = getter()
        incremental = params['transform'].params
        current = moving_imgset.get_tmat()
        params['transform'] = current @ incremental
        return params

    return lo, _getter


def resize_panel(static_imgset, moving_imgset: Imgset, img_key):
    return pn.Column(pn.Spacer()), lambda: {}


home_message = r'''
This GUI panel provides several tools for aligning image sets in HDF5 format.
<br><br>

Each image set contains several images, which share the same coordinate
system
```
e.g. amplitude and phase are two versions of the 'same' image
```
<br>
The 'Align-on' dropdown allows you to choose which sub-image to display.

- The alignment done is computed from a 'Moving' image set onto a 'Static'
image set.
- To load a different set of images as static/moving you **must
re-execute the notebook cell**.

Each page computes a transformation matrix which maps from 'moving'
onto 'static'.

- All progress is stored in the file itself
- To commit the actions of a page you
**must press the 'Save transformation' button** which will store it to a file

As of this time there is no 'Undo' action, though on this page you can fully
reset the transformation matrix for the chosen pair of image sets.
'''


def current_transform_panel(static_imgset, moving_imgset: Imgset, img_key):
    dataset_md = pn.pane.Markdown(object=f'''
<br>
**Static imgset**: {static_imgset.imgset_name}

**Moving imgset**: {moving_imgset.imgset_name}
''')

    tmat = moving_imgset.get_tmat()
    matrix_header = 'Current transform:'
    display_md = pn.pane.Markdown(object=array_format(tmat, header=matrix_header))
    reset_btn = pn.widgets.Button(name='Reset transform', button_type='success', max_width=200)

    def _reset_transform(*event):
        moving_imgset.clear_tmat()
        display_md.object = array_format(moving_imgset.get_tmat(), header=matrix_header)
        pn.io.push_notebook(display_md)

    reset_btn.on_click(_reset_transform)

    return pn.Row(dataset_md, pn.Column(display_md, reset_btn)), lambda: {}


align_panes = {
    'Home': current_transform_panel,
    'Resize / Crop': resize_panel,
    'Auto-align': autoalign_panel,
    'Points': pointsalign_panel,
    'Fine-adjust': finealign_panel,
}
pane_messages = {
    'Home': home_message,
    'Resize / Crop': 'Not implemented yet',
    'Auto-align': autoalign_message,
    'Points': pointsalign_message,
    'Fine-adjust': finealign_message,
}


def get_header(key):
    pane_message = pane_messages.get(key, '')
    if pane_message:
        pane_message = '\n' + pane_message
    return f'### {key}' + pane_message


def align_cell(static_imgset, moving_imgset):
    options = [*align_panes.keys()]
    init_value = options[0]

    static_arrays = set(static_imgset.get_2d_image_keys())
    moving_arrays = set(moving_imgset.get_2d_image_keys())
    img_options = list(static_arrays.intersection(moving_arrays))
    if 'unwrapped_phase' in img_options:
        img_options.insert(0, img_options.pop(img_options.index('unwrapped_phase')))
    image_type = pn.widgets.Select(name='Align-on',
                                   options=img_options,
                                   max_width=200)
    init_layout, res_getter = align_panes[init_value](static_imgset,
                                                      moving_imgset,
                                                      image_type.value)
    pane_select = pn.widgets.Select(name='Page', options=[*align_panes.keys()], max_width=200)
    load_btn = pn.widgets.Button(name='Load',
                                 max_width=150,
                                 button_type='primary',
                                 align='end')
    save_btn = pn.widgets.Button(name='Save transformation',
                                 max_width=150,
                                 button_type='success',
                                 align='end')

    spinner = pn.indicators.LoadingSpinner(height=40, width=40, align='end')
    default_message = 'Not yet saved'
    message_text = pn.widgets.StaticText(value=default_message, align='end')
    title_md = pn.pane.Markdown(object=get_header(init_value), max_width=700)
    indicator_row = pn.Row(load_btn, pane_select, spinner, image_type, save_btn, message_text)
    panel_layout = pn.Column(init_layout)
    layout = pn.Column(indicator_row,
                       title_md,
                       panel_layout)

    def _save_tmat(*event):
        try:
            transform_meta = res_getter()
            try:
                tmat = transform_meta['transform'].params
            except AttributeError:
                tmat = transform_meta['transform']
            assert isinstance(tmat, np.ndarray)
            assert tmat.shape == (3, 3)
            moving_imgset.save_tmat(tmat)
            message_text.value = 'Saved transform'
            pn.io.push_notebook(message_text)
        except KeyError:
            message_text.value = 'No transform to save'
            pn.io.push_notebook(message_text)

    def _load_pane(event):
        nonlocal res_getter
        to_load = pane_select.value

        load_btn.loading = True
        save_btn.loading = True
        spinner.value = True
        message_text.value = 'Loading...'
        title_md.object = get_header(to_load)
        pn.io.push_notebook(load_btn, save_btn, spinner, message_text, title_md)

        panel_layout.clear()
        img_key = image_type.value
        new_layout, res_getter = align_panes[to_load](static_imgset, moving_imgset, img_key)
        panel_layout.append(new_layout)
        pn.io.push_notebook(panel_layout)

        load_btn.loading = False
        save_btn.loading = False
        spinner.value = False
        message_text.value = default_message
        pn.io.push_notebook(load_btn, save_btn, spinner, message_text)

    load_btn.on_click(_load_pane)
    save_btn.on_click(_save_tmat)
    return layout


def imgset_select_panel(data_path, do_display=True):
    file_meta = H5file(data_path)

    static_select = pn.widgets.Select(name='Static image', options=file_meta.ref_imageset_name)
    moving_select = pn.widgets.Select(name='Moving image', options=file_meta.imageset_names)
    lo = pn.Row(static_select, moving_select)

    def get_imgsets():
        static_imgset = Imgset(data_path, static_select.value)
        moving_imgset = Imgset(data_path, moving_select.value)
        return {'static_imgset': static_imgset, 'moving_imgset': moving_imgset}

    if do_display:
        from IPython.display import display as ipydisplay
        ipydisplay(lo)

    return lo, get_imgsets
