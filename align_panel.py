import functools
import itertools
from typing import Optional, TYPE_CHECKING
import numpy as np
import panel as pn

from aperture.display.figure import BokehFigure
from aperture.display.utils.colormaps import get_bokeh_palette
from aperture.layout.panes import SimpleTwoPane as TwoPane
from aperture.layout.panes import Panes

from image_transformer import ImageTransformer

from bokeh.plotting import figure

if TYPE_CHECKING:
    import pandas as pd
    import skimage.transform as sktransform


def get_base_figure(array: np.ndarray, name: str):
    """
    Get a figure with an Image glyph of array
    Set the title to name and return the colormap toolbox for the image
    """
    figure = BokehFigure()
    figure.set_title(name)
    image = figure.add_image(array=array)
    toolbox = image.get_image_toolbox(name=f'{name} toolbox')
    return figure, image, toolbox
    

def get_joint_pointset(static_figure: BokehFigure, moving_figure: BokehFigure, initial_points=None):
    """
    Place one scatter plot on each figure, and set up callbacks such that
    points created or deleted on one or the other are mirrored in the other

    This is currently a manual process, particular auto-filling values
    such as the color for each point. The underlying library doesn't really
    support this yet and will be upgraded to do so in the future.
    """
    color_iterator = itertools.cycle(get_bokeh_palette())
    
    defaults = {'cx': -10000,
                'cy': -10000,
                'moving_cx': -10000,
                'moving_cy': -10000,
                'color': '#000000'}

    if initial_points is not None:
        initial_points['color'] = [next(color_iterator) for _ in range(len(initial_points.index))]
        initial_data = initial_points.reset_index().to_dict(orient='list')
        initial_data = {k: v for k, v in initial_data.items() if k in defaults.keys()}
    else:
        initial_data = {k: [] for k in defaults.keys()}

    static_pointset = static_figure.add_pointset(fill_values={**defaults, '_propagate': False},
                                                 data=initial_data,
                                                 allow_edit=True,
                                                 fill_color='color')
    moving_pointset = moving_figure.add_pointset(fill_values={**defaults, '_propagate': False},
                                                 source=static_pointset.cds,
                                                 keys=('moving_cy', 'moving_cx'),
                                                 allow_edit=True,
                                                 fill_color='color')
    point_add = static_figure.add_free_point(pointset=static_pointset)
    moving_figure.add_free_point(pointset=moving_pointset)
    static_figure.set_active_tool(point_add.tool_label())
    moving_figure.set_active_tool(moving_pointset.tool_label())

    default_fill = -1

    def _sync_points(attr, old, new):
        # The synchronization callback, will be called each time
        # the data source is changed / modified so we first
        # exit early if the source length hasn't changed !
        if not new['cx'] or len(old['cx']) == len(new['cx']):
            return
        # Use the color as a proxy to recognize which points are new
        to_patch_ix = [i for i, c in enumerate(new['color']) if c in [defaults['color'], default_fill]]
        # Exit early if no points are new, this is the case for point deletion
        if not to_patch_ix:
            return

        patches = {'color': [(i, next(color_iterator)) for i in to_patch_ix]}
        cx_patches = 'cx', [(i, new['moving_cx'][i]) for i in to_patch_ix if new['cx'][i] in [defaults['cx'], default_fill]]
        cy_patches = 'cy', [(i, new['moving_cy'][i]) for i in to_patch_ix if new['cy'][i] in [defaults['cy'], default_fill]]
        moving_cx_patches = 'moving_cx', [(i, new['cx'][i]) for i in to_patch_ix if new['moving_cx'][i] in [defaults['moving_cx'], default_fill]]
        moving_cy_patches = 'moving_cy', [(i, new['cy'][i]) for i in to_patch_ix if new['moving_cy'][i] in [defaults['moving_cy'], default_fill]]
        valid_point_patches = {k: v for k, v in [cx_patches, cy_patches, moving_cx_patches, moving_cy_patches] if v}
        patches.update(valid_point_patches)
        static_pointset.cds.patch(patches)
    
    static_pointset.cds.on_change('data', _sync_points)

    return static_pointset, moving_pointset
    

def array_format(array: np.ndarray):
    """
    Format a 3x3 array nicely as a Markdown string
    This is quite hacky, can be much improved
    """
    assert array.shape == (3, 3)
    str_array = np.array2string(array,
                                precision=2,
                                suppress_small=True,
                                sign=' ',
                                floatmode='fixed')
    substrings = str_array.split('\n')
    return f'''
Transformation matrix:
```
{substrings[0]}  
{substrings[1]}  
{substrings[2]}
```'''


def assure_size(array: np.ndarray, target_shape: tuple[int, int]):
    """
    Inserts array into another array of size target_shape
    Will either crop or zero-pad as necessary
    Array is inserted from the top-left
    """
    h, w = array.shape
    th, tw = target_shape
    ph = min(h, th)
    pw = min(w, tw)
    canvas = np.zeros(target_shape, dtype=array.dtype)
    canvas[:ph, :pw] = array[:ph, :pw]
    return canvas


def point_registration(static: np.ndarray, moving: np.ndarray, initial_points: Optional['pd.DataFrame'] = None):
    """
    Provides a UI panel for pointset-to-pointset image registration
    from the moving image onto the static image
    Initial points can be supplied as a Pandas DataFrame
    with the following 4 columns:
        - `cx`, `cy` for points in the static image,
        - `moving_cx`, `moving_cy` for corresponding points in the moving image.
    """
    transformer_moving = ImageTransformer(moving)

    static_fig, static_im, static_toolbox = get_base_figure(static, 'Static')
    overlay_image = static_fig.add_image(array=assure_size(moving, static.shape))
    alpha_slider = overlay_image.get_alpha_slider(name='Overlay alpha', alpha=0., max_width=200)
    moving_fig, moving_im, moving_toolbox = get_base_figure(moving, 'Moving')
    static_pointset, moving_pointset = get_joint_pointset(static_fig, moving_fig, initial_points=initial_points)
       
    transformations = {s.title(): s for s in ImageTransformer.available_transforms()}
    method_select = pn.widgets.Select(name='Transformation type',
                                      options=[*transformations.keys()],
                                      max_width=200)
    run_button = pn.widgets.Button(name='Run',
                                   button_type='primary',
                                   max_width=150,
                                   align='end')
    output_md = pn.pane.Markdown(object='No transform defined',
                                 width=450)
    clear_button = pn.widgets.Button(name='Clear points',
                                     max_width=150,
                                     align='end')

    dif_im = static - transformer_moving.get_transformed_image(output_shape=static.shape)
    zero_fig, zero_im, zero_toolbox = get_base_figure(static*0, 'Difference')
    overlay_image_dif = zero_fig.add_image(array=assure_size(dif_im, static.shape))
    
    async def _clear(event):
        static_pointset.clear_data()
        # static_fig.refresh_pane()
        # moving_fig.refresh_pane()

    clear_button.on_click(_clear)

    transform_points = None
    async def _compute_transform(event):
        nonlocal transform_points

        method = transformations[method_select.value]
        static_points = static_pointset.export_data()[['cx', 'cy']].to_numpy().reshape(-1, 2)
        moving_points = moving_pointset.export_data()[['moving_cx', 'moving_cy']].to_numpy().reshape(-1, 2)
        transform_points = static_pointset.export_data()
        ns = static_points.shape[0]
        nm = moving_points.shape[0]
        if ns != nm:
            output_md.object = f'Mismatching number of points - static: {ns}, moving: {nm}'
            return
        elif ns == 0:
            output_md.object = f'No points defined'
            return
        try:
            transform = transformer_moving.estimate_transform(static_points,
                                                              moving_points,
                                                              method=method,
                                                              clear=True)
        except Exception as e:
            output_md.object = f'Error computing transform: {str(e)}'
            return
        try:
            output_md.object = array_format(transform.params)
        except Exception as e:
            output_md.object = f'Post-transform error (format?)'
            return
            
        warped_moving = transformer_moving.get_transformed_image(output_shape=static.shape)
        overlay_image.update_raw_image(warped_moving, fix_clims=True)#change
        static_fig.refresh_pane()

        overlay_image_dif.update_raw_image(static - warped_moving, fix_clims=True)#change
        zero_fig.refresh_pane()

    run_button.on_click(_compute_transform)    

    layout = Panes()
    layout.panes[0].append(static_fig)
    layout.panes[0].append(pn.Row(static_toolbox, alpha_slider))
    layout.panes[0].append(method_select)
    layout.panes[0].append(pn.Row(run_button, clear_button))

    layout.panes[1].append(moving_fig)
    layout.panes[1].append(moving_toolbox)
    layout.panes[1].append(output_md)

    layout.panes[2].append(zero_fig)
    
    layout.finalize()
    
    def getter():
        return {'points': transform_points,
                'transform': transformer_moving.get_combined_transform()}
    
    return layout, getter


def fine_adjust(static: np.ndarray, moving: np.ndarray,
                initial_transform: Optional['sktransform.AffineTransform'] = None):
    """
    Provides a UI panel to manually align the image moving onto static
    Optionally provide a skimage.transform.GeometricTransform object
    to pre-transform moving
    """
    transformer_moving = ImageTransformer(moving)
    if initial_transform:
        transformer_moving.add_transform(initial_transform, output_shape=static.shape)
    else:
        # To be sure we set the output shape to match static
        transformer_moving.add_null_transform(output_shape=static.shape)

    static_name = 'Static'
    moving_name = 'Difference'

    fig = BokehFigure()
    static_im = fig.add_image(array=static)
    #moving_im = fig.add_image(array=transformer_moving.get_transformed_image(cval=np.nan))
    #this is my change
    diff = static - transformer_moving.get_transformed_image(cval=np.nan)
    moving_im = fig.add_image(array=diff)

    static_im.change_cmap('Blues')
    static_im.set_nan_transparent()
    moving_im.set_color_mapper(palette='Spectra')
    moving_im.set_nan_transparent()
    static_im.add_colorbar(title=static_name)
    moving_im.add_colorbar(title=moving_name)

    static_alpha = static_im.get_alpha_slider(name=f'{static_name} alpha', alpha=0.0)
    static_cmap = static_im.get_cbar_select(title=f'{static_name} colormap')
    static_clims = static_im.get_cbar_slider(title=f'{static_name} contrast')
    static_invert = static_im.get_invert_cmap_box(name='Invert')

    overlay_alpha = moving_im.get_alpha_slider(name=f'{moving_name} alpha', alpha=1)        
    overlay_cmap = moving_im.get_cbar_select(title=f'{moving_name} colormap')
    overlay_clims = moving_im.get_cbar_slider(title=f'{moving_name} contrast')
    overlay_invert = moving_im.get_invert_cmap_box(name='Invert')

    translate_step_input = pn.widgets.FloatInput(name='Translate step (px):',
                                                 value=1.,
                                                 start=0.1,
                                                 end=100.,
                                                 width=125)
    
    def update_moving_sync():
        moving_im.update_raw_image(static - transformer_moving.get_transformed_image(), fix_clims=True)
        fig.refresh_pane()

    async def update_moving():
        update_moving_sync()

    async def fine_translate(event, x=0, y=0):
        if not x and not y:
            print('No translate requested')
            return
        raw_adjust = -1 * translate_step_input.value
        transformer_moving.translate(xshift=x * raw_adjust, yshift=y * raw_adjust)
        await update_moving()

    origin_cursor = fig.add_cursor(image=static_im,
                                   line_color='cyan',
                                   line_alpha=0.)
    fig.set_active_tool(origin_cursor.tool_label())
    about_center_cbox = pn.widgets.Checkbox(name='Center-origin',
                                            value=True,
                                            width=125)

    def _set_cursor_alpha(event):
        origin_cursor._cursor_glyph.line_alpha = float(not event.new)

    about_center_cbox.param.watch(_set_cursor_alpha, 'value')

    rotate_step_input = pn.widgets.FloatInput(name='Rotate step (deg):',
                                              value=5.,
                                              start=0.1,
                                              end=100.,
                                              width=125)

    async def fine_rotate(event, dir=0):
        if not dir:
            print('No rotate requested')
            return
        about_center = about_center_cbox.value
        true_rotate = -1 * rotate_step_input.value * dir
        if about_center:
            transformer_moving.rotate_about_center(rotation_degrees=true_rotate)
        else:
            cx, cy = origin_cursor.get_posxy()
            transformer_moving.rotate_about_point((cy, cx), rotation_degrees=true_rotate)
        await update_moving()


    scale_step_input = pn.widgets.FloatInput(name='Scale step (%):',
                                             value=1.,
                                             start=0.1,
                                             end=100.,
                                             width=125)

    async def fine_scale(event, xdir=0, ydir=0):
        if not xdir and not ydir:
            print('No scaling requested')
            return
        about_center = about_center_cbox.value
        xscale = 1 - (scale_step_input.value * xdir / 100)
        yscale = 1 - (scale_step_input.value * ydir / 100)
        if about_center:
            transformer_moving.xy_scale_about_center(xscale=xscale, yscale=yscale)
        else:
            cx, cy = origin_cursor.get_posxy()
            transformer_moving.xy_scale_about_point((cy, cx), xscale=xscale, yscale=yscale)
        await update_moving()


    def translate_from_path(path_dict):
        xs = path_dict['xs']
        ys = path_dict['ys']

        xshift = -1 * (xs[-1] - xs[0])
        yshift = -1 * (ys[-1] - ys[0])
        transformer_moving.translate(xshift=xshift, yshift=yshift)
        update_moving_sync()

    fig.add_free_callback(callback=translate_from_path)

    async def _undo(event):
        transformer_moving.remove_transform()
        await update_moving()

    undo_button = pn.widgets.Button(name='Undo',
                                    max_width=125,
                                    button_type='primary')
    undo_button.on_click(_undo)

    def getter():
        return {'transform': transformer_moving.get_combined_transform()}


    return pn.Column(pn.Row(static_alpha, overlay_alpha),
                     pn.Row(fig, pn.Column(translate_step_input,
                                           translate_buttons(fine_translate),
                                           about_center_cbox,
                                           rotate_step_input,
                                           rotate_buttons(fine_rotate),
                                           scale_step_input,
                                           scale_buttons(fine_scale),
                                           pn.Spacer(width=40, height=40),
                                           undo_button,
                                          )),
                     pn.Row(static_cmap, static_clims, static_invert),
                     pn.Row(overlay_cmap, overlay_clims, overlay_invert)), getter


# Unicode arrow codes used for defining UI buttons
LEFT_ARROW = '\u25C1'
UP_ARROW = '\u25B3'
RIGHT_ARROW = '\u25B7'
DOWN_ARROW = '\u25BD'
ROTATE_RIGHT_ARROW = '\u21B7'
ROTATE_LEFT_ARROW = '\u21B6'


def translate_buttons(cb):
    """
    A button array for up/down/left/right
    Configured for y-axis pointing down!!
    """
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    left = pn.widgets.Button(name=LEFT_ARROW, **kwargs)
    left.on_click(functools.partial(cb, x=-1))
    up = pn.widgets.Button(name=UP_ARROW, **kwargs)
    up.on_click(functools.partial(cb, y=-1))
    right = pn.widgets.Button(name=RIGHT_ARROW, **kwargs)
    right.on_click(functools.partial(cb, x=1))
    down = pn.widgets.Button(name=DOWN_ARROW, **kwargs)
    down.on_click(functools.partial(cb, y=1))
    lo = pn.Column(pn.Row(sp, up, sp, margin=(0, 0)),
                   pn.Row(left, sp, right, margin=(0, 0)),
                   pn.Row(sp, down, sp, margin=(0, 0)), margin=(0, 0))
    return lo


def rotate_buttons(cb):
    """A button array for rotate acw / cw"""
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    acw_btn = pn.widgets.Button(name=ROTATE_LEFT_ARROW, **kwargs)
    acw_btn.on_click(functools.partial(cb, dir=-1))
    cw_btn = pn.widgets.Button(name=ROTATE_RIGHT_ARROW, **kwargs)
    cw_btn.on_click(functools.partial(cb, dir=1))
    return pn.Row(sp, acw_btn, cw_btn, margin=(0, 0))


def scale_buttons(cb):
    """A button array for scaling x / y / xy up and down"""
    width = height = 40
    margin = (2, 2)
    text_kwargs = {'width': width // 2,
                   'height': height // 2,
                   'margin': margin,
                   'align': ('end', 'center')}
    button_kwargs = {'width': width,
                     'height': height,
                     'margin': margin,
                     'button_type': 'primary'}
    x_row = up_down_pair('X:',
                         cb,
                         {'xdir': 1},
                         {'xdir': -1},
                         text_kwargs,
                         button_kwargs)
    y_row = up_down_pair('Y:',
                         cb,
                         {'ydir': 1},
                         {'ydir': -1},
                         text_kwargs,
                         button_kwargs)
    xy_row = up_down_pair('XY:',
                          cb,
                          {'xdir': 1, 'ydir': 1},
                          {'xdir': -1, 'ydir': -1},
                          text_kwargs,
                          button_kwargs)
    lo = pn.Column(x_row,
                   y_row,
                   xy_row, margin=(0, 0))
    return lo


def up_down_pair(name, cb, upkwargs, downkwargs, text_kwargs, button_kwargs):
    sp = pn.Spacer(**text_kwargs)
    text = pn.widgets.StaticText(value=name, **text_kwargs)
    compress = pn.widgets.Button(name=f'{RIGHT_ARROW} {LEFT_ARROW}', **button_kwargs)
    compress.on_click(functools.partial(cb, **downkwargs))
    expand = pn.widgets.Button(name=f'{LEFT_ARROW} {RIGHT_ARROW}', **button_kwargs)
    expand.on_click(functools.partial(cb, **upkwargs))
    return pn.Row(sp, text, compress, expand, margin=(0, 0))
