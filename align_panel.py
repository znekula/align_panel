import functools
import itertools
import numpy as np
import skimage.transform as sktransform
import panel as pn

from libertem_ui.display.figure import BokehFigure
from libertem_ui.display.colormaps import get_bokeh_palette
from libertem_ui.layout.auto import TwoPane

from image_transformer import ImageTransformer


available_transforms = ['affine', 'euclidean', 'similarity', 'projective']


def get_base_figure(array: np.ndarray, name: str):
    figure = BokehFigure()
    figure.set_title(name)
    image = figure.add_image(array=array)
    toolbox = image.get_image_toolbox(name=f'{name} toolbox')
    return figure, image, toolbox
    

def get_pointset(figure: BokehFigure, with_add=False):
    pointset = figure.add_pointset(data={'cx': [],
                                         'cy': [],
                                         'color': [],
                                         'ix': []}, allow_edit=True, fill_color='color')
    if with_add:
        point_add = figure.add_free_point(pointset=pointset)
        figure.set_active_tool(point_add.tool_label())
    else:
        figure.set_active_tool(pointset.tool_label())
    return pointset
    

def compute_transform(static_points, moving_points, method='affine'):
    assert method in available_transforms
    return sktransform.estimate_transform(method,
                                          static_points.reshape(-1, 2),
                                          moving_points.reshape(-1, 2))


def compute_image(array, transform, output_shape=None, order=None):
    return sktransform.warp(array,
                            transform,
                            output_shape=output_shape,
                            mode='constant',
                            order=order,
                            cval=np.nan,
                            clip=True,
                            preserve_range=True)
    


def point_registration(static: np.ndarray, moving: np.ndarray):
    static_fig, static_im, static_toolbox = get_base_figure(static, 'Static')
    overlay_image = static_fig.add_image(array=moving)
    alpha_slider = overlay_image.get_alpha_slider(name='Overlay alpha', alpha=0., max_width=200)
    static_pointset = get_pointset(static_fig, with_add=True)
    
    moving_fig, moving_im, moving_toolbox = get_base_figure(moving, 'Moving')
    moving_pointset = get_pointset(moving_fig)
    
    color_iterator = itertools.cycle(get_bokeh_palette())
    
    def _copy_new_points(attr, old, new):
        source_cds = static_pointset.cds
        target_cds = moving_pointset.cds
        target_length = len(target_cds.data['cx'])
        source_length = len(new['cx'])
        if target_length == source_length:
            return
        elif target_length > source_length:
            raise RuntimeError('De-sync data sources ??')
        to_add = source_length - target_length
        new_colors = [next(color_iterator) for _ in range(to_add)]
        max_ix = max(0, max(new['ix']))
        new_indexes = [ix for ix in range(max_ix, max_ix + to_add)]
        source_update = {'color': [(slice(source_length-to_add, source_length), new_colors)],
                         'ix': [(slice(source_length-to_add, source_length), new_indexes)]}
        source_cds.patch(source_update)
        update_data = {k: new[k][-to_add:] for k in target_cds.column_names}
        target_cds.stream(update_data)
    
    static_pointset.cds.on_change('data', _copy_new_points)    
    
    transformations = {s.title(): s for s in available_transforms}
    method_select = pn.widgets.Select(name='Transformation type',
                                      options=[*transformations.keys()],
                                      max_width=250)
    run_button = pn.widgets.Button(name='Run',
                                   button_type='primary',
                                   max_width=150,
                                   align='end')
    output_md = pn.pane.Markdown(object='No transform defined',
                                 width=600)
    transform = None
    async def _compute_transform(event):
        nonlocal transform 

        method = transformations[method_select.value]
        static_points = static_pointset.export_data()[['cx', 'cy']].to_numpy().reshape(-1, 2)
        moving_points = moving_pointset.export_data()[['cx', 'cy']].to_numpy().reshape(-1, 2)
        ns = static_points.shape[0]
        nm = moving_points.shape[0]
        if ns != nm:
            output_md.object = f'Mismatching number of points - static: {ns}, moving: {nm}'
            return
        elif ns == 0:
            output_md.object = f'No points defined'
            return
        try:
            transform = compute_transform(static_points, moving_points, method=method)
        except Exception as e:
            output_md.object = f'Error computing transform: {str(e)}'
            return
        try:
            output_md.object = f'{np.array2string(transform.params, precision=2)}'
        except AttributeError:
            output_md.object = f'Unrecognized transform'
            return
            
        warped_moving = compute_image(moving,
                                      transform,
                                      output_shape=static.shape)
        overlay_image.update_raw_image(warped_moving)
        static_fig.refresh_pane()
    
    run_button.on_click(_compute_transform)    
    
    layout = TwoPane()
    layout.add_title('Point-based image registration')
    layout.header[0].width = 600
    layout.first.append(static_fig)
    layout.first.append(pn.Row(static_toolbox, alpha_slider))
    layout.second.append(moving_fig)
    layout.second.append(moving_toolbox)
    layout.first.append(pn.Row(method_select, run_button))    
    layout.second.append(output_md)
    
    layout.finalize()
    layout.body.make_row()
    
    
    def getter():
        export_keys = ['cx', 'cy', 'ix']
        return {'static_points': static_pointset.export_data()[export_keys],
                'moving_points': moving_pointset.export_data()[export_keys],
                'transform': transform}
    
    return layout, getter


def fine_adjust(static, moving, initial_transform=None):
    transformer_moving = ImageTransformer(moving)
    if initial_transform:
        transformer_moving.add_transform(initial_transform, output_shape=static.shape)
    else:
        # To be sure we set the output shape to match static
        transformer_moving.add_null_transform(output_shape=static.shape)

    static_name = 'Static'
    moving_name = 'Moving'

    fig = BokehFigure()
    static_im = fig.add_image(array=static)
    moving_im = fig.add_image(array=transformer_moving.get_transformed_image(cval=np.nan))
    static_im.change_cmap('Blues')
    static_im.set_nan_transparent()
    moving_im.set_color_mapper(palette='Reds')
    moving_im.set_nan_transparent()
    static_im.add_colorbar(title=static_name)
    moving_im.add_colorbar(title=moving_name)

    static_alpha = static_im.get_alpha_slider(name=f'{static_name} alpha', alpha=0.5)
    static_cmap = static_im.get_cbar_select(title=f'{static_name} colormap')
    static_clims = static_im.get_cbar_slider(title=f'{static_name} contrast')
    static_invert = static_im.get_invert_cmap_box(name='Invert')

    overlay_alpha = moving_im.get_alpha_slider(name=f'{moving_name} alpha', alpha=0.5)        
    overlay_cmap = moving_im.get_cbar_select(title=f'{moving_name} colormap')
    overlay_clims = moving_im.get_cbar_slider(title=f'{moving_name} contrast')
    overlay_invert = moving_im.get_invert_cmap_box(name='Invert')

    translate_step_input = pn.widgets.FloatInput(name='Translate step (px):',
                                                 value=1.,
                                                 start=0.1,
                                                 end=100.,
                                                 width=125)
    
    def update_moving_sync():
        moving_im.update_raw_image(transformer_moving.get_transformed_image(cval=np.nan))
        fig.refresh_pane()

    async def update_moving():
        update_moving_sync()

    async def fine_translate(event, x=0, y=0):
        if not x and not y:
            print('No translate requested')
            return
        raw_adjust = translate_step_input.value
        anchor_shift = {'xshift': -1 * x * raw_adjust, 'yshift': -1 * y * raw_adjust}
        with transformer_moving.group_transforms(key=moving_key):
            transformer_moving.translate(**anchor_shift)
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
                                              value=1.,
                                              start=0.1,
                                              end=100.,
                                              width=125)

    async def fine_rotate(event, dir=0):
        if not dir:
            print('No rotate requested')
            return
        about_center = about_center_cbox.value
        true_rotate = -1 * rotate_step_input.value * dir
        with transformer_moving.group_transforms(key=moving_key):
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
        with transformer_moving.group_transforms(key=moving_key):
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

        with transformer_moving.group_transforms(key=moving_key):
            transformer_moving.translate(xshift=xshift, yshift=yshift)
        
        update_moving_sync()

    fig.add_free_callback(callback=translate_from_path)

    def getter():
        return {'transformer': transformer_moving}


    return pn.Column(pn.Row(static_alpha, overlay_alpha),
                     pn.Row(fig, pn.Column(translate_step_input,
                                           translate_buttons(fine_translate),
                                           about_center_cbox,
                                           rotate_step_input,
                                           rotate_buttons(fine_rotate),
                                           scale_step_input,
                                           scale_buttons(fine_scale),
                                          )),
                     pn.Row(static_cmap, static_clims, static_invert),
                     pn.Row(overlay_cmap, overlay_clims, overlay_invert)), getter


def translate_buttons(cb):
    """
    Configured for y-axis pointing down!!
    """
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    left = pn.widgets.Button(name='\u25C1', **kwargs)
    left.on_click(functools.partial(cb, x=-1))
    up = pn.widgets.Button(name='\u25B3', **kwargs)
    up.on_click(functools.partial(cb, y=-1))
    right = pn.widgets.Button(name='\u25B7', **kwargs)
    right.on_click(functools.partial(cb, x=1))
    down = pn.widgets.Button(name='\u25BD', **kwargs)
    down.on_click(functools.partial(cb, y=1))
    lo = pn.Column(pn.Row(sp, up, sp, margin=(0, 0)),
                   pn.Row(left, sp, right, margin=(0, 0)),
                   pn.Row(sp, down, sp, margin=(0, 0)), margin=(0, 0))
    return lo


def rotate_buttons(cb):
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width // 3, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    acw_btn = pn.widgets.Button(name='\u21B6', **kwargs)
    acw_btn.on_click(functools.partial(cb, dir=-1))
    cw_btn = pn.widgets.Button(name='\u21B7', **kwargs)
    cw_btn.on_click(functools.partial(cb, dir=1))
    return pn.Row(acw_btn, sp, cw_btn, margin=(0, 0))


def scale_buttons(cb):
    width = height = 40
    margin = (2, 2)
    sp = pn.Spacer(width=width, height=height, margin=margin)
    kwargs = {'width': width, 'height': height, 'margin': margin, 'button_type': 'primary'}
    x_compress = pn.widgets.Button(name='\u2B72', **kwargs)
    x_compress.on_click(functools.partial(cb, xdir=-1))
    y_compress = pn.widgets.Button(name='\u2B73', **kwargs)
    y_compress.on_click(functools.partial(cb, ydir=-1))
    x_expand = pn.widgets.Button(name='\u21D4', **kwargs)
    x_expand.on_click(functools.partial(cb, xdir=1))
    y_expand = pn.widgets.Button(name='\u21D5', **kwargs)
    y_expand.on_click(functools.partial(cb, ydir=1))
    xy_expand = pn.widgets.Button(name='\u2922', **kwargs)
    xy_expand.on_click(functools.partial(cb, ydir=1, xdir=1))  
    xy_compress = pn.widgets.Button(name='\u2B79', **kwargs)
    xy_compress.on_click(functools.partial(cb, ydir=-1, xdir=-1))
    lo = pn.Column(pn.Row(sp, y_compress, xy_expand, margin=(0, 0)),
                   pn.Row(x_compress, sp, x_expand, margin=(0, 0)),
                   pn.Row(xy_compress, y_expand, sp, margin=(0, 0)), margin=(0, 0))
    return lo