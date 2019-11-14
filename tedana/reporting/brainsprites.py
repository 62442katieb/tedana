import json
from os.path import join as opj
import numpy as np
import nibabel as nib
from matplotlib.pyplot import imsave

from nilearn._utils.niimg import _safe_get_data
from nilearn.plotting.js_plotting_utils import colorscale, get_html_template
from nilearn.plotting.html_stat_map import (_get_cut_slices, _data_to_sprite, 
    _save_sprite, _save_cm, _json_view_params, _json_view_to_html, _json_view_size, 
    StatMapView, _mask_stat_map, _load_bg_img, _resample_stat_map)
from nilearn.image import new_img_like

def create_sprite(out_dir, bg_img, comp_map_img, cmap, threshold, title, output_sprite,
                  cbar_img_path, viewer_name, comp_map_sprite, symmetric_cmap=False, 
                  vmax=100, vmin=0, colorbar=True, black_bg=False, opacity=0.9,
                  draw_cross=False, annotate=False):
    output_sprite = opj(out_dir, 'figures/bg_sprite.png')
    comp_map_sprite = opj(out_dir, 'figures/stat_sprite.png')
    bg_img = opj(out_dir, bg_img)
    comp_map_img = opj(out_dir, comp_map_img)

    json_view = dict.fromkeys(['bg', 'stat_map', 'cmap', 'params'])

    #let's do things the nilearn way
    mask_img, comp_map_img, data, threshold = _mask_stat_map(
            comp_map_img, threshold)

    bg_img, bg_min, bg_max, black_bg = _load_bg_img(comp_map_img, bg_img,
                                                        black_bg, dim=False)

    comp_map_img, mask_img = _resample_stat_map(comp_map_img, bg_img, mask_img,
                                                    resampling_interpolation='continuous')

    # Create a sprite for the background
    bg_data = _safe_get_data(bg_img, ensure_finite=True)

    sprite = _data_to_sprite(bg_data)
    sprite[sprite == 0] = np.max(bg_data)

    imsave(output_sprite, sprite, vmin=np.min(bg_data), vmax=np.max(bg_data), cmap='Greys_r',
        format='png')
    json_view['bg'] = output_sprite

    # Create a sprite for the component map
    data = _safe_get_data(comp_map_img, ensure_finite=True)
    data[np.isclose(data, 0)] = 0
    mask = _safe_get_data(mask_img, ensure_finite=True)

    sprite = _data_to_sprite(data)
    if mask is not None:
        mask2 = _data_to_sprite(mask)
        sprite = np.ma.array(sprite, mask=mask2)

    imsave(comp_map_sprite, sprite, vmin=vmin, vmax=vmax, cmap=cmap,
        format='png')
    json_view['stat_map'] = comp_map_sprite
    n_colors = 256

    colors = colorscale(cmap, data.ravel(), threshold=threshold,
                            symmetric_cmap=symmetric_cmap, vmax=vmax,
                            vmin=vmin)

    # Create a colormap image
    if colorbar:
        _save_cm(cbar_img_path, colors['cmap'], 'png')
        data = np.arange(0., n_colors) / (n_colors - 1.)
        data = data.reshape([1, n_colors])
        json_view['cmap'] = cbar_img_path 
    else:
        json_view['cmap'] = ''

    cut_slices = _get_cut_slices(comp_map_img, cut_coords=None, threshold=None)        
    params = _json_view_params(
        comp_map_img.shape, comp_map_img.affine, colors['vmin'],
        colors['vmax'], cut_slices, black_bg, opacity, draw_cross, annotate,
        title, colorbar, value=False)

    js_dict = dict.fromkeys(['params', 'canvas'])
    js_dict['params'] = params
    params['canvas'] = viewer_name
    json_view['viewerId'] = viewer_name
    js_dict['canvas'] = params['canvas']
    width, height = _json_view_size(js_dict['params'])
    js_dict['params'] = json.dumps(js_dict['params'])

    sprite_script = get_html_template('./data/html/sprite_template.js')
    sprite_script = sprite_script.safe_substitute(js_dict)

    #Load the html template, and plug in all the data
    html_view = get_html_template('./data/html/sprite_template.html')
    html_view = html_view.safe_substitute(json_view)

    return html_view, sprite_script