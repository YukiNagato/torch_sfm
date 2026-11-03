"""Offline HTML visualization of Torch SfM checkpoints and in-memory results."""

from html import escape
from pathlib import Path

import torch
from PIL import Image

from ..geometry.two_view import camera_matrix


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise ImportError('HTML visualization requires plotly: python -m pip install plotly') from error
    return go


def _cpu(value, dtype=torch.float64):
    return torch.as_tensor(value).detach().to(device='cpu', dtype=dtype)


def _ranges(values):
    low, high = values.amin(0), values.amax(0)
    extent = (high - low).amax().clamp_min(1e-3)
    padding = torch.maximum((high - low) * .08, extent * .02)
    return torch.stack((low - padding, high + padding), -1).tolist()


def _aspect_ratio(bounds):
    """Equal world-unit scale even when the displayed bounds crop outliers."""
    spans = torch.tensor(bounds, dtype=torch.float64).diff(dim=-1).squeeze(-1)
    spans = spans / spans.prod().pow(1 / 3)
    return dict(zip('xyz', spans.tolist()))


def _view_cameras(rotations, view_up):
    """Viewing directions only; never rotate the saved or plotted coordinates.

    Image -Y is camera up, so its world direction is -R[1] for world-to-camera R.
    Averaging camera up is a display heuristic, not a gravity estimate.
    """
    basis = rotations[0] if len(rotations) else torch.eye(3, dtype=torch.float64)
    if len(rotations):
        ups = torch.nn.functional.normalize(-rotations[:, 1], dim=-1)
        average = ups.mean(0)
        auto = average if average.norm() > .25 else ups[0]
    else:
        auto = basis[2]
    vectors = {'auto': auto}
    for index, axis in enumerate('xyz'):
        direction = torch.eye(3, dtype=torch.float64)[index]
        vectors[axis], vectors['-' + axis] = direction, -direction
    if isinstance(view_up, str):
        if view_up not in vectors:
            raise ValueError('view_up must be auto, x, y, z, -x, -y, -z or a 3-vector')
        selected = view_up
    else:
        direction = _cpu(view_up)
        if direction.shape != (3,) or not torch.isfinite(direction).all() or direction.norm() <= 1e-12:
            raise ValueError('view_up must be a finite nonzero 3-vector')
        vectors['custom'] = direction
        selected = 'custom'
    cameras = {}
    for name, direction in vectors.items():
        up = direction / direction.norm()
        backward = -basis[2]
        backward = backward - torch.dot(backward, up) * up
        if backward.norm() < 1e-8:
            # Looking along up makes orbit/roll undefined. Use another camera axis.
            fallback = basis[torch.argmin((basis @ up).abs())]
            backward = fallback - torch.dot(fallback, up) * up
        backward = backward / backward.norm()
        right = torch.linalg.cross(up, backward)
        eye = 1.4 * right + up + 1.8 * backward
        cameras[name] = {'eye': dict(zip('xyz', eye.tolist())),
                         'up': dict(zip('xyz', up.tolist()))}
    return cameras, selected


def _segments(vertices, edges):
    """Plotly line coordinates with a gap between independent segments."""
    xyz = [[], [], []]
    for start, end in edges:
        for axis in range(3):
            xyz[axis].extend((float(vertices[start, axis]), float(vertices[end, axis]), None))
    return xyz


def _colmap_colors(folder, points):
    from ..io import Reconstruction
    folder = Path(folder)
    if not (folder / 'points3D.bin').exists() or not (folder / 'images.bin').exists():
        return None
    model = Reconstruction.from_binary(folder)
    if sorted(model.points3D) != list(range(1, len(points) + 1)):
        return None
    records = [model.points3D[index + 1] for index in range(len(points))]
    if not records:
        return torch.empty((0, 3), dtype=torch.uint8)
    xyz = torch.tensor([point.xyz.tolist() for point in records], dtype=points.dtype)
    # Do not apply colors from a stale reconstruction with different point IDs.
    if not torch.allclose(points, xyz, rtol=1e-6, atol=1e-8):
        return None
    return torch.tensor([point.rgb.tolist() for point in records], dtype=torch.uint8)


@torch.no_grad()
def build_figure(result, *, colors=None, image_paths=None, image_sizes=None,
                 camera_scale=None, max_points=100000, view_up='auto'):
    """Return a Plotly figure. R is world-to-camera and centers are world xyz.

    image_sizes are (height, width). Missing image files use an approximate
    centered sensor size inferred from K. All frustums use original-image K.
    view_up chooses the display up direction only; auto uses the cameras' mean
    image-up direction (or world Z if no camera is registered).
    """
    go = _plotly()
    data = result if isinstance(result, dict) else vars(result)
    points = _cpu(data['points3d'])
    rotations, centers = _cpu(data['rotations']), _cpu(data['centers'])
    registered = _cpu(data['registered'], torch.bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points3d must have shape (N, 3)')
    if rotations.shape != (len(centers), 3, 3) or centers.shape != (len(registered), 3):
        raise ValueError('Expected one rotation, center and registration flag per frame')
    if max_points is not None and (not isinstance(max_points, int) or max_points < 1):
        raise ValueError('max_points must be a positive integer or None')
    finite = torch.isfinite(points).all(-1)
    valid_cameras = registered & torch.isfinite(centers).all(-1) & torch.isfinite(rotations).all(dim=(-2, -1))
    camera_ids = torch.where(valid_cameras)[0]
    selected = torch.where(finite)[0]
    if len(selected) == 0 and len(camera_ids) == 0:
        raise ValueError('No finite points or registered cameras to visualize')
    if max_points is not None and len(selected) > max_points:
        selected = selected[torch.linspace(0, len(selected) - 1, max_points).long()]
    cloud = points[selected]
    active_centers = centers[camera_ids]
    all_values = torch.cat((cloud, active_centers))
    extent = float((all_values.amax(0) - all_values.amin(0)).amax().clamp_min(1e-3))
    if camera_scale is None:
        camera_extent = float((active_centers.amax(0) - active_centers.amin(0)).norm()) if len(camera_ids) > 1 else 0.
        camera_scale = max(camera_extent * .06, extent * .003, 1e-4)
    if not isinstance(camera_scale, (int, float)) or not 0 < camera_scale < float('inf'):
        raise ValueError('camera_scale must be finite and positive')
    if colors is None:
        colors = data.get('colors')
    point_colors = '#b8dce8'
    if colors is not None:
        rgb = _cpu(colors)
        if rgb.shape != points.shape or not torch.isfinite(rgb).all():
            raise ValueError('colors must be a finite (N, 3) RGB array')
        if len(rgb) and rgb.max() <= 1 and torch.as_tensor(colors).is_floating_point():
            rgb = rgb * 255
        point_colors = [f'rgb({r},{g},{b})' for r, g, b in rgb[selected].round().clamp(0, 255).long().tolist()]
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(name='Point cloud', mode='markers',
        x=cloud[:, 0].tolist(), y=cloud[:, 1].tolist(), z=cloud[:, 2].tolist(),
        marker={'size': 2, 'color': point_colors, 'opacity': .95},
        customdata=selected.tolist(),
        hovertemplate='Point %{customdata}<br>X %{x:.4f}<br>Y %{y:.4f}<br>Z %{z:.4f}<extra></extra>'))

    paths = image_paths if image_paths is not None else data.get('image_paths', [])
    matrices = camera_matrix(_cpu(data['intrinsics'])) if len(camera_ids) else None
    if matrices is not None and matrices.ndim == 2:
        matrices = matrices.expand(len(centers), 3, 3)
    if matrices is not None and matrices.shape != (len(centers), 3, 3):
        raise ValueError('Intrinsics must be shared or provided for every frame')
    lines, axes = [[], [], []], [[[], [], []] for _ in range(3)]
    frustum_vertices, names = [], []
    for frame_id in camera_ids.tolist():
        matrix, rotation, center = matrices[frame_id], rotations[frame_id], centers[frame_id]
        if image_sizes is not None:
            height, width = image_sizes[frame_id]
        elif frame_id < len(paths) and Path(paths[frame_id]).is_file():
            with Image.open(paths[frame_id]) as image:
                width, height = image.size
        else:
            width = max(2 * float(matrix[0, 2]), 1.)
            height = max(2 * float(matrix[1, 2]), 1.)
        pixels = points.new_tensor([[0., 0., 1.], [width, 0., 1.],
                                    [width, height, 1.], [0., height, 1.], [width / 2, -.18 * height, 1.]])
        local = torch.linalg.solve(matrix, pixels.T).T * camera_scale
        # Row vectors: X_world = X_camera @ R + center.
        vertices = torch.cat((center[None], local @ rotation + center))
        frustum_vertices.append(vertices)
        segments = _segments(vertices, ((0, 1), (0, 2), (0, 3), (0, 4),
                                       (1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (5, 2)))
        for axis in range(3):
            lines[axis].extend(segments[axis])
        for axis in range(3):
            end = center + rotation[axis] * camera_scale * .7
            segment = _segments(torch.stack((center, end)), ((0, 1),))
            for dimension in range(3):
                axes[axis][dimension].extend(segment[dimension])
        name = str(paths[frame_id]) if frame_id < len(paths) else f'frame {frame_id}'
        names.append(f'Camera {frame_id}: {escape(Path(name).name)}')
    figure.add_trace(go.Scatter3d(name='Camera frustums', mode='lines', x=lines[0], y=lines[1], z=lines[2],
        line={'color': '#56d6e6', 'width': 3}, hoverinfo='skip'))
    figure.add_trace(go.Scatter3d(name='Camera centers', mode='markers',
        x=active_centers[:, 0].tolist(), y=active_centers[:, 1].tolist(), z=active_centers[:, 2].tolist(),
        marker={'size': 3, 'color': '#ffb86b', 'line': {'color': '#fff1d6', 'width': .5}},
        text=names, hovertemplate='%{text}<br>X %{x:.4f}<br>Y %{y:.4f}<br>Z %{z:.4f}<extra></extra>'))
    figure.add_trace(go.Scatter3d(name='Image-order links', mode='lines', visible=False,
        x=active_centers[:, 0].tolist(), y=active_centers[:, 1].tolist(), z=active_centers[:, 2].tolist(),
        line={'color': '#ffb86b', 'width': 3}, hoverinfo='skip'))
    for axis, color in enumerate(('#ff6464', '#69da95', '#719dff')):
        figure.add_trace(go.Scatter3d(name=f'Camera {"XYZ"[axis]} axis', mode='lines', visible=False,
            x=axes[axis][0], y=axes[axis][1], z=axes[axis][2],
            line={'color': color, 'width': 3}, hoverinfo='skip'))
    vertices = torch.cat(frustum_vertices) if frustum_vertices else active_centers
    full = _ranges(torch.cat((cloud, vertices)))
    if len(cloud) > 10:
        subject = torch.quantile(cloud, cloud.new_tensor([.01, .99]), dim=0)
    else:
        subject = cloud
    subject = _ranges(torch.cat((subject, vertices)))
    camera_bounds = _ranges(vertices) if len(vertices) else full
    view_cameras, selected_up = _view_cameras(rotations[camera_ids], view_up)
    initial_camera = view_cameras[selected_up]
    axis_style = {'showbackground': False, 'gridcolor': '#233247', 'zerolinecolor': '#405066'}
    figure.update_layout(template='plotly_dark', paper_bgcolor='#0b1220', plot_bgcolor='#0b1220',
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0}, showlegend=False, autosize=True,
        scene={'aspectmode': 'manual', 'aspectratio': _aspect_ratio(subject),
               'dragmode': 'orbit', 'camera': initial_camera,
               'xaxis': dict(axis_style, title='X', range=subject[0]),
               'yaxis': dict(axis_style, title='Y', range=subject[1]),
               'zaxis': dict(axis_style, title='Z', range=subject[2])},
        meta={'points_total': int(finite.sum()), 'points_shown': len(cloud), 'cameras': len(camera_ids),
              'camera_scale': camera_scale, 'bounds': {'subject': subject, 'full': full, 'cameras': camera_bounds},
              'initial_camera': initial_camera, 'view_cameras': view_cameras, 'view_up': selected_up,
              'trace_groups': {'points': [0], 'cameras': [1, 2],
                                                               'trajectory': [3], 'axes': [4, 5, 6]}})
    return figure


def visualize_reconstruction(source, output_path=None, *, colors=None, image_paths=None, image_sizes=None,
                              camera_scale=None, max_points=100000, colmap_path=None, view_up='auto'):
    """Write a self-contained HTML viewer and return its Path.

    source accepts a workspace, .pt checkpoint, dict or GlomapResult. Plotly is
    an optional generation dependency; opening the HTML needs no network/server.
    """
    workspace = None
    if isinstance(source, (str, Path)):
        checkpoint = Path(source)
        if checkpoint.is_dir():
            checkpoint = checkpoint / 'reconstruction.pt'
        workspace = checkpoint.parent
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    else:
        data = source if isinstance(source, dict) else vars(source)
    if output_path is None:
        output_path = (workspace or Path.cwd()) / 'reconstruction.html'
    output_path = Path(output_path)
    if colmap_path is None and workspace is not None:
        colmap_path = workspace / 'sparse' / '0'
    if colors is None and data.get('colors') is None and colmap_path is not None:
        colors = _colmap_colors(colmap_path, _cpu(data['points3d']))
    figure = build_figure(data, colors=colors, image_paths=image_paths, image_sizes=image_sizes,
                          camera_scale=camera_scale, max_points=max_points, view_up=view_up)
    from .template import html_page
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_page(figure), encoding='utf-8')
    return output_path
