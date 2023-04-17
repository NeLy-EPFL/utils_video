""" 3D plotting based on df3d package. """
import numpy as np

try:
    from df3d.config import config
except ImportError:
    print('df3d is not installed! Plotting functions might not work.')


def plot_drosophila_3d(
        ax_3d,
        points3d,
        cam_id,
        bones=config["bones"],
        ang=None,
        draw_joints=None,
        colors=None,
        zorder=None,
        thickness=None,
        lim=None,
        scatter=False,
        axis=False
):
    points3d = np.array(points3d)
    if draw_joints is None:
        draw_joints = np.arange(config["skeleton"].num_joints)
    if colors is None:
        colors = config["skeleton"].colors
    colors_tmp = ["#%02x%02x%02x" % c for c in colors]
    if zorder is None:
        zorder = config["skeleton"].get_zorder(cam_id)
    if thickness is None:
        thickness = np.ones((points3d.shape[0])) * 3

    colors = []
    for i in range(config["skeleton"].num_joints):
        colors.append(colors_tmp[config["skeleton"].get_limb_id(i)])
    colors = np.array(colors)

    white = (1.0, 1.0, 1.0, 0.0)
    ax_3d.w_xaxis.set_pane_color(white)
    ax_3d.w_yaxis.set_pane_color(white)

    ax_3d.w_xaxis.line.set_color(white)
    ax_3d.w_yaxis.line.set_color(white)
    ax_3d.w_zaxis.line.set_color(white)

    if ang is not None:
        ax_3d.view_init(ax_3d.elev, ang)
    else:
        if cam_id < 3:
            ax_3d.view_init(ax_3d.elev, -60 + 30 * cam_id)
        else:
            ax_3d.view_init(ax_3d.elev, -60 + 45 * cam_id)

    if lim:
        max_range = lim
        mid_x = 0
        mid_y = 0
        mid_z = 0
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    if "fly" in config["name"]:
        for j in range(config["skeleton"].num_joints):
            if config["skeleton"].is_tracked_point(j, config["skeleton"].Tracked.STRIPE) and config[
                "skeleton"].is_joint_visible_left(j):
                points3d[j] = (points3d[j] + points3d[j + (config["skeleton"].num_joints // 2)]) / 2
                points3d[j + config["skeleton"].num_joints // 2] = points3d[j]

    if scatter:
        for j in draw_joints:
            ax_3d.scatter(points3d[j, 0],
                points3d[j, 1],
                points3d[j, 2],
                c=colors[j],
                linewidth=thickness[config["skeleton"].get_limb_id(j)],
                zorder=zorder[j])

    for bone in bones:
        if bone[0] in draw_joints and bone[1] in draw_joints:
            ax_3d.plot(
                points3d[bone, 0],
                points3d[bone, 1],
                points3d[bone, 2],
                c=colors[bone[0]],
                linewidth=thickness[config["skeleton"].get_limb_id(bone[0])],
                zorder=zorder[bone[0]],
            )
    for bone in config["skeleton"].bones3d:
        if bone[0] in draw_joints and bone[1] in draw_joints:
            ax_3d.plot(
                points3d[bone, 0],
                points3d[bone, 1],
                points3d[bone, 2],
                c=colors[bone[0]],
                linewidth=5,
                zorder=zorder[bone[0]],
            )