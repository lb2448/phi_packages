import numpy as np
import scipy.stats as st
import numba as nb
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import rclpy
import triangle as tr
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from circle_3d_module.lsq_fit import rodrigues
from sensor_msgs.msg import Image, PointCloud2, PointField


def make_point_cloud(header, points, colors):
    fields = [
        PointField(offset=0, name='x', count=1, datatype=PointField.FLOAT32),
        PointField(offset=4, name='y', count=1, datatype=PointField.FLOAT32),
        PointField(offset=8, name='z', count=1, datatype=PointField.FLOAT32),
        PointField(offset=12, name='rgb', count=1, datatype=PointField.UINT32)]
    
    data = np.hstack([points.astype(np.float32).view(np.uint8), colors.view(np.uint8)])

    return PointCloud2(
        header=header,
        height=1,
        width=data.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=data.shape[1],
        row_step=data.size,
        data=data.tobytes()
    )


def make_plane_marker(shape, normal, zero_point, color, ns='plane_marker', id=0, frame_id='map', duration=None):
    marker = Marker()
    marker.type = Marker.CUBE
    marker.header.frame_id = frame_id
    marker.id = id
    marker.ns = ns
    duration = rclpy.duration.Duration(seconds=5.) if duration is None else duration
    marker.lifetime = duration.to_msg()
    r, g, b, a = np.array(color, dtype=float)
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a
    marker.scale.x = float(shape[0])
    marker.scale.y = float(shape[1])
    marker.scale.z = float(shape[2]) if len(shape) == 3 else .0001 
    x, y, z = np.array(zero_point, dtype=float)
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    if not np.allclose(normal, [0, 0, 1]):
        rot = rodrigues([0, 0, 1], normal)
        x, y, z, w = rot.as_quat()
        marker.pose.orientation.x = x
        marker.pose.orientation.y = y
        marker.pose.orientation.z = z
        marker.pose.orientation.w = w
    return marker


def make_sphere_marker(rad, zero_point, color, ns='sphere_marker', id=0, frame_id='map', duration=None):
    marker = Marker()
    marker.type = Marker.SPHERE
    marker.header.frame_id = frame_id
    marker.id = id
    marker.ns = ns
    duration = rclpy.duration.Duration(seconds=5.) if duration is None else duration
    marker.lifetime = duration.to_msg()
    r, g, b, a = np.array(color, dtype=float)
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a
    marker.scale.x = float(rad) * 2
    marker.scale.y = float(rad) * 2
    marker.scale.z = float(rad) * 2
    x, y, z = np.array(zero_point, dtype=float)
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    return marker


def make_cylinder_marker(center, height, rad, color, id, ns='cylinder_marker', duration=None):
    marker = Marker()
    marker.type = Marker.CYLINDER
    marker.header.frame_id = "map"
    marker.id = id
    marker.ns = ns
    duration = rclpy.duration.Duration(seconds=5.) if duration is None else duration
    marker.lifetime = duration.to_msg()
    x, y, z = np.array(center, dtype=float)
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    r, g, b, a = np.array(color, dtype=float)
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a
    marker.scale.x = rad * 2
    marker.scale.y = rad * 2
    marker.scale.z = height
    return marker


def circle(rad, pts_n):
    theta = np.linspace(0, 2 * np.pi, num=pts_n)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * rad
    idx = np.arange(pts_n)
    seg = np.stack([idx, idx + 1], axis=1) % pts_n
    return pts, seg


def make_annulus_marker(center, normal, rads, colors, id=0, ns='annulus_marker', duration=None, size=.01):
    rad1, rad2 = rads
    pts_n_1 = int(2 * np.pi * rad1 / size)
    pts_n_2 = int(2 * np.pi * rad2 / size)
    pts_n_1 = min(max(pts_n_1, 10), 100)
    pts_n_2 = min(max(pts_n_2, 10), 100)
    circle1, outer_edges1 = circle(rad1, pts_n_1)
    circle2, outer_edges2 = circle(rad2, pts_n_2)
    pts = np.vstack([circle1, circle2])
    seg = np.vstack([outer_edges1, outer_edges2 + outer_edges1.shape[0]])
    
    tri_in_dict = dict(vertices=pts, segments=seg, holes=[[0, 0]])
    tri_out_dict = tr.triangulate(tri_in_dict, 'p')

    tri = tri_out_dict["triangles"]
    tri_points = pts[tri].reshape(-1, 2)
    tri_points = np.pad(tri_points, ((0, 0), (0, 1)))
    #tr.compare(plt, tri_in_dict, tri_out_dict)
    #plt.show()

    if not np.allclose(normal, [0, 0, 1], atol=.01):
        rot = rodrigues([0, 0, 1], normal)
        tri_points = rot.apply(tri_points.reshape(-1, 3))
    tri_points += center
    marker = Marker()
    marker.type = Marker.TRIANGLE_LIST
    marker.header.frame_id = "map"
    marker.id = id
    marker.ns = ns
    duration = rclpy.duration.Duration(seconds=5.) if duration is None else duration
    marker.lifetime = duration.to_msg()
    scale = float(1)
    marker.scale.x = scale
    marker.scale.y = scale
    marker.scale.z = scale
    marker_points = []
    marker_colors = []
    colors = np.array(colors, dtype=float).reshape(-1, 4)
    marker_points += [Point(x=tri_points[i, 0], y=tri_points[i, 1], z=tri_points[i, 2]) for i in range(tri_points.shape[0])]
    if colors.shape[0] == tri_points.shape[0]:
        marker_colors += [ColorRGBA(r=colors[i, 0], g=colors[i, 1], b=colors[i, 2], a=colors[i, 3]) for i in range(colors.shape[0])]
    else:
        marker_colors += [ColorRGBA(r=colors[0, 0], g=colors[0, 1], b=colors[0, 2], a=colors[0, 3]) for _ in range(tri_points.shape[0])]
    marker.points = marker_points
    marker.colors = marker_colors
    return marker


def make_points_marker(pointss, colorss, id=0, scale=0.01, ns='points_marker', duration=None, type=Marker.CUBE_LIST):
    pointss = [np.array(points).reshape(-1, 3).astype(float) for points in pointss]
    colorss = [np.array(colors).reshape(-1, 4).astype(float) for colors in colorss]
    marker = Marker()
    marker.type = type
    marker.header.frame_id = "map"
    marker.id = id
    marker.ns = ns
    duration = rclpy.duration.Duration(seconds=5.) if duration is None else duration
    marker.lifetime = duration.to_msg()
    if hasattr(scale, '__iter__'):
        scale_x, scale_y, scale_z = [float(s) for s in scale]
    else:
        scale = float(scale)
        scale_x, scale_y, scale_z = scale, scale, scale
    marker.scale.x = scale_x
    marker.scale.y = scale_y
    marker.scale.z = scale_z
    marker_points = []
    marker_colors = []
    for points, colors in zip(pointss, colorss):
        marker_points += [Point(x=points[i, 0], y=points[i, 1], z=points[i, 2]) for i in range(points.shape[0])]
        if colors.shape[0] == points.shape[0]:
            marker_colors += [ColorRGBA(r=colors[i, 0], g=colors[i, 1], b=colors[i, 2], a=colors[i, 3]) for i in range(colors.shape[0])]
        else:
            marker_colors += [ColorRGBA(r=colors[0, 0], g=colors[0, 1], b=colors[0, 2], a=colors[0, 3]) for _ in range(points.shape[0])]
    marker.points = marker_points
    marker.colors = marker_colors
    return marker
    

def make_text_marker(text, point=None, color=None, id=0, scale=.25, ns='text_marker', duration=None, frame_id="map", frame_locked=True):
    if point is None:
        point = [0., 0., 1] 
    if color is None:
        color = [.75, .75, .75, 1]
    marker = Marker()
    marker.text = text
    marker.type = Marker.TEXT_VIEW_FACING
    marker.header.frame_id = frame_id
    marker.id = id
    marker.ns = ns
    duration = rclpy.duration.Duration(seconds=10.) if duration is None else duration
    marker.lifetime = duration.to_msg()
    scale = float(scale)
    marker.scale.z = scale
    r, g, b, a = np.array(color, dtype=float)
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = a
    x, y, z = np.array(point, dtype=float)
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.frame_locked = frame_locked
    return marker    