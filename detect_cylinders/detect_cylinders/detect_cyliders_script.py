#!/usr/bin/python3


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy import Parameter

from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy import ndimage, fft, signal
from scipy.cluster.vq import vq, kmeans, whiten
import pyransac3d as pyrsc
import cc3d
'''import hdbscan
import pyfftw
import multiprocessing
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()'''


import cv2 as cv
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
#import numba as nb
from sensor_msgs_py import point_cloud2 as pc2

from circle_3d_module.lsq_fit import make_plane_points, make_sphere_points, fit_sphere, sphere_dist, cartesian2spherical
from circle_3d_module.rviz_markers import make_plane_marker, make_sphere_marker, make_points_marker, make_cylinder_marker

from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from connected_segmentation.msg import LabeledPointCloud2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy.stats as st
from time import time as tik
import collections


def real2img(data, max_value=None, cmap=None):
    max_value = np.nanmax(data) if max_value is None else max_value
    data = data.copy()
    if data.dtype is np.float32:
        data[np.isnan(data)] = np.inf
    data[data > max_value] = max_value
    data = data / max_value
    data = (data * 255).astype(np.uint8)
    if cmap is not None:
        data = cv.applyColorMap(data, cmap)
    return data


def ransac_interations(data_n, model_candidates, outlier_p, ransac_success_p):
    d = np.ceil(data_n*(1-outlier_p))
    w = st.hypergeom(data_n, d, model_candidates).pmf(model_candidates)
    k = np.ceil(np.log(1-ransac_success_p)/np.log(1-w)).astype(int) 
    return k


class Cylinder_detector(Node):
    def __init__(self, nodename="cylinder_detector", frequency=5):
        super().__init__(nodename)

        # params
        section_heights_param = self.declare_parameter('section_heights', [.1, .2, .3])
        max_plane_dist_param = self.declare_parameter('max_plane_dist', .015)
        debug_index_param = self.declare_parameter('debug_index', -1)
        min_intersection_area_param = self.declare_parameter('min_intersection_area', .8)
        max_cylinder_param = self.declare_parameter('max_cylinder_dist', 1.5)
        running_mean_window_size_param = self.declare_parameter('running_mean_window_size', 10)

        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.timer_period = 1/frequency
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=timer_callback_group)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1
                )
        
        # Subscribers to point clods
        self.labeld_pc_sub = self.create_subscription(LabeledPointCloud2, "/segmentation/labeled_pointcloud2", self.labeled_pc_callback, qos_profile_sensor_data)

        self.det_debug = self.create_publisher(Image, "/debug", qos_profile)
        self.detection_pub = self.create_publisher(MarkerArray, "/classification/cylinder/detection", QoSReliabilityPolicy.BEST_EFFORT)
        self.demo_pub = self.create_publisher(MarkerArray, "/classification/cylinder/demonastration", QoSReliabilityPolicy.BEST_EFFORT)

        # Publishers
        #self.seg_conn_pub = self.create_publisher(PointCloud2, "/segmentation/connected/points", qos_profile)

        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=60.))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.cylinders = []


    def timer_callback(self):
        marker_array = MarkerArray()
        '''marker_array.markers = [self.make_detection_marker(centers.mean(axis=0), heights.mean(), rads.mean(), colors.mean(axis=0), i) 
                        for i, (centers, rads, heights, colors ) in enumerate(self.cylinders)]'''
        for i, cylinders in enumerate(self.cylinders):
            centers = np.array([ring[0] for ring in cylinders])
            rads = np.array([ring[1] for ring in cylinders])
            heights = np.array([ring[2] for ring in cylinders])
            colors = np.array([ring[3] for ring in cylinders])
            color = colors.mean(axis=0) / 255
            marker = make_cylinder_marker(centers.mean(axis=0), heights.mean(), rads.mean(), [*color[::-1], 1], id=i, ns='cylinder_detection')
            marker_array.markers.append(marker)
        self.detection_pub.publish(marker_array)


    def labeled_pc_callback(self, lpc2_message: LabeledPointCloud2):
        pointcloud_message = lpc2_message.pointcloud
        header = pointcloud_message.header

        try:
            depth_to_world_transform : TransformStamped = self.tf_buffer.lookup_transform_full(
            target_frame="map",
            target_time=header.stamp,
            source_frame=header.frame_id,
            source_time=header.stamp,
            fixed_frame='map',
            timeout=rclpy.duration.Duration(seconds=0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e: 
            print(e)
            return
                
        h, w = pointcloud_message.height, pointcloud_message.width
        points = pc2.read_points_numpy(pointcloud_message, field_names=("x", "y", "z")).reshape(h, w, 3)
        bgr = pc2.read_points_numpy(pointcloud_message, field_names=("rgb",))
        bgr = bgr.view(np.uint8).reshape(h, w, -1)[..., :3]
        depth = np.linalg.norm(points, axis=-1)
        connected_labels, connected_labels_n = np.frombuffer(lpc2_message.connected_labels, dtype=np.uint8).reshape(h, w), lpc2_message.connected_labels_n
        color_labels, color_labels_n = np.frombuffer(lpc2_message.color_labels, dtype=np.uint8).reshape(h, w), lpc2_message.color_labels_n

        debug_index = self.get_parameter('debug_index').get_parameter_value().integer_value
        min_intersection_area = self.get_parameter('min_intersection_area').get_parameter_value().double_value

        color_labes_list = range(color_labels_n) if debug_index < 0 else [debug_index]
        demo_markers = []

        for conn_lab in range(connected_labels_n):
            connected_mask = connected_labels == conn_lab
            connected_mask_size = connected_mask.sum()
            for color_lab in color_labes_list:
                color_mask = color_labels == color_lab
                color_mask_size = color_mask.sum()
                intersection_mask = connected_mask * color_mask
                intersection_size = intersection_mask.sum()
                if intersection_size < connected_mask_size * min_intersection_area or intersection_size < color_mask_size * min_intersection_area:
                    continue
                else:
                    img = real2img(intersection_mask)
                    img = self.bridge.cv2_to_imgmsg(img)
                    self.det_debug.publish(img)
                    detection, new_markers = self.process_intersection(points, bgr, intersection_mask, depth_to_world_transform, (conn_lab, color_lab))
                    demo_markers += new_markers
                    if detection is not None:
                        self.add_cylinder(detection)
        marker_array = MarkerArray()
        marker_array.markers = demo_markers
        self.demo_pub.publish(marker_array)

    
    def add_cylinder(self, detection):
        max_cylinder_dist = self.get_parameter('max_cylinder_dist').get_parameter_value().double_value
        running_mean_window_size = self.get_parameter('running_mean_window_size').get_parameter_value().integer_value
        candidate_center, candidate_rad, candidate_height, candidate_color = detection
        filter_cylinders = None
        for compare_cylinders in self.cylinders:
            compare_centers = np.array([ring[0] for ring in compare_cylinders])
            compare_rads = np.array([ring[1] for ring in compare_cylinders])
            same_center = np.linalg.norm(compare_centers.mean(axis=0) - candidate_center) < max_cylinder_dist
            same_rad = np.allclose(compare_rads.mean(), candidate_rad, atol=1e-2)
            if same_center and same_rad:
                filter_cylinders = compare_cylinders

        if filter_cylinders is None:
            filter_cylinders = collections.deque([], maxlen=running_mean_window_size)
            self.cylinders.append(filter_cylinders)

        filter_cylinders.append([candidate_center, candidate_rad, candidate_height, candidate_color])


    def process_intersection(self, points, rgb, detection_mask, transform, intersection_labels):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])

        section_heights = self.get_parameter('section_heights').get_parameter_value().double_array_value
        max_plane_dist = self.get_parameter('max_plane_dist').get_parameter_value().double_value

        fit_points = points[detection_mask]
        fit_points = rot.apply(fit_points)
        fit_points = fit_points + [translation.x, translation.y, translation.z]
        demo_markers = []
        section_detections = []
        marker_id = 0
        for i, section_height in enumerate(section_heights):

            section_mask = (fit_points[..., 2] < section_height * 1.05) * (fit_points[..., 2] > section_height * 0.95)
            fit_section_points = fit_points[section_mask]
            if fit_section_points.size < 10:
                continue

            outlier_p = .20
            ransac_success_p = .99
            model_candidates = 8
            data_n = section_mask.sum()
            k = ransac_interations(data_n, model_candidates, outlier_p, ransac_success_p)
            #plane1 = pyrsc.Plane()
            #model, inliers = plane1.fit(fit_points, thresh=max_plane_dist, minPoints=model_candidates, maxIteration=k)
            (center, r), residuals = fit_sphere(fit_section_points)

            xptp = np.ptp(fit_section_points[..., 0]) * 1.5
            yptp = np.ptp(fit_section_points[..., 1]) * 1.5
            
            '''plane_points = make_plane_points((xptp, yptp), [0, 0, 1], center)
            sphere_points = make_sphere_points(r, center)
            colors = np.ones(list(plane_points.shape[:-1]) + [4]) * [1, 0, 0, 1]
            marker = self.make_points_marker([sphere_points, plane_points], [[0, 1, 1, 1], [1, 1, 0, 1]], i)'''

            plane_marker = make_plane_marker((xptp, yptp), [0, 0, 1], center, [1, 1, 0, 1], id=i, ns=f"plane_intersection_{intersection_labels}")
            sphere_marker = make_sphere_marker(r, center, [0, 1, 1, .3], id=i, ns=f"sphere_intersection_{intersection_labels}")

            dist = sphere_dist((center, r), fit_section_points)
            theta, phi, _ = cartesian2spherical((center, r), fit_section_points)
            theta = theta - theta.min()
            phi = phi - phi.min()
            #print(i, center, r, (xptp, yptp), theta.max(), phi.max())

            circle_valid = np.max([theta.max(), phi.max()]) > np.pi/4 and dist.max() <= max_plane_dist and r < .4
            if circle_valid:
                section_detections.append((center, r))
            red, green = [1, 0, 0, 1], [0, 1, 0, 1]
            circle_points = make_points_marker([fit_section_points], [green if circle_valid else red], id=i, ns=f"points_intersection_{intersection_labels}",)

            demo_markers += [plane_marker, sphere_marker, circle_points]

        if len(section_detections) == 3:
            rad = np.array([r for _, r in section_detections]).mean()
            center = np.array([center for center, _ in section_detections]).mean(axis=0)
            height = fit_points[..., 2].max()
            color = rgb[detection_mask].mean(axis=0)
            detection = (center, rad, height, color)
        else:
            detection = None

        return detection, demo_markers





def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = Cylinder_detector()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning cylinder detection node, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

