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

from circle_3d_module.lsq_fit import annulus_dist, fit_annulus, sphere_dist, plane_dist, rodrigues
from circle_3d_module.rviz_markers import make_plane_marker, make_sphere_marker, make_points_marker, make_annulus_marker

from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from connected_segmentation.msg import LabeledPointCloud2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, Point, PoseArray, Pose
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


class Ring_detector(Node):
    def __init__(self, nodename="ring_detector", frequency=5):
        super().__init__(nodename)

        # params
        max_plane_dist_param = self.declare_parameter('max_plane_dist', .015)
        max_ring_err_param = self.declare_parameter('max_ring_err', 5.)
        max_ring_rad_param = self.declare_parameter('max_ring_rad', .15)
        debug_index_param = self.declare_parameter('debug_index', -1)
        min_intersection_area_param = self.declare_parameter('min_intersection_area', .35)
        min_plane_inliers_param = self.declare_parameter('min_plane_inliers', .8)
        max_ring_dist_param = self.declare_parameter('max_ring_dist', .8)
        running_mean_window_size_param = self.declare_parameter('running_mean_window_size', 10)
        normal_sim_tol_param = self.declare_parameter('normal_sim_tol', .1)
        max_diag_size_param = self.declare_parameter('max_diag_size', 1.)

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
        self.detection_pub = self.create_publisher(MarkerArray, "/classification/ring/detection", qos_profile)
        self.detection_pose_pub = self.create_publisher(PoseArray, "/classification/ring/detection/pose", qos_profile)
        self.demo_pub = self.create_publisher(MarkerArray, "/classification/ring/demonastration", qos_profile)

        # Publishers
        #self.seg_conn_pub = self.create_publisher(PointCloud2, "/segmentation/connected/points", qos_profile)

        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=60.))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.rings = []
        

    def timer_callback(self):
        marker_array = MarkerArray()
        pose_array = PoseArray()
        for i, rings in enumerate(self.rings):
            centers = np.array([ring[0] for ring in rings])
            rads_1 = np.array([ring[2] for ring in rings])
            rads_2 = np.array([ring[3] for ring in rings])
            normals = np.array([ring[4] for ring in rings])
            colors = np.array([ring[5] for ring in rings])
            color = np.hstack((colors.mean(axis=0) / 255, 1))
            marker = make_annulus_marker(centers.mean(axis=0), normals.mean(axis=0), (rads_1.mean(), rads_2.mean()), color, id=i, ns='ring_detection')
            marker_array.markers.append(marker)
            pose = Pose()
            pose.position.x = centers.mean(axis=0)[0]
            pose.position.y = centers.mean(axis=0)[1]
            pose.position.z = centers.mean(axis=0)[2]
            pose_array.poses.append(pose)
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
        max_diag_size = self.get_parameter('max_diag_size').get_parameter_value().double_value
        color_labes_list = range(color_labels_n) if debug_index < 0 else [debug_index]
        demo_markers = []

        for conn_lab in range(connected_labels_n):
            connected_mask = connected_labels == conn_lab
            connected_mask_size = connected_mask.sum()
            points_min = points[connected_mask].min(axis=0)
            points_max = points[connected_mask].max(axis=0)
            diag_size = np.linalg.norm(points_max - points_min)
            if diag_size > max_diag_size:
                continue
            for color_lab in color_labes_list:
                color_mask = color_labels == color_lab
                color_mask_size = color_mask.sum()
                intersection_mask = connected_mask * color_mask
                intersection_size = intersection_mask.sum()
                #cor = cv.filter2D(connected_mask * 1., ddepth=-1, kernel=(color_mask * 1.))
                if intersection_size < connected_mask_size * min_intersection_area or intersection_size < color_mask_size * min_intersection_area:
                    continue
                else:
                    img = real2img(intersection_mask)
                    img = self.bridge.cv2_to_imgmsg(img)
                    self.det_debug.publish(img)
                    detection, new_markers = self.process_intersection(points, bgr, intersection_mask, depth_to_world_transform, (conn_lab, color_lab))
                    demo_markers += new_markers
                    if detection is not None:
                        self.add_ring(detection)
        marker_array = MarkerArray()
        marker_array.markers = demo_markers
        self.demo_pub.publish(marker_array)

    
    def match_temp(self, img, template):
        bounds = np.where(img > 0)
        (x_min, y_min), (x_max, y_max) = np.min(bounds, axis=1), np.max(bounds, axis=1)
        img = img[x_min:x_max, y_min:y_max].astype(np.float32)
        
        bounds = np.where(template > 0)
        (x_min, y_min), (x_max, y_max) = np.min(bounds, axis=1), np.max(bounds, axis=1)
        template = template[x_min:x_max, y_min:y_max].astype(np.float32)

        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        th = .8
        loc = res >= th
        print("tiles", res.sum(), res.mean(), res.std())
        


    def add_ring(self, detection):
        max_ring_dist = self.get_parameter('max_ring_dist').get_parameter_value().double_value
        running_mean_window_size = self.get_parameter('running_mean_window_size').get_parameter_value().integer_value
        normal_sim_tol = self.get_parameter('normal_sim_tol').get_parameter_value().double_value

        candidate_center, candidate_rad, candidate_rad_1, candidate_rad_2, candidate_normal, candidate_color = detection
        filter_rings = None
        for compare_rings in self.rings:
            compare_centers = np.array([ring[0] for ring in compare_rings])
            compare_normals = np.array([ring[4] for ring in compare_rings])
            same_center = np.linalg.norm(compare_centers.mean(axis=0) - candidate_center) < max_ring_dist
            cos_sim = np.dot(compare_normals.mean(axis=0), candidate_normal)
            same_normal = np.isclose(np.abs(cos_sim), 1, atol=normal_sim_tol)
            if same_normal:
                candidate_normal = np.copysign(candidate_normal, compare_normals.mean(axis=0))
            if same_center:
                filter_rings = compare_rings

        if filter_rings is None:
            filter_rings = collections.deque([], maxlen=running_mean_window_size)
            self.rings.append(filter_rings)

        filter_rings.append([candidate_center, candidate_rad, candidate_rad_1, candidate_rad_2, candidate_normal, candidate_color])

    def process_intersection(self, points, bgr, detection_mask, transform, intersection_labels):
        min_plane_inliers = self.get_parameter('min_plane_inliers').get_parameter_value().double_value
        max_plane_dist = self.get_parameter('max_plane_dist').get_parameter_value().double_value
        max_ring_err = self.get_parameter('max_ring_err').get_parameter_value().double_value
        max_ring_rad = self.get_parameter('max_ring_rad').get_parameter_value().double_value

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])

        fit_points = points[detection_mask]
        fit_points = rot.apply(fit_points)
        fit_points = fit_points + [translation.x, translation.y, translation.z]
        fit_points = fit_points.reshape(-1, 3)
        demo_markers = []
        section_detections = []

        outlier_p = .20
        ransac_success_p = .99
        model_candidates = 8
        data_n = fit_points.shape[0]
        k = ransac_interations(data_n, model_candidates, outlier_p, ransac_success_p)
        plane1 = pyrsc.Plane()
        plane_model, plane_inliers = plane1.fit(fit_points, thresh=max_plane_dist, minPoints=model_candidates, maxIteration=k)

        '''plane_model, residuals = fit_plane(fit_points)
        plane_inliers = plane_dist(plane_model, fit_points).max() < max_plane_dist'''
 
        in_plane = plane_inliers.shape[0] > fit_points.shape[0] * min_plane_inliers

        fit_points = fit_points[plane_inliers]
        ptp = fit_points.max(axis=0) - fit_points.min(axis=0)
        ptp = np.linalg.norm(ptp)

        plane_marker = make_plane_marker((ptp, ptp), plane_model[:3], fit_points.mean(axis=0), [1, 1, 0, .75], ns=f"plane_intersection_{intersection_labels}")

        if in_plane:
            rot = rodrigues(plane_model[:3], [0, 0, 1])
            fit_points_flat = rot.apply(fit_points)
            fit_points_flat[:, 2] = fit_points_flat[:, 2].mean()
            #print(fit_points_flat.shape)
            fit_points = rot.inv().apply(fit_points_flat)

        annulus_model = fit_annulus(fit_points, plane_model[:3])
        (ring_center, rad), normal, (r1, r2) = annulus_model
        sphere_marker = make_sphere_marker(rad, ring_center, [0, 1, 1, .3], ns=f"sphere_intersection_{intersection_labels}")
        points_sphere_dist = sphere_dist((ring_center, rad), fit_points)
        points_annulus_dist = annulus_dist(annulus_model, fit_points)

        err_sphere = np.abs(points_sphere_dist) 
        err_ann = np.abs(points_annulus_dist)
        err_sphere = err_sphere.max()
        err_ann = err_ann.max()

        hanging = np.abs(plane_model[2]) < .1
        has_hole = r1 > .01
        not_too_big = r2 < .15
        not_too_filled = r1 * 2.5 > r2
        ring_valid = err_ann <= max_ring_err and in_plane and hanging and has_hole and not_too_big and not_too_filled
        print(f"{ring_center}, {err_ann = }, {err_sphere = }, {r1 = }, {r2 = }, {rad = }, {ring_valid =}")
        
        red, green = [1, 0, 0, 1], [0, 1, 0, 1]
        circle_points = make_points_marker([fit_points], [green if ring_valid else red], ns=f"points_intersection_{intersection_labels}")

        demo_markers += [plane_marker, sphere_marker, circle_points]

        if ring_valid:
            color = bgr[detection_mask].mean(axis=0)
            detection = (ring_center, rad, r1, r2, plane_model[:3], color[::-1])
        else:
            detection = None

        return detection, demo_markers


def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = Ring_detector()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning ring detection node, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

