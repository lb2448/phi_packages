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

from circle_3d_module.lsq_fit import annulus_dist, fit_circle, sphere_dist, plane_dist, rodrigues
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
import time
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


class Parking_spot_detector(Node):
    def __init__(self, nodename="parking_spot_detector", frequency=5):
        super().__init__(nodename)

        # params
        valid_parking_rads_param = self.declare_parameter('valid_parking_rads', [.15, .4])
        max_plane_dist_param = self.declare_parameter('max_plane_dist', .015)
        max_parking_spot_err_param = self.declare_parameter('max_parking_spot_err', 1.)
        max_parking_spot_rad_param = self.declare_parameter('max_parking_spot_rad', .15)
        debug_index_param = self.declare_parameter('debug_index', -1)
        min_intersection_area_param = self.declare_parameter('min_intersection_area', .8)
        min_plane_inliers_param = self.declare_parameter('min_plane_inliers', .8)
        max_parking_spot_dist_param = self.declare_parameter('max_parking_spot_dist', .5)
        running_mean_window_size_param = self.declare_parameter('running_mean_window_size', 10)
        normal_sim_tol_param = self.declare_parameter('normal_sim_tol', .1)

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
        self.detection_pub = self.create_publisher(MarkerArray, "/classification/parking_spot/detection", qos_profile)
        self.detection_pose_pub = self.create_publisher(PoseArray, "/classification/parking_spot/detection/pose", qos_profile)
        self.demo_pub = self.create_publisher(MarkerArray, "/classification/parking_spot/demonastration", qos_profile)

        # Publishers
        #self.seg_conn_pub = self.create_publisher(PointCloud2, "/segmentation/connected/points", qos_profile)

        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=60.))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.parking_spots = []
        


    def timer_callback(self):
        marker_array = MarkerArray()
        pose_array = PoseArray()
        pose_array.poses = []
        for i, parking_spots in enumerate(self.parking_spots):
            centers = np.array([spot[0] for spot in parking_spots])
            rads = np.array([spot[1] for spot in parking_spots])
            rad = rads.mean()
            normals = np.array([spot[2] for spot in parking_spots])
            marker = make_annulus_marker(centers.mean(axis=0), normals.mean(axis=0), (rad -.025, rad), [0, 0, 0, 1], id=i, ns='parking_spot_detection')
            marker_array.markers.append(marker)
            pose = Pose()
            pose.position.x = centers.mean(axis=0)[0]
            pose.position.y = centers.mean(axis=0)[1]
            pose.position.z = centers.mean(axis=0)[2]
            pose_array.poses.append(pose)
        self.detection_pose_pub.publish(pose_array)
        self.detection_pub.publish(marker_array)


    def labeled_pc_callback(self, lpc2_message: LabeledPointCloud2):
        max_plane_dist = self.get_parameter('max_plane_dist').get_parameter_value().double_value
        min_plane_inliers = self.get_parameter('min_plane_inliers').get_parameter_value().double_value
        valid_parking_rads = self.get_parameter('valid_parking_rads').get_parameter_value().double_array_value


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
              
        translation = depth_to_world_transform.transform.translation
        rotation = depth_to_world_transform.transform.rotation
        rot = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])

        h, w = pointcloud_message.height, pointcloud_message.width
        points = pc2.read_points_numpy(pointcloud_message, field_names=("x", "y", "z")).reshape(h, w, 3)
        bgr = pc2.read_points_numpy(pointcloud_message, field_names=("rgb",))
        bgr = bgr.view(np.uint8).reshape(h, w, -1)[..., :3]
        depth = np.linalg.norm(points, axis=-1)
        connected_labels, connected_labels_n = np.frombuffer(lpc2_message.connected_labels, dtype=np.uint8).reshape(h, w), lpc2_message.connected_labels_n

        hls = cv.cvtColor(bgr, cv.COLOR_BGR2HLS_FULL)
        ground_mask = connected_labels == connected_labels_n
        black_mask = hls[..., 1] < 15
        circle_mask = np.logical_and(ground_mask, black_mask)
        
        if circle_mask.sum() < 20:
            return

        img = real2img(circle_mask)
        img = self.bridge.cv2_to_imgmsg(img)
        self.det_debug.publish(img)

        circle_points = points[circle_mask]
        circle_points = rot.apply(circle_points)
        circle_points = circle_points + [translation.x, translation.y, translation.z]
        circle_points = circle_points.reshape(-1, 3)

        outlier_p = .50
        ransac_success_p = .99
        model_candidates = 10
        data_n = circle_points.shape[0]
        k = ransac_interations(data_n, model_candidates, outlier_p, ransac_success_p)
        plane1 = pyrsc.Plane()
        plane_model, plane_inliers = plane1.fit(circle_points, thresh=max_plane_dist, minPoints=model_candidates, maxIteration=k)
        normal = plane_model[:3]

        if len(plane_model) == 4 and not plane_inliers.shape[0] > circle_points.shape[0] * min_plane_inliers:
            return
        
        fit_points = circle_points[plane_inliers]
        center, rad = fit_circle(fit_points, normal)

        ptp = fit_points.max(axis=0) - fit_points.min(axis=0)
        ptp = np.linalg.norm(ptp)

        plane_marker = make_plane_marker((ptp, ptp), normal, fit_points.mean(axis=0), [1, 1, 0, .75], ns=f"plane_intersection_parking")
        sphere_marker = make_sphere_marker(rad, center, [0, 1, 1, .75], ns=f"sphere_intersection_parking")
        
        marker_array = MarkerArray()
        marker_array.markers = [plane_marker, sphere_marker]
        self.demo_pub.publish(marker_array)

        if rad > valid_parking_rads[0] and rad < valid_parking_rads[1]:
            self.add_parking_spot((center, rad, normal))

    
    def add_parking_spot(self, detection):
        max_parking_spot_dist = self.get_parameter('max_parking_spot_dist').get_parameter_value().double_value
        running_mean_window_size = self.get_parameter('running_mean_window_size').get_parameter_value().integer_value
        normal_sim_tol = self.get_parameter('normal_sim_tol').get_parameter_value().double_value

        candidate_center, candidate_rad, candidate_normal = detection
        filter_parking_spots = None
        for compare_parking_spots in self.parking_spots:
            compare_centers = np.array([ring[0] for ring in compare_parking_spots])
            compare_rads = np.array([ring[1] for ring in compare_parking_spots])
            compare_normals = np.array([ring[2] for ring in compare_parking_spots])
            same_center = np.linalg.norm(compare_centers.mean(axis=0) - candidate_center) < max_parking_spot_dist
            same_rad = np.allclose(compare_rads.mean(), candidate_rad, atol=1e-2)
            cos_sim = np.dot(compare_normals.mean(axis=0), candidate_normal)
            same_normal = np.isclose(np.abs(cos_sim), 1, atol=normal_sim_tol)
            if same_normal:
                candidate_normal = np.copysign(candidate_normal, compare_normals.mean(axis=0))
            if same_center and same_normal:
                filter_parking_spots = compare_parking_spots
                
        #filtered_parking_spot = [np.array([candidate_center]), np.array([candidate_rad]), np.array([candidate_normal]),]
        if filter_parking_spots is None:
            filter_parking_spots = collections.deque([], maxlen=running_mean_window_size)
            self.parking_spots.append(filter_parking_spots)

        filter_parking_spots.append([candidate_center, candidate_rad, candidate_normal,])



def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = Parking_spot_detector()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning parking_spot detection node, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

