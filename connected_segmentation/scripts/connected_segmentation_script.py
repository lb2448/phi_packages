#!/usr/bin/python3

import rclpy
import rclpy.duration
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


from sensor_msgs.msg import Image, PointCloud2, PointField
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
from circle_3d_module.lsq_fit import fit_sphere, fit_plane, plane_dist

DPI = 300

def filter_small_labels(labels_2d, lab_n, min_siurface=20):
    removed = []
    for label in range(1, lab_n + 1):
        label_mask = labels_2d == label
        if label_mask.sum() < min_siurface:
            removed.append(label)
            labels_2d[labels_2d == label] = 0
    return labels_2d, lab_n - len(removed), np.array(removed)

def ransac_interations(data_n, model_candidates, outlier_p, ransac_success_p):
    d = np.ceil(data_n*(1-outlier_p))
    w = st.hypergeom(data_n, d, model_candidates).pmf(model_candidates)
    k = np.ceil(np.log(1-ransac_success_p)/np.log(1-w)).astype(int) 
    return k

def thresh(img):
    # Otsu's thresholding
    ret1, th1 = cv.threshold(img[..., 0], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    ret2, th2 = cv.threshold(img[..., 1], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    ret3, th3 = cv.threshold(img[..., 2], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    img[img[..., 2] < ret3] = 0
    #img[img[..., 1] < ret2, 1] = 0
    #img[img[..., 2] < ret3, 2] = 0
    return img

def eqalize(img):
    c1 = cv.equalizeHist(img[..., 0])
    c2 = cv.equalizeHist(img[..., 1])
    c3 = cv.equalizeHist(img[..., 2])
    return cv.merge((img[..., 0], img[..., 1], c3))

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

def img2matplotlib(img, cmap=None):
    (h, w), ch = img.shape[:2], img.shape[2] if len(img.shape) == 3 else 1
    fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.margins(0)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB) if ch == 3 else img
    ax.imshow(img, aspect='auto', cmap=cmap)
    return fig, ax

def matplotlib2img(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    plt.close(fig)
    return img

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


class Segmentator(Node):
    def __init__(self, nodename="connected_segmentation", check_subs_frequency=5, publish_frequency=120):
        super().__init__(nodename)

        # params
        connectivity_max_dist_param = self.declare_parameter('connectivity_max_dist', 2.)
        connectivity_z_res_param = self.declare_parameter('connectivity_z_res', .025)
        plane_filter_z_param = self.declare_parameter('plane_filter_z', .10)
        plane_filter_y_param = self.declare_parameter('plane_filter_y', 1.5)
        plane_filter_x_param = self.declare_parameter('plane_filter_x', .75)
        max_plane_dist_param = self.declare_parameter('max_plane_dist', .015)
        plane_fit_sample_size_param = self.declare_parameter('plane_fit_sample_size', 250)
        min_labels_size_param = self.declare_parameter('min_labels_size', 10)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        #self.add_on_set_parameters_callba

        # General stuff, for controlling node execution
        check_subs_timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.check_subs_timer = self.create_timer(1/check_subs_frequency, self.check_subs_timer_callback, callback_group=check_subs_timer_cb_group)
        publish_timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.publish_timer = self.create_timer(1/publish_frequency, self.publish_timer_callback, callback_group=publish_timer_cb_group)

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.SYSTEM_DEFAULT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1
                )
        
        # Subscribers to point clods
        self.camera_points_message = None

        # Publishers
        self.seg_conn_pub = self.create_publisher(PointCloud2, "/segmentation/connected/points", qos_profile)
        self.seg_color_pub = self.create_publisher(PointCloud2, "/segmentation/color/points", qos_profile)
        self.seg_conn_img_pub = self.create_publisher(Image, "/segmentation/connected/image", qos_profile)
        self.seg_color_img_pub = self.create_publisher(Image, "/segmentation/color/image", qos_profile)
        self.pointclound__cb_group = ReentrantCallbackGroup()
        self.segmentation_pc_pub = self.create_publisher(LabeledPointCloud2, "/segmentation/labeled_pointcloud2", qos_profile)
        self.publish_queue = []

        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=60.))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cam_to_base_link_transform = None

        self.t = False

        self.rand_permute = np.random.permutation(1000)
        self.fps_t = None
        self.fps = 0


    def publish_timer_callback(self):
        now = self.get_clock().now()
        if self.fps_t is None:
            self.fps_t = now
            self.fps = 0
        elif (now - self.fps_t) > rclpy.duration.Duration(seconds=2.5):
            fps = self.fps / ((now - self.fps_t).nanoseconds / 1e9)
            print(f"FPS: {fps}")
            self.fps_t = self.get_clock().now()
            self.fps = 0

        while len(self.publish_queue) > 0:
            job = self.publish_queue[0]
            #print(len(self.publish_queue))
            if not job.done:
                if (self.get_clock().now() - job.stamp) > rclpy.duration.Duration(seconds=.5):
                    self.publish_queue.pop(0)
                    continue
                else:
                    break
            else:
                self.publish_queue.pop(0)
            (connected_seg_pc, connected_seg_img), (color_seg_pc, color_seg_img), lpc2 = job.payload
            self.segmentation_pc_pub.publish(lpc2)
            self.seg_conn_pub.publish(connected_seg_pc)
            self.seg_color_pub.publish(color_seg_pc)
            connected_seg_img = self.bridge.cv2_to_imgmsg(connected_seg_img)
            color_seg_img = self.bridge.cv2_to_imgmsg(color_seg_img)
            self.seg_conn_img_pub.publish(connected_seg_img)
            self.fps += 1
            #self.seg_color_img_pub.publish(color_seg_img)


    def check_subs_timer_callback(self):
        cam_subs = self.seg_conn_pub.get_subscription_count()
        cam_subs += self.seg_color_pub.get_subscription_count()
        cam_subs += self.seg_conn_img_pub.get_subscription_count()
        cam_subs += self.seg_color_img_pub.get_subscription_count()
        cam_subs += self.segmentation_pc_pub.get_subscription_count() 
        if cam_subs > 0 and self.camera_points_message is None:
            self.camera_points_message = self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self.points_callback, 
                                                                  qos_profile_sensor_data, callback_group=self.pointclound__cb_group)
        if cam_subs == 0 and self.camera_points_message is not None:
            #self.camera_points_message.destroy()
            self.camera_points_message = None
 

    def find_valid_points(self, points, depth):
        not_nan_mask = ~np.isnan(points).any(axis=-1)
        max_dist = self.get_parameter('connectivity_max_dist').get_parameter_value().double_value
        valid_dist_mask = depth <= max_dist
        valid_points_mask = np.logical_and(valid_dist_mask, not_nan_mask)
        return valid_points_mask


    '''
    Fit plane    
    '''
    def fit_plane(self, points):
        translation = self.cam_to_base_link_transform.transform.translation
        rotation = self.cam_to_base_link_transform.transform.rotation

        #rot_mat = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix()
        points_t = points.reshape(-1, 3)
        #points_t = np.einsum('ij,kj->ki', rot_mat, points_t)
        points_t = points_t + [translation.x, translation.y, translation.z]
        plane_filter_z = self.get_parameter('plane_filter_z').get_parameter_value().double_value
        plane_filter_y = self.get_parameter('plane_filter_y').get_parameter_value().double_value
        plane_filter_x = self.get_parameter('plane_filter_x').get_parameter_value().double_value
        hi = points_t[:, 2] <= plane_filter_z
        '''lo = points_t[:, 2] >= -plane_filter_z
        l = points_t[:, 0] <= plane_filter_x
        r = points_t[:, 0] >= -plane_filter_x
        f = points_t[:, 1] <= plane_filter_y
        filter = hi*1. + lo*1. + l*1. + r*1. + f*1.
        filter = filter == 5'''

        fit_points = points_t[hi]
        sample_size = self.get_parameter('plane_fit_sample_size').get_parameter_value().integer_value
        sample = np.random.choice(fit_points.shape[0], sample_size)

        max_plane_dist = self.get_parameter('max_plane_dist').get_parameter_value().double_value
        outlier_p = .20
        ransac_success_p = .90
        model_candidates = 8
        data_n = sample.size
        k = ransac_interations(data_n, model_candidates, outlier_p, ransac_success_p)
        plane1 = pyrsc.Plane()
        model, inliers = plane1.fit(fit_points[sample], thresh=max_plane_dist, minPoints=model_candidates, maxIteration=k)

        if model is None:
            return None
        normal = model[:3] / np.linalg.norm(model[:3])
        x, y, z = np.abs(normal)
        if np.max([x, y]) > .8:
            return None

        total_inliers = plane_dist(model, points_t) < max_plane_dist

        '''ground_points = fit_points[sample]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d') 
        ax.plot(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], '.')
        a, b, c, d = model
        normal = [a, b, c]
        norm = normal / np.linalg.norm(normal) * 5
        ax.quiver(0,0,0, *norm, color='g')
        set_axes_equal(ax)
        labels_display = matplotlib2img(fig)
        labels_display = self.bridge.cv2_to_imgmsg(labels_display)
        self.seg_color_img_pub.publish(labels_display)'''
        
        return total_inliers


    def find_ground_plane_points(self, points, valid_points_mask):
        ground_plane = self.fit_plane(points[valid_points_mask])
        ground_plane_mask = np.zeros_like(valid_points_mask)
        ground_plane_mask[valid_points_mask == 1] = ground_plane
        return ground_plane_mask


    ''' 
    Produces map of pixel, to label of connected region. 
    Outputs are: labels_2d_mask (mask of valid labels), labels_2d (the map), lab_n (number of labels),
                ground_plane_mask (mask of ground plane pixels)
    The label "lab_n" represnts ground plane points.
    The label "lab_n+1" represnts NaN points or points that are >= MAX_DIST.
    Labels "0...lab_n-1" are connected components sorted by mean distance to camera.  
    '''
    def generate_connected_labels(self, points, depth, points_to_segmnet_mask):
        tok = tik()

        r_min = depth[points_to_segmnet_mask].min()
        r_max = depth[points_to_segmnet_mask].max()
        connectivity_z_res = self.get_parameter('connectivity_z_res').get_parameter_value().double_value
        z_len = int(np.ceil((r_max - r_min) / connectivity_z_res))
        h_l, w_l = points.shape[:2]
        occupancy_array = np.zeros((h_l, w_l, z_len), dtype=np.uint8)
        '''z_len = 100
        z_bins = [r_min]
        for i in range(z_len):'''

        z_bins = np.linspace(r_min, r_max, num=z_len)
        z_bins = (z_bins[1:] + z_bins[:-1])/2
        z_not_nan_points = depth[points_to_segmnet_mask]
        acc = .005
        z_range_low = z_not_nan_points * (1 - acc)
        z_range_high = z_not_nan_points * (1 + acc)
        z_idxs = np.digitize(z_not_nan_points, z_bins, right=True)
        z_low_idxs = np.digitize(z_range_low, z_bins, right=True)
        z_high_idxs = np.digitize(z_range_high, z_bins, right=True)
        occupancy_array[points_to_segmnet_mask, z_idxs] = 1
        '''occupancy_array[points_to_segmnet_mask, z_low_idxs] = 1
        occupancy_array[points_to_segmnet_mask, z_high_idxs] = 1'''


        tok=tik()
        strut = occupancy_array.sum(axis=(1,2)) < 5
        occupancy_array[strut] = 0
        #print(tik() - tok, "find strut full plane")
      
        tok=tik()
        #occupancy_array = ndimage.binary_opening(occupancy_array, structure=np.ones((1, 3, 1))) #.astype(occupancy_array.dtype)
        #occupancy_array = ndimage.convolve(occupancy_array, np.ones((2, 2, 1)))
        #t= ndimage.binary_opening(depth > 1., structure=np.ones((2, 2))) #.astype(occupancy_array.dtype)
        #print(tik() - tok, "erosion")

        '''x, y, z = np.where(occupancy_array)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-90, azim=-180, roll=0)
        ground_points = label_points.reshape(-1, 3)[::100]
        #ax.plot(ground_points[:, 0], ground_points[:, 1], ground_points[:, 2], '.')
        ax.plot(x, -y, z, '.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        set_axes_equal(ax)
        plt.show()
        exit()
        labels_display = matplotlib2img(fig)
        labels_display = self.bridge.cv2_to_imgmsg(labels_display)
        self.segmentation_image2_pub.publish(labels_display)'''
        
        # label array
        tok = tik()
        labels_3d, lab_n = cc3d.connected_components(occupancy_array, return_N=True, connectivity=18)
        #labels_3d = cc3d.dust(labels_3d, threshold=10, connectivity=26, in_place=False)
        labels_2d = labels_3d.sum(axis=2).astype(np.int8)
        #print(tik() - tok, "connectivity", lab_n)
        return labels_2d, lab_n
    

    def filter_labels(self, labels, counts, lab_n):
        min_labels_size = self.get_parameter('min_labels_size').get_parameter_value().integer_value
        discard_mask = counts < min_labels_size
        discard_mask[0] = True
        discard = np.arange(lab_n)[discard_mask]
        discard_labels_mask = (labels == discard[:, None, None]).sum(axis=0) > 0
        return discard_labels_mask


    def postprocess_connected_labels(self, depth, connected_labels_2d, connected_labels_n, points_to_segment_mask, ground_plane_mask, valid_points_mask):
        _, counts = np.unique(connected_labels_2d, return_counts=True)
        discard_labels_mask = self.filter_labels(connected_labels_2d, counts, connected_labels_n + 1)
        points_to_segment_mask = np.logical_and(points_to_segment_mask, ~discard_labels_mask)
        valid_points_mask = np.logical_and(valid_points_mask, ~discard_labels_mask)
        uniqes, new_connected_labels_2d = np.unique(connected_labels_2d[points_to_segment_mask], return_inverse=True)
        connected_labels_n = uniqes.size
        
        GROUND_PLANE = connected_labels_n
        INVALID_POINT = connected_labels_n + 1
        connected_labels_2d[points_to_segment_mask] = new_connected_labels_2d
        connected_labels_2d[~valid_points_mask] = INVALID_POINT
        connected_labels_2d[ground_plane_mask] = GROUND_PLANE

        # sort and order labels by min distance to camera
        label_list = np.arange(connected_labels_n)
        min_range_label = np.zeros((connected_labels_n,))
        for label in label_list:
            label_mask = connected_labels_2d[valid_points_mask] == label
            depth_masked = depth[valid_points_mask][label_mask]
            min_range_label[label] = np.nanmean(depth_masked) if depth_masked.size > 0 else np.inf
        order = np.argsort(min_range_label)
        sort_map = np.argsort(order)
        sort_map = np.concatenate((sort_map, [GROUND_PLANE, INVALID_POINT]))
        h, w = connected_labels_2d.shape
        connected_labels_2d = sort_map[connected_labels_2d.flat]
        connected_labels_2d = connected_labels_2d.reshape((h, w))
        return connected_labels_2d, connected_labels_n


    def generate_color_labels(self, bgr, valid_points_mask):
        tok = tik()
        h, w = bgr.shape[:2]
        #rgb = eqalize(rgb)
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
        hls = cv.cvtColor(bgr, cv.COLOR_BGR2HLS_FULL)
        #hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV_FULL)
        #print(tik() - tok, "convert")
        tok = tik()

        sat_mask = hls[..., 2] > 80
        l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        ab_len = np.sqrt(a ** 2 + b ** 2) 
        ab_len_mask = ab_len > 3
        near_black_mask = l < 15
        color_seg = ab_len_mask + sat_mask
        relevant_color_mask = np.logical_and(color_seg, valid_points_mask)
        relevant_black_mask = np.logical_and(near_black_mask, valid_points_mask)
        #print(tik() - tok, "mask")
        tok = tik()
        hue_k = 12
        '''hue_bins = np.linspace(0, 255, hue_k)
        sat_k = 6
        sat_bins = np.linspace(0, 255, sat_k)
        light_k = 3
        light_bins = np.linspace(0, 255, light_k)
        ab_len_k = 12
        ab_len_bins = np.linspace(0, 255, ab_len_k)
        l_k = 6
        l_k_bins = np.linspace(0, 255, l_k)
        ab_k = 6'''
        '''a_bins = np.linspace(0, 255, ab_k)
        b_bins = np.linspace(0, 255, ab_k)'''
        #hue_binned = np.digitize(hls[relevant_color_mask, 0], hue_bins, right=False)
        hue_binned = np.round((hls[relevant_color_mask, 0] / (255/hue_k)))
        '''sat_binned = np.digitize(hls[relevant_color_mask, 2], sat_bins, right=False)
        light_binned = np.digitize(hls[relevant_color_mask, 1], light_bins, right=False)'''
        #ab_len_binned = np.digitize(ab_len[relevant_color_mask], ab_len_bins, right=False)
        '''l_binned = np.digitize(l[relevant_color_mask], l_k_bins, right=False)
        a_binned = np.digitize(a[relevant_color_mask], a_bins, right=False)
        b_binned = np.digitize(b[relevant_color_mask], b_bins, right=False)'''
        #print(tik() - tok, "binns")
        tok = tik()
        #features = np.dstack([hue_binned, ab_len_binned])[0].astype(np.uint8).view(np.uint16)
        features = hue_binned
        uniques, color_lab = np.unique(features, axis=0, return_inverse=True)
        color_lab_n = uniques.size
        labels_2d = np.zeros((h, w), dtype=np.uint8)
        labels_2d[relevant_color_mask] = color_lab + 2
        labels_2d[relevant_black_mask] = 1
        #print(tik() - tok, "binns uniq")
        tok = tik()
        labels_2d = cv.medianBlur(labels_2d, 3)
        labels_2d, color_lab_n = cc3d.connected_components(labels_2d, return_N=True, connectivity=6, delta=.1)
        labels_2d = cv.medianBlur(labels_2d, 3)
        bgr = real2img(labels_2d, cmap=cv.COLORMAP_JET)
        color_seg_img = self.bridge.cv2_to_imgmsg(bgr)
        #self.seg_color_img_pub.publish(color_seg_img)
        #print(tik() - tok, "conn")
        tok = tik()
        unique, labels_2d, counts = np.unique(labels_2d, return_inverse=True, return_counts=True)
        #print(tik() - tok, "uniq conn")
        tok = tik()
        labels_2d = labels_2d.reshape(h, w)
        discard_labels_mask = self.filter_labels(labels_2d, counts, unique.size)
        #print(tik() - tok, "filter")
        tok = tik()
        unique, new_labels_2d = np.unique(labels_2d[~discard_labels_mask], return_inverse=True)
        color_lab_n = unique.size
        labels_2d[~discard_labels_mask] = new_labels_2d
        labels_2d[discard_labels_mask] = color_lab_n
        #print(tik() - tok, "unique")
        tok = tik()
        bgr = real2img(labels_2d, cmap=cv.COLORMAP_JET)
        color_seg_img = self.bridge.cv2_to_imgmsg(bgr)
        #self.seg_color_img_pub.publish(color_seg_img)
        return labels_2d, color_lab_n


    def process_points(self, pointcloud_message):
        h, w = pointcloud_message.height, pointcloud_message.width
        points = pc2.read_points_numpy(pointcloud_message, field_names=("x", "y", "z")).reshape(h, w, 3)
        bgr = pc2.read_points_numpy(pointcloud_message, field_names=("rgb",))
        bgr = bgr.view(np.uint8).reshape(h, w, -1)[..., :3]
        depth = np.linalg.norm(points, axis=-1)

        tok = tik()
        valid_mask = self.find_valid_points(points, depth)
        ground_mask = self.find_ground_plane_points(points, valid_mask)
        if ground_mask is None:
            return None
        #print(tik() - tok, "ground_mask")

        to_segment_mask = np.logical_and(valid_mask, ~ground_mask)

        tok = tik()
        if to_segment_mask.sum() < 10:
            connected_labs = np.zeros(points.shape[:2], dtype=np.uint8)
            connected_labs_n = 0
            color_labs_2d = np.zeros(points.shape[:2], dtype=np.uint8)
            color_labs_n = 0
        else:
            connected_labs, connected_labs_n = self.generate_connected_labels(points, depth, to_segment_mask)
            #print(tik() - tok, "connected")

            tok = tik()
            connected_labs, connected_labs_n = self.postprocess_connected_labels(depth, connected_labs, connected_labs_n, to_segment_mask, ground_mask, valid_mask)
            #print(tik() - tok, "post connected")

            tok = tik()
            color_labs_2d, color_labs_n = self.generate_color_labels(bgr, to_segment_mask)
            #print(tik() - tok, "color")


        tok = tik()
        # connected labels image
        if connected_labs_n > 0:
            connected_seg_img = real2img(connected_labs, max_value=(connected_labs_n - 1), cmap=cv.COLORMAP_JET) #cv.COLORMAP_JET
        else:
            connected_seg_img = np.zeros(list(connected_labs.shape) + [3], dtype=np.uint8)
        connected_seg_img[connected_labs == connected_labs_n + 1] = [0,0,0] # if label is nan or MAX_RANGE show as black
        connected_seg_img[connected_labs == connected_labs_n] = [255,255,255] # if point is ground show as white

        # color labels image
        if color_labs_n > 0:
            color_seg_img = real2img(color_labs_2d, max_value=(color_labs_n - 1), cmap=cv.COLORMAP_JET) #cv.COLORMAP_JET
        else:
            color_seg_img = np.zeros(list(color_labs_2d.shape) + [3], dtype=np.uint8)
        color_seg_img[color_labs_2d == color_labs_n] = [0,0,0] # if label is nan or background show as black

        # labeled pointcloud
        lpc2 = LabeledPointCloud2()
        lpc2.pointcloud = pointcloud_message
        lpc2.connected_labels_n = connected_labs_n
        lpc2.connected_labels = connected_labs.astype(np.uint8).tobytes()
        lpc2.color_labels_n = color_labs_n
        lpc2.color_labels = color_labs_2d.astype(np.uint8).tobytes()

        conn_seg_pc = self.point_cloud(pointcloud_message.header, points, connected_seg_img)
        color_seg_pc = self.point_cloud(pointcloud_message.header, points, color_seg_img)
        #print(tik() - tok, "messags")

        return (conn_seg_pc, connected_seg_img), (color_seg_pc, color_seg_img), lpc2


    def points_callback(self, pointcloud_message):
        header = pointcloud_message.header
        if self.cam_to_base_link_transform is None:
            try:
                self.cam_to_base_link_transform : TransformStamped = self.tf_buffer.lookup_transform(
                    target_frame='base_link',
                    source_frame=header.frame_id, 
                    time=header.stamp)  
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.cam_to_base_link_transform = None
            return
        if len(self.publish_queue) > 0:
            return
        job = Job(self.get_clock().now())
        
        self.publish_queue.append(job)

        payload = self.process_points(pointcloud_message)
        if payload is not None:
            for q_job in self.publish_queue:
                if q_job.__eq__(job):
                    job.done = True
                    job.payload = payload


    def point_cloud(self, header, points, colors):
        fields = [
            PointField(offset=0, name='x', count=1, datatype=PointField.FLOAT32),
            PointField(offset=4, name='y', count=1, datatype=PointField.FLOAT32),
            PointField(offset=8, name='z', count=1, datatype=PointField.FLOAT32),
            PointField(offset=12, name='rgb', count=1, datatype=PointField.UINT32)]
        

        data = np.hstack([points.reshape(-1, 3).astype(np.float32).view(np.uint8), 
                          colors.reshape(-1, 3).view(np.uint8)])

        return PointCloud2(
            header=header,
            height=points.shape[0],
            width=points.shape[1],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=data.shape[1],
            row_step=data.size,
            data=data.tobytes()
        )


class Job():

    def __init__(self, stamp):
        self.done = False
        self.stamp : rclpy.Time = stamp
        self.payload = None

    def __eq__(self, other):
        return self.stamp.__eq__(other.stamp)


def main():
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = Segmentator()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning connected_Segmentation node, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
