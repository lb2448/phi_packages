#!/usr/bin/env python3


import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy import Parameter

from circle_3d_module.rviz_markers import make_plane_marker, make_sphere_marker, make_points_marker, make_cylinder_marker
from circle_3d_module.lsq_fit import annulus_dist, fit_circle, sphere_dist, plane_dist, rodrigues

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PointStamped, Point, PoseArray, Pose
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped, PointStamped, TransformStamped, Point, Pose, Quaternion
from visualization_msgs.msg import MarkerArray

from scipy.spatial.transform import Rotation as R
from scipy.stats import linregress

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO
import pyransac3d as pyrsc

import tf2_ros
import tf2_geometry_msgs

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

from time import time as tik
import collections


class Face_detector(Node):

    def __init__(self, frequency=5):
        super().__init__('detect_faces')
        self.frequency = frequency
        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
        ])

        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.timer_period = 1/frequency
        timer_callback_group = MutuallyExclusiveCallbackGroup()
        client_callback_group = MutuallyExclusiveCallbackGroup()
        self.timer = self.create_timer(self.timer_period, self.timer_callback, callback_group=timer_callback_group)

        self.device = self.get_parameter('device').get_parameter_value().string_value
        max_face_dist_param = self.declare_parameter('max_face_dist', .6) 
        running_mean_window_size_param = self.declare_parameter('running_mean_window_size', 10)
        normal_sim_tol_param = self.declare_parameter('normal_sim_tol', 1e-1)

        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.bridge = CvBridge()
        self.scan = None
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data,
                                                       callback_group=client_callback_group)
        self.poses_pub = self.create_publisher(PoseArray, "/classification/poeple/detection/pose", qos_profile)
        self.image_pub = self.create_publisher(Image, "/classification/poeple/demonstration/image", qos_profile)
        self.people_markers_pub = self.create_publisher(MarkerArray, "/classification/poeple/detection", qos_profile)

        self.model = YOLO("yolov8n.pt")

        self.faces = []

        self.tf_buffer = tf2_ros.Buffer(rclpy.time.Duration(seconds=15.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()

        self.people = []
    

    def timer_callback(self):
        people_marker_array = MarkerArray()
        poses_array = PoseArray()
        poses_array.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        for i, person in enumerate(self.people):
            centers = np.array([person[0] for person in person])
            normals = np.array([person[1] for person in person])
            pointss = [person[2] for person in person]
            argmax_points_size = np.argmax([p.size for p in pointss])
            points = pointss[argmax_points_size]
            colorss = [person[3] for person in person]
            colors = colorss[argmax_points_size].reshape(-1, 3) / 255
            colors = np.pad(colors[:, ::-1], ((0, 0), (0, 1)), constant_values=1.)
            
            if centers.shape[0] > 3:
                center = centers.mean(axis=0)
                normal = normals.mean(axis=0)


                qx, qy, qz, qw = rodrigues(normal, [1, 0, 0]).as_quat()
                orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
                pose = Pose(position=Point(x=center[0], y=center[1], z=center[2]), orientation=orientation)

                marker = make_points_marker([points], [colors], orientation=orientation, id=i, ns='faces_marker', header=poses_array.header, type=Marker.POINTS)
                people_marker_array.markers.append(marker)
                poses_array.poses.append(pose)


        self.poses_pub.publish(poses_array)
        self.people_markers_pub.publish(people_marker_array)
        #print(i)


    def add_person(self, candidate_person):
        max_face_dist = self.get_parameter('max_face_dist').get_parameter_value().double_value
        normal_sim_tol = self.get_parameter('normal_sim_tol').get_parameter_value().double_value
        running_mean_window_size = self.get_parameter('running_mean_window_size').get_parameter_value().integer_value

        candidate_center, candidate_normal, candidate_points, candidate_colors = candidate_person
        if np.any(np.isnan(candidate_center)):
            return
        
        filter_person = None
        for compare_person in self.people:
            compare_centers = np.array([person[0] for person in compare_person])
            compare_normals = np.array([person[1] for person in compare_person])

            cos_sim = np.dot(compare_normals.mean(axis=0), candidate_normal)
            same_normal = np.isclose(cos_sim, 1, atol=normal_sim_tol)
            same_center = np.linalg.norm(compare_centers.mean(axis=0) - candidate_center) < max_face_dist
            #print("same_center", same_center, "same_normal", same_normal)
            if same_normal and same_center:
                filter_person = compare_person
            
        compare_person = [np.array([candidate_center]), np.array([candidate_normal])]
        self.faces.append(compare_person)
        if filter_person is None:
            filter_person = collections.deque([], maxlen=running_mean_window_size)
            self.people.append(filter_person)

        filter_person.append([candidate_center, candidate_normal, candidate_points, candidate_colors])
            
    def pointcloud_callback(self, pointcloud_message):
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
        
        tok = tik()
        # process pointclod message
        h, w = pointcloud_message.height, pointcloud_message.width
        points = pc2.read_points_numpy(pointcloud_message, field_names=("x", "y", "z")).reshape(h, w, 3)
        bgr = pc2.read_points_numpy(pointcloud_message, field_names=("rgb",))
        bgr = bgr.view(np.uint8).reshape(h, w, 4)[..., :3]
        bgr = np.pad(bgr,((8,8), (0,0), (0,0)))
        depth = np.linalg.norm(points, axis=-1)
        
        #img = self.bridge.cv2_to_imgmsg(rgb)
        #self.image_pub.publish(img)

        # run inference
        res = self.model.predict(bgr, imgsz=bgr.shape[:2], show=False, verbose=False, classes=[0], device=self.device)

        # iterate over results
        for detection in res:
            bbox = detection.boxes.xyxy
            if bbox.nelement() == 0: # skip if empty
                continue

            face = self.process_person_detection(detection, depth_to_world_transform.transform, points, bgr)
            self.add_person(face)


    def process_person_detection(self, detection, transform, points, bgr):
        
        translation = transform.translation
        trans_vec = [translation.x, translation.y, translation.z]
        rotation = transform.rotation
        rot = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])

        h, w = points.shape[:2]     
        bbox = detection.boxes.xyxy[0]    
        x1, y1, x2, y2 = [int(t) for t in bbox.cpu().numpy()]

        det_h, det_w = int(x2 - x1), int(y2 - y1)
        points_t = points[y1:y2, x1:x2]
        rgb_det = bgr[y1:y2, x1:x2].reshape(-1, 3)
        points_shape = points_t.shape
        points_t = rot.apply(points_t.reshape(-1, 3))#.reshape(points_shape)
        points_t = points_t + trans_vec
        '''normal = np.arctan2((points[h//2,x1-5:x1+5,1].mean() - points[h//2,x2-5:x2+5,1].mean()),
                             (points[h//2,x1-5:x1+5,0].mean() - points[h//2,x2-5:x2+5,0].mean()))'''
        plane1 = pyrsc.Plane()
        model, inliers = plane1.fit(points_t, thresh=.005, minPoints=10, maxIteration=50)
        center = points_t.mean(axis=(0,))
        robot_pos = rot.apply([0,0,0]) + trans_vec
        normal = model[:3]/np.linalg.norm(model[:3])
        dirx = 1 if center[0] < robot_pos[0] else -1
        diry = 1 if center[1] < robot_pos[1] else -1
        normal = np.abs(normal)
        normal[0] *= dirx
        normal[1] *= diry

        return center, normal, points_t[inliers], rgb_det[inliers].astype(np.uint8)


    def point_cloud(self, header, points, colors):
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


def main():
    rclpy.init(args=None)
    node = Face_detector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info('Beginning people detection node, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down.\n')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
