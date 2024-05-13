#!/usr/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy import Parameter
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from autonomous_explore.robot_commander import RobotCommander

import numpy as np
import tf2_ros

from enum import Enum
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, Pose, PoseArray
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

from circle_3d_module.lsq_fit import annulus_dist, fit_circle, sphere_dist, plane_dist, rodrigues
from circle_3d_module.rviz_markers import make_point_cloud, make_points_marker, make_text_marker

from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, PointField
from connected_segmentation.msg import LabeledPointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from autonomous_explore.MapSearch import MapSearch

from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped, PointStamped, TransformStamped, Point
from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation as R
import time
from time import time as tik


def map_cleanup(map: OccupancyGrid):
    info = map.info
    map = np.uint8(map.data).reshape(info.height, info.width)
    map = np.flip(map, axis=0)
    map = map == 0
    bounds = np.where(map == 0)
    (x_min, y_min), (x_max, y_max) = np.min(bounds, axis=1), np.max(bounds, axis=1)
    map = map[x_min:x_max, y_min:y_max]
    return info, map, (x_min, y_min, x_max, y_max)


def map_transforms(info, bounds):
    to_map = TransformStamped()
    to_map.transform.translation.x = info.origin.position.x - bounds[1]*info.resolution
    to_map.transform.translation.y = info.height*info.resolution + info.origin.position.y - bounds[0]*info.resolution
    to_map.transform.translation.z = info.origin.position.z*info.resolution
    to_map.transform.rotation = info.origin.orientation
    to_grid = TransformStamped()
    to_grid.transform.translation.x = -info.origin.position.x/info.resolution  - bounds[1]
    to_grid.transform.translation.y = +info.height + info.origin.position.y/info.resolution - bounds[0]
    to_grid.transform.translation.z = -info.origin.position.z/info.resolution
    to_grid.transform.rotation.x = info.origin.orientation.x
    to_grid.transform.rotation.y = info.origin.orientation.y
    to_grid.transform.rotation.z = info.origin.orientation.z
    to_grid.transform.rotation.w = info.origin.orientation.w
    return to_map, to_grid


def real2img(data, max_value=None, cmap=None):
    max_value = np.nanmax(data) if max_value is None else max_value
    data = data.copy()
    if data.dtype is np.float32:
        data[np.isnan(data)] = np.inf
    data[data > max_value] = max_value
    data = data / max_value
    data = (data * 255).astype(np.uint8)
    if cmap is not None:
        data = cv2.applyColorMap(data, cmap)
    return data


amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)


class AutonomousExplore(Node):

    def __init__(self, commander_node, nodename='autonomous_explore_node', frequency=2):
        super().__init__(nodename)

        self.robot_commander = commander_node

        plane_sample_size_param = self.declare_parameter('plane_sample_size', 500)
        max_goal_view_param = self.declare_parameter('max_goal_view', 1.5)
        goal_bubble_radius_param = self.declare_parameter('goal_bubble_radius', 1.)
        map_dilate_reps_param = self.declare_parameter('map_dilate_reps', 10)
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        qos_profile = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Dirt publish
        self.dirt_publish_timer_period = 1/frequency
        self.dirt_publish_timer = self.create_timer(self.dirt_publish_timer_period, self.publish_dirt)

        pc_cb_group = MutuallyExclusiveCallbackGroup()
        map_cb_group = MutuallyExclusiveCallbackGroup()
        park_cb_group = MutuallyExclusiveCallbackGroup()

        self.map_subscription = self.create_subscription(OccupancyGrid, '/map', self.handle_map_subscription, qos_profile,
                                                         callback_group=map_cb_group)
        self.seg_subscription = self.create_subscription(LabeledPointCloud2, "/segmentation/labeled_pointcloud2", self.labeled_pc_callback, 
                                                          qos_profile_sensor_data, callback_group=pc_cb_group)

        self.park_subscription = self.create_subscription(PoseArray, "/classification/parking_spot/detection/pose", self.park_spots_array, 
                                                          qos_profile_sensor_data, callback_group=park_cb_group)

        qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1
                )

        self.image_pub = self.create_publisher(Image, "/autonomous_explore/map_image", qos_profile)
        self.dirt_pub = self.create_publisher(Marker, "/autonomous_explore/dirt", qos_profile)
        self.broom_pub = self.create_publisher(Marker, "/autonomous_explore/broom", qos_profile)
        self.robot_state_text_pub = self.create_publisher(Marker, "/autonomous_explore/state_text", qos_profile)
        self.goal_pub = self.create_publisher(PoseStamped, "/autonomous_explore/goal", qos_profile)

        self.dirt_points = None
        self.remaining_goal_points = None
        self.seen_points = None
        self.ignored_points = None
        self.obstacle_points = None
        self.map_search = None
        self.robot_pos = None
        self.map_info = None
        self.park_spots = []
        self.parked = False

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=60.))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.depth_to_world_transform = None


    def park_spots_array(self, park_array: PoseArray):
        self.park_spots = park_array.poses
        self.debug("Parking spots received.")


    def publish_dirt(self):
        if self.dirt_points is None or self.remaining_goal_points is None or self.ignored_points is None:
            self.debug("No dirt points to publish.")
            return
        
        dirt_points = np.indices(self.dirt_points.shape)[:, self.remaining_goal_points].T
        ignored_points = np.indices(self.dirt_points.shape)[:, self.ignored_points].T
        world_dirt_points = self.occupancy_to_world(dirt_points)
        world_dirt_points = np.pad(world_dirt_points, ((0, 0), (0, 1)))
        world_ignored_points = self.occupancy_to_world(ignored_points)
        world_ignored_points = np.pad(world_ignored_points, ((0, 0), (0, 1)))
        marker = make_points_marker([world_dirt_points, world_ignored_points], [[1, 1, .8, .5], [.8, 1, 1, .5]], scale=self.map_info.resolution)
        self.debug("Dirt published.")
        self.dirt_pub.publish(marker)

        #img = real2img(self.map_search.map_state)
        img = self.map_search.search_result_image()
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2BGR)
        img = self.bridge.cv2_to_imgmsg(img)
        self.image_pub.publish(img)


    def labeled_pc_callback(self, lpc2_message: LabeledPointCloud2):
        if self.map_info is None:
            return
        
        pointcloud_message = lpc2_message.pointcloud
        header = pointcloud_message.header

        try:
            depth_to_world_transform : TransformStamped = self.tf_buffer.lookup_transform_full(
            target_frame="map",
            target_time=self.tf_buffer.get_latest_common_time('map', header.frame_id), #header.stamp,
            source_frame=header.frame_id,
            source_time=header.stamp,
            fixed_frame='map',
            timeout=rclpy.duration.Duration(seconds=0.05))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException, tf2_ros.TransformException) as e: 
            print(e)
            return
                
        # process pointclod message
        h, w = pointcloud_message.height, pointcloud_message.width
        points = pc2.read_points_numpy(pointcloud_message, field_names=("x", "y", "z")).reshape(h, w, 3)
        depth = np.linalg.norm(points, axis=-1)
        connected_labels, connected_labels_n = np.frombuffer(lpc2_message.connected_labels, dtype=np.uint8).reshape(h, w), lpc2_message.connected_labels_n
        ground_plane_mask = connected_labels == connected_labels_n
        ground_points = points[ground_plane_mask]

        translation = depth_to_world_transform.transform.translation
        rotation = depth_to_world_transform.transform.rotation
        rot = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])

        dist = np.linalg.norm(ground_points[:, :2], axis=1)

        max_goal_view = self.get_parameter('max_goal_view').get_parameter_value().double_value
        points_t = ground_points[dist < max_goal_view]
        sample_size = self.get_parameter('plane_sample_size').get_parameter_value().integer_value
        sample = np.random.choice(points_t.shape[0], sample_size)

        points_t = np.vstack(([[0,0,0]], points_t[sample])) # get robot pos
        points_t = rot.apply(points_t)
        points_t = points_t + [translation.x, translation.y, translation.z]
        color = np.tile(np.array([255, 255, 200, 0], dtype=np.uint8), (points_t.shape[0], 1))

        #pc = make_point_cloud(Header(stamp=header.stamp, frame_id='map'), points_t, color)
        #self.broom_pub.publish(pc)
        #colors =  np.random.uniform(0, 1, (points_t.shape[0] - 1, 3))
        #colors = np.pad(colors, ((0, 0), (0, 1)), constant_values=1)
        broom_marker = make_points_marker([points_t[1:]], [[1., .85, 0, 1]], type=Marker.SPHERE_LIST, scale=.05)
        self.broom_pub.publish(broom_marker)

        new_seen_points = self.world_to_occupancy(points_t)
        self.update_view(new_seen_points[0], new_seen_points[1:]) # first point is robot pos
        self.debug("Pointcloud published.")


    def update_view(self, new_robot_pos, new_seen_points_idx):
        self.robot_pos = new_robot_pos[:2]

        new_seen_points_idx = new_seen_points_idx[new_seen_points_idx[:, 0] < self.seen_points.shape[0]]
        new_seen_points_idx = new_seen_points_idx[new_seen_points_idx[:, 1] < self.seen_points.shape[1]]
        new_seen_points = np.zeros_like(self.seen_points, dtype=float)
        new_seen_points[new_seen_points_idx[:, 0], new_seen_points_idx[:, 1]] = 1.
        stel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5 , 5))
        new_seen_points = cv2.dilate(new_seen_points, stel, iterations=1)
        self.seen_points = np.logical_or(new_seen_points > 0, self.seen_points)
        self.update_remaining_goals(new_robot_pos)


    def handle_map_subscription(self, map: OccupancyGrid):
        if self.map_info is not None:
            return
        
        map_dilate_reps = self.get_parameter('map_dilate_reps').get_parameter_value().integer_value

        info, map, bounds = map_cleanup(map)
        self.map_bounds = bounds
        self.map_info = info
        self.map = map
        self.map_search = MapSearch(~map)

        self.transform_to_world, self.transform_to_map = map_transforms(info, bounds)
        stel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3 , 3))
        tmap = np.pad(map, 1)
        dirt = cv2.dilate((~tmap).astype(np.float32), stel, iterations=map_dilate_reps)
        self.dirt_points = dirt[1:-1, 1:-1] == 0
        self.remaining_goal_points = self.dirt_points
        self.obstacle_points = map == 0
        self.seen_points = np.zeros_like(map) == 1

        self.robot_commander.waitUntilNav2Active()

        while self.robot_commander.is_docked is None:
            self.info("Waiting for is_docked...")
            rclpy.spin_once(self.robot_commander, timeout_sec=0.5)

        if self.robot_commander.is_docked:
            self.robot_commander.undock()

        self.explore()


    def explore(self):
        self.info("Exploration started...")
        while self.remaining_goal_points.sum() > 20:
            world_goal, curr_pose = self.find_goal()
            self.robot_state_text_pub.publish(make_text_marker("I'm exploring!", frame_id='base_link'))
            goal_pose = self.goal_from_pos(world_goal)
            self.goal_pub.publish(goal_pose)
            self.robot_commander.goToPose(goal_pose)
            while not self.robot_commander.isTaskComplete():
                self.info("Waiting to reach goal...")
                time.sleep(1)
            self.robot_state_text_pub.publish(make_text_marker("I'm looking around!", frame_id='base_link'))
            self.robot_commander.spin(-np.pi/2, 5)
            self.robot_commander.spin(np.pi/2, 5)
            self.info(f"{len(self.park_spots)} park spots found...")
            if len(self.park_spots) > 2 and not self.parked:
                self.robot_state_text_pub.publish(make_text_marker("I think I'll park for a second...", frame_id='base_link'))
                self.robot_commander.goToPose(self.park_spots[2])
                while not self.robot_commander.isTaskComplete():
                    self.info("Waiting to park...")
                    time.sleep(1)
                self.parked = True
                    

    def goal_from_pos(self, pos):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.robot_commander.get_clock().now().to_msg()

        goal_pose.pose.position.x = pos[0]
        goal_pose.pose.position.y = pos[1]
        yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
        goal_pose.pose.orientation = self.robot_commander.YawToQuaternion(yaw)
        return goal_pose


    def find_goal(self):
        robot_pose: PoseWithCovarianceStamped = self.robot_commander.current_pose
        pos = robot_pose.pose.position
        x, y, z = pos.x, pos.y, pos.z
        world_pos = np.array([[x, y, z]])
        occupany_pos = self.world_to_occupancy(world_pos)[0]

        self.update_remaining_goals(occupany_pos)
        self.map_search.reset()
        self.map_search.set_start_goals(tuple(occupany_pos), self.remaining_goal_points)

        path, cost = self.map_search.search()
        x0, y0 = occupany_pos   
        self.seen_points[x0, y0] = True
        occupancy_goal = np.array([path[0]])
        world_goal = self.occupancy_to_world(occupancy_goal)[0]
        return world_goal, [x, y, z]


    def update_remaining_goals(self, pos):
        x0, y0 = pos   
        r = self.get_parameter('goal_bubble_radius').get_parameter_value().double_value 
        r /= self.map_info.resolution
        x, y = np.indices(self.dirt_points.shape)
        bubble = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < r
        remaining_goal_points = np.logical_and(self.dirt_points, ~self.seen_points)
        self.ignored_points = np.logical_and(remaining_goal_points, bubble)
        self.remaining_goal_points = np.logical_and(remaining_goal_points, ~bubble)


    def occupancy_to_world(self, occupancy_grid_points):
        info = self.map_info
        resolution = self.map_info.resolution
        bounds = self.map_bounds
        world_points = occupancy_grid_points.copy() * resolution
        world_points[:, 1] = world_points[:, 1] + (info.origin.position.x - bounds[1]*resolution)
        world_points[:, 0] = - world_points[:, 0] + (info.height*resolution + info.origin.position.y - bounds[0]*resolution)
        world_points [:, [1, 0]] = world_points [:, [0, 1]]
        return world_points


    def world_to_occupancy(self, world_points):
        info = self.map_info
        resolution = self.map_info.resolution
        bounds = self.map_bounds
        occupancy_grid_points = world_points.copy()
        occupancy_grid_points[:, [1, 0]] = occupancy_grid_points[:, [0, 1]]
        occupancy_grid_points[:, 1] = occupancy_grid_points[:, 1] + (-info.origin.position.x  + bounds[1]*resolution)
        occupancy_grid_points[:, 0] = -(occupancy_grid_points[:, 0] + (-info.height*resolution - info.origin.position.y + bounds[0]*resolution))
        occupancy_grid_points = occupancy_grid_points / resolution
        occupancy_grid_points = np.round(occupancy_grid_points).astype(int)
        return occupancy_grid_points[:, :2]
       

    def info(self, msg):
        self.get_logger().info(msg)
        return


    def warn(self, msg):
        self.get_logger().warn(msg)
        return


    def error(self, msg):
        self.get_logger().error(msg)
        return


    def debug(self, msg):
        self.get_logger().debug(msg)
        return


def main():  
    rclpy.init()
    executor = MultiThreadedExecutor()
    commander_node = RobotCommander()
    explore_node = AutonomousExplore(commander_node)
    executor.add_node(explore_node)
    try:
        explore_node.get_logger().info('Beginning autonomous_explore node and robot_commander node, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        explore_node.get_logger().info('Keyboard interrupt, shutting down.\n')
    explore_node.destroy_node()
    commander_node.destroy_node()
    rclpy.shutdown()
    return
    map = cv2.imread('map4.pgm')


    map = cv2.cvtColor(map, cv2.COLOR_BGR2GRAY)
    map = np.uint8(map < 254)
    bounds = np.where(map == 0)
    (x_min, y_min), (x_max, y_max) = np.min(bounds, axis=1) - 5, np.max(bounds, axis=1) + 5
    map = map[x_min:x_max, y_min:y_max]

    plt.imshow(map)
    plt.show()


    normals = np.arctan2(cv2.Sobel(src=map, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3),
                         cv2.Sobel(src=map, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3))

    viewer = (40, 30, np.pi / 4)  # -np.pi/4
    explorer = AutonomousExplore(map)
    los_mask, ray_mask = explorer.los_mask(viewer)
    plt.imshow(los_mask + ray_mask * 2)
    plt.show()



    ms = MapSearch(map)
    # ms.heuristic = "manhattan"
    los_goals = np.zeros(ms.map_view.shape)
    los_goals[90, 115] = 1
    ms.set_goals(los_goals == 1)
    # ms.set_goals(ms.map_view == 1)
    ms.set_start((40, 20))
    start_time = time.time()
    path, cost = ms.search()
    ms.search_result_image()
    print("--- %s seconds ---" % (time.time() - start_time))
    ms.reset()

    '''rospy.init_node('los_client')
     send_map = rospy.ServiceProxy('los_service/set_map', SetMap)
     r = send_map(map.tobytes(), map.shape[0], map.shape[1])
     print(r)'''

    # plt.imshow(ms.map_view)
    # plt.show()
    # cv2.imwrite("test.bmp", np.uint8(ms.map_view * 255))
