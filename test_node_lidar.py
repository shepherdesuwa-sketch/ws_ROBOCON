import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, Image, PointField
from cv_bridge import CvBridge

import numpy as np
import cv2
import struct


class PostFusionNode(Node):

    def __init__(self):
        super().__init__('post_fusion_node')

        self.get_logger().info("=== Post Fusion Node (Cluster Projection DEBUG) ===")

        # ================= Subscribers =================
        self.sub_lidar = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback,
            10
        )

        self.valid_points_pub = self.create_publisher(
            PointCloud2,
            '/fusion/valid_points',
            10
        )

        self.clusters_pub = self.create_publisher(
            PointCloud2,
            '/fusion/clusters',
            10
        )

        # ================= Utils =================
        self.bridge = CvBridge()
        self.latest_image = None

        self.get_logger().info("=== Node Ready ===")

    # =====================================================
    # LiDAR Callback
    # =====================================================
    def lidar_callback(self, msg):

        points = self.pointcloud2_to_xyz(msg)
        self.get_logger().info(f"Raw points: {points.shape[0]}")

        if points.shape[0] == 0:
            return 

        valid_points, clusters = self.cluster_points(points)

        # ===== publish valid points =====
        if valid_points.shape[0] > 0:
            self.valid_points_pub.publish(
                self.xyz_to_pointcloud2(valid_points, msg.header)
            )

        # ===== publish clusters =====
        if len(clusters) > 0:
            cluster_msg = self.clusters_to_pointcloud2(clusters, msg.header)
            if cluster_msg is not None:
                self.clusters_pub.publish(cluster_msg)

    # =====================================================
    # Processing
    # =====================================================
    def cluster_points(self, points):
        from pyransac3d import Plane
        from sklearn.cluster import DBSCAN

        # 过滤掉地面、天花板、雷达后方的点
        valid_points = points[
            (points[:, 2] > 0.02) &
            (points[:, 2] < 0.5) &
            (points[:, 0] > 0.0)
        ]

        if valid_points.shape[0] < 20:
            return np.empty((0,3)), []

        # 平面分割（去除地面）
        plane = Plane()
        try:
            plane_eq, inliers = plane.fit(valid_points, thresh=0.02, maxIteration=1000)
        except Exception:
            return valid_points, []

        non_ground_points = np.delete(valid_points, inliers, axis=0)
        if non_ground_points.shape[0] < 20:
            return np.empty((0,3)), []

        # DBSCAN 聚类
        dbscan = DBSCAN(eps=0.15, min_samples=10)
        labels = dbscan.fit_predict(non_ground_points)

        clusters = []
        for cid in np.unique(labels):
            if cid == -1:
                continue
            clusters.append(non_ground_points[labels == cid])

        return non_ground_points, clusters

    # =====================================================
    # Utils
    # =====================================================
    def pointcloud2_to_xyz(self, msg):
        from sensor_msgs_py import point_cloud2
        return np.array([
            [p[0], p[1], p[2]]
            for p in point_cloud2.read_points(msg, skip_nans=True)
        ])

    def xyz_to_pointcloud2(self, points, header):
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]

        fields = []
        for i, name in enumerate(['x', 'y', 'z']):
            pf = PointField()
            pf.name = name
            pf.offset = i * 4
            pf.datatype = PointField.FLOAT32
            pf.count = 1
            fields.append(pf)
        msg.fields = fields

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        msg.data = b''.join([struct.pack('fff', *p) for p in points])
        return msg

    def clusters_to_pointcloud2(self, clusters, header):
        points = []
        for cid, cluster in enumerate(clusters):
            rgb = self.cluster_color(cid)
            for p in cluster:
                points.append([p[0], p[1], p[2], rgb])

        if not points:
            return None

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)

        fields = []
        for i, name in enumerate(['x', 'y', 'z', 'rgb']):
            pf = PointField()
            pf.name = name
            pf.offset = i * 4
            pf.datatype = PointField.FLOAT32 if name != 'rgb' else PointField.UINT32
            pf.count = 1
            fields.append(pf)
        msg.fields = fields

        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True

        msg.data = b''.join([struct.pack('fffI', *p) for p in points])
        return msg

    def cluster_color(self, cid):
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        r, g, b = colors[cid % len(colors)]
        return (r << 16) | (g << 8) | b

def main():
    rclpy.init()
    node = PostFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
