import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, Image, PointField
from cv_bridge import CvBridge

import numpy as np
import cv2
import struct

from ultralytics import YOLO
from sklearn.cluster import DBSCAN


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

        self.sub_image = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        # ================= Publishers =================
        self.debug_image_pub = self.create_publisher(
            Image,
            '/fusion/debug_image',
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

        # ================= YOLO =================
        self.get_logger().info("Loading YOLO model ...")
        self.yolo = YOLO("/home/shepherd/ws_ROBOCON/models/bestl.pt")
        self.get_logger().info("YOLO loaded")

        # ================= Camera Intrinsics =================
        self.fx = 642.3447
        self.fy = 641.3488
        self.cx = 648.6090
        self.cy = 362.6446

        # ================= LiDAR → Camera Extrinsic =================
        self.lidar_to_camera = np.array([
            [0.0, -1.0,  0.0,  0.03 ],
            [0.0,  0.0, -1.0, -0.022],
            [1.0,  0.0,  0.0, -0.04 ],
            [0.0,  0.0,  0.0,  1.0  ]
        ])

        self.R_lc = self.lidar_to_camera[:3, :3]
        self.T_lc = self.lidar_to_camera[:3, 3]

        self.get_logger().info("=== Node Ready ===")

    # =====================================================
    # Image Callback
    # =====================================================
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    # =====================================================
    # LiDAR Callback
    # =====================================================
    def lidar_callback(self, msg):

        if self.latest_image is None:
            self.get_logger().warn("No image yet, skip lidar")
            return

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

        # ===== YOLO =====
        detections = self.run_yolo(self.latest_image)
        debug_img = self.latest_image.copy()
        h, w = debug_img.shape[:2]

        # ===== LiDAR Cluster Projection =====
        for cid, cluster in enumerate(clusters):
            uv = self.project_cluster(cluster)
            if uv.size == 0:
                continue

            bbox_lidar = self.get_2d_bbox(uv, w, h)
            if bbox_lidar is not None:
                self.draw_bbox(debug_img, bbox_lidar, (0, 0, 255), text=f"LiDAR_{cid}")

        # ===== YOLO Visualization =====
        for det in detections:
            self.draw_bbox(debug_img, det['bbox'], (0, 255, 0), text=det['class_name'])

        # ===== publish debug image =====
        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        debug_msg.header = msg.header
        self.debug_image_pub.publish(debug_msg)

    # =====================================================
    # Processing
    # =====================================================
    def cluster_points(self, points):
        import open3d as o3d
        valid_points = points[
            (points[:, 2] > 0.02) &
            (points[:, 2] < 0.5) &
            (points[:, 0] > 0.0)
        ]# 过滤掉地面、天花板、雷达后方的点

        if valid_points.shape[0] < 20:
            return np.empty((0,3)), []
        # 进行 DBSCAN 聚类
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000
        )
        non_ground_cloud = pcd.select_by_index(inliers, invert=True)
        if len(non_ground_cloud.points) < 20:
            return np.empty((0,3)),[]
        labels = np.array(
            non_ground_cloud.cluster_dbscan(eps=0.15, min_points=20)
        )
        clusters = []
        pts = np.asarray(non_ground_cloud.points)

        for cid in np.unique(labels):
            if cid == -1:
                continue
            clusters.append(pts[labels == cid])

        return pts, clusters

    def run_yolo(self, image):
        results = self.yolo(image, verbose=False)[0]
        dets = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            dets.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': self.yolo.names[cls]
            })

        return dets

    # =====================================================
    # LiDAR → Camera Projection (齐次坐标)
    # =====================================================
    def project_cluster(self, cluster):

        ones = np.ones((cluster.shape[0], 1))
        pts_lidar_h = np.hstack([cluster, ones])           # (N,4)
        pts_cam_h = (self.lidar_to_camera @ pts_lidar_h.T).T
        pts_cam = pts_cam_h[:, :3]

        pts_cam = pts_cam[pts_cam[:, 2] > 0]
        if pts_cam.shape[0] == 0:
            return np.array([])

        u = self.fx * pts_cam[:, 0] / pts_cam[:, 2] + self.cx
        v = self.fy * pts_cam[:, 1] / pts_cam[:, 2] + self.cy

        return np.stack([u, v], axis=1)

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

    def get_2d_bbox(self, uv, w, h):
        if uv.shape[0] < 5:
            return None

        x1 = np.clip(np.min(uv[:, 0]), 0, w - 1)
        y1 = np.clip(np.min(uv[:, 1]), 0, h - 1)
        x2 = np.clip(np.max(uv[:, 0]), 0, w - 1)
        y2 = np.clip(np.max(uv[:, 1]), 0, h - 1)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        return [x1, y1, x2, y2]

    def draw_bbox(self, img, bbox, color, text=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if text:
            cv2.putText(img, text, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    rclpy.init()
    node = PostFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
