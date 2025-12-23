import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import numpy as np
import cv2
from ultralytics import YOLO

from scipy.spatial.transform import Rotation as R


class PostFusionNode(Node):

    def __init__(self):
        super().__init__('post_fusion_node')

        self.get_logger().info("Initializing Post-Fusion Node...")

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

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/fusion/markers',
            10
        )

        self.bridge = CvBridge()

        self.get_logger().info("Loading YOLO model...")
        #self.yolo = YOLO("/home/shepherd/ws_ROBOCON/models/bestl.pt")
        self.yolo = YOLO("/home/shepherd/ws_ROBOCON/models/bestx.pt")
        self.get_logger().info("YOLO model loaded successfully")

        self.latest_image = None

        # Camera intrinsics(d455)
        #self.fx = 642.3447
        #self.fy = 641.3488
        #self.cx = 648.6090
        #self.cy = 362.6446

        # Camera intrinsics(d435)
        self.fx = 606.1944 
        self.fy = 606.1345 
        self.cx = 321.8288 
        self.cy = 250.1631


        # Extrinsics (lidar → camera)
        # 此参数为待定值，需根据实际标定结果修改
        #self.R_lc = np.array([
        #    [0.0, -1.0,  0.0],
        #    [0.0,  0.0, -1.0],
         #   [1.0,  0.0,  0.0]
        #])
        #self.T_lc = np.array([0.03, -0.022, -0.04])

        # Extrinsics (lidar → camera) measured
        self.R_lc = np.array([
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0]
        ])
        self.T_lc = np.array([0.061, -0.011, -0.068])

        self.get_logger().info("Post-Fusion Node Started")
        self.get_logger().info("Subscribed to /livox/lidar and /camera/image")

    # ================= Image Callback =================
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.get_logger().debug(
            f"Image received: {msg.width} x {msg.height}"
        )

    # ================= Lidar Callback =================
    def lidar_callback(self, msg):
        self.get_logger().debug("Lidar callback triggered")

        if self.latest_image is None:
            self.get_logger().warn(
                "No image received yet, skipping this lidar frame"
            )
            return
        # 将雷达点云图转为 numpy 数组
        points = self.pointcloud2_to_xyz(msg)
        self.get_logger().info(
            f"PointCloud received: {points.shape[0]} points"
        )

        if points.shape[0] == 0:
            self.get_logger().warn("Empty point cloud, skipping")
            return
        # 进行 DBSCAN 聚类,得到少量有效点簇
        clusters = self.cluster_points(points)
        self.get_logger().info(
            f"DBSCAN clusters found: {len(clusters)}"
        )
        # 运行 YOLO 检测图像中的物体
        detections = self.run_yolo(self.latest_image)
        self.get_logger().info(
            f"YOLO detections: {len(detections)}"
        )
        # Marker 初始化
        markers = MarkerArray()

        # 清空上一帧 Marker
        clear_marker = Marker()
        clear_marker.header.frame_id = msg.header.frame_id
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = ""
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        marker_id = 0
        matched_any = False
        #将每一簇点云投影到图像平面
        for cluster in clusters:
            uv = self.project_cluster(cluster)
            if uv.size == 0:
                self.get_logger().debug("Cluster projected outside camera FOV")
                continue
        #画出投影边界框    
            h, w = self.latest_image.shape[:2]
            bbox_2d = self.get_2d_bbox(uv, w, h)
            if bbox_2d is None:
                self.get_logger().debug("Invalid 2D bbox from projected cluster")
                continue
        # 与每个 YOLO 结果计算 IoU，匹配成功则生成 3D Marker
            for det in detections:
                iou = self.compute_iou(bbox_2d, det['bbox'])
                if iou > 0.10:
                    matched_any = True
                    self.get_logger().info(
                        f"Matched cluster with {det['class_name']} (IoU={iou:.2f})"
                    )

                    bcube_marker, text_marker = self.create_3d_marker(
                        cluster,
                        det['class_name'],
                        marker_id,
                        msg.header.frame_id
                    )

                    markers.markers.append(bcube_marker)
                    markers.markers.append(text_marker)

                    marker_id += 1
        #发布Marker
        self.marker_pub.publish(markers)
        self.get_logger().info(
            f"Published MarkerArray with {marker_id} 3D boxes"
        )

        if not matched_any:
            self.get_logger().warn("No LiDAR–Camera matches found in this frame")

    # ================= Processing =================
    def cluster_points(self, points):
        from sklearn.cluster import DBSCAN
        from pyransac3d import Plane
        valid_points = points[
            #(points[:, 2] > 0.02) &
            #(points[:, 2] < 0.5) &
            (points[:, 0] > 0.0)
        ]# 过滤掉地面、天花板、雷达后方的点

        if valid_points.shape[0] < 20:
            return []
        
        plane = Plane()
        try:
            plane_eq, inliers = plane.fit(valid_points, thresh=0.02, maxIteration=1000)
        except Exception:
            return []
        non_ground_points = np.delete(valid_points, inliers, axis=0)
        if non_ground_points.shape[0] < 20:
            return []        

        # 进行 DBSCAN 聚类
        db = DBSCAN(eps=0.15, min_samples=20).fit(non_ground_points)
        clusters = []
        for label in set(db.labels_):
            if label == -1:
                continue
            clusters.append(non_ground_points[db.labels_ == label])
        return clusters

    def run_yolo(self, image):
        results = self.yolo(image, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_name': self.yolo.names[cls]
            })
        return detections

    def project_cluster(self, cluster):
        #通过转化矩阵将点云从雷达坐标系转到相机坐标系
        pts_cam = (self.R_lc @ cluster.T).T + self.T_lc
        pts_cam = pts_cam[pts_cam[:, 2] > 0]

        if pts_cam.shape[0] == 0:
            return np.array([])
        #计算投影后的像素坐标
        u = self.fx * pts_cam[:, 0] / pts_cam[:, 2] + self.cx
        v = self.fy * pts_cam[:, 1] / pts_cam[:, 2] + self.cy
        return np.stack([u, v], axis=1)

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / (areaA + areaB - inter + 1e-6)

    # ================= Marker =================
    def create_3d_marker(self, cluster, label, marker_id, frame_id):
        min_pt = np.min(cluster, axis=0)
        max_pt = np.max(cluster, axis=0)

        mean_x = (min_pt[0]+max_pt[0])/2
        mean_y = (min_pt[1]+max_pt[1])/2   
        mean_z = (min_pt[2]+max_pt[2])/2
    
        bcube = Marker()
        bcube.header.frame_id = frame_id
        bcube.header.stamp = self.get_clock().now().to_msg()
        bcube.ns = "fusion_bcube"
        bcube.id = marker_id
        bcube.type = Marker.CUBE
        bcube.action = Marker.ADD
        bcube.pose.position.x = mean_x
        bcube.pose.position.y = mean_y
        bcube.pose.position.z = mean_z
        bcube.color.a = 0.5
        bcube.pose.orientation.w = 1.0
        bcube.scale.x = 0.25
        bcube.scale.y= 0.25
        bcube.scale.z= 0.25

        if label == "red":
            bcube.color .r, bcube.color.g, bcube.color.b = 1.0, 0.0, 0.0
        elif label == "green":
            bcube.color .r, bcube.color.g, bcube.color.b = 0.0, 1.0, 0.0
        elif label == "blue":
            bcube.color .r, bcube.color.g, bcube.color.b = 0.0, 0.0, 1.0
        elif label == "grey":
            bcube.color .r, bcube.color.g, bcube.color.b = 0.5, 0.5, 0.5
        else:
            bcube.color .r, bcube.color.g, bcube.color.b = 1.0, 1.0, 1.0

        text = Marker()
        text.header = bcube.header
        text.ns = "fusion_text"
        text.id = marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = (min_pt[0] + max_pt[0]) / 2
        text.pose.position.y = (min_pt[1] + max_pt[1]) / 2
        text.pose.position.z = max_pt[2] + 0.3
        text.pose.orientation.w = 1.0
        text.scale.z = 0.4
        text.color = bcube.color
        text.text = label

        return bcube, text

    def _pt(self, xyz):
        p = Point()
        xyz = np.array(xyz, dtype=float)
        p.x, p.y, p.z = xyz
        return p

    # ================= Utils =================
    def pointcloud2_to_xyz(self, msg):
        from sensor_msgs_py import point_cloud2
        points = []
        for p in point_cloud2.read_points(msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        return np.array(points)

    def get_2d_bbox(self, uv, img_w, img_h):
        if uv.shape[0] < 5:
            return None
        u, v = uv[:, 0], uv[:, 1]
        x1, y1 = np.clip(np.min(u), 0, img_w - 1), np.clip(np.min(v), 0, img_h - 1)
        x2, y2 = np.clip(np.max(u), 0, img_w - 1), np.clip(np.max(v), 0, img_h - 1)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None
        return [x1, y1, x2, y2]


def main():
    rclpy.init()
    node = PostFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()