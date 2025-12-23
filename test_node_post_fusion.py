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
        self.yolo = YOLO("/home/shepherd/ws_ROBOCON/models/bestx.pt")
        self.get_logger().info("YOLO model loaded successfully")

        self.latest_image = None

        # Camera intrinsics(d435)
        self.fx = 606.1944 
        self.fy = 606.1345 
        self.cx = 321.8288 
        self.cy = 250.1631

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
        marker_array = MarkerArray()

        # 清空上一帧 Marker
        clear_marker = Marker()
        clear_marker.header.frame_id = msg.header.frame_id
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = ""
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        marker_id = 0
        matched_any = False

        # 取出yolo识别中右下角的框
        bottom_right_det = bottom_left_det = top_right_det = top_left_det = None
        for det in detections:
            if det['position'] == 'bottom_right':
                bottom_right_det = det
            if det['position'] == 'bottom_left':
                bottom_left_det = det
            if det['position'] == 'top_right':
                top_right_det = det
            if det['position'] == 'top_left':
                top_left_det = det  

        if None in [bottom_right_det, bottom_left_det, top_right_det, top_left_det]:
            self.get_logger().warn("No detection found, skipping fusion")
            return

        # 将每一簇点云投影到图像平面
        for cluster in clusters:
            uv = self.project_cluster(cluster)
            if uv.size == 0:
                self.get_logger().debug("Cluster projected outside camera FOV")
                continue
        # 画出投影边界框    
            h, w = self.latest_image.shape[:2]
            bbox_2d = self.get_2d_bbox(uv, w, h)
            if bbox_2d is None:
                self.get_logger().debug("Invalid 2D bbox from projected cluster")
                continue  
             
        # 与每个 YOLO 结果计算 IoU，匹配成功则生成 3D Marker
            iou = self.compute_iou(bbox_2d, bottom_right_det['bbox'])
            if iou > 0.10:
                matched_any = True
                self.get_logger().info( 
                    f"Matched bottom-right cluster with {bottom_right_det['class_name']} (IoU={iou:.2f})"
                )

                bottom_right_cluster_centroid = {
                    'xyz':np.mean(cluster, axis=0),
                    'label':'bottom_right',
                    'cls':bottom_right_det['class_name']
                }
                bottom_left_cluster_centroid = {
                    'xyz':bottom_right_cluster_centroid['xyz'] + np.array([-0.85, 0.0, 0.0]),
                    'label':'bottom_left',
                    'cls':bottom_left_det['class_name']
                }
                top_right_cluster_centroid = {
                    'xyz':bottom_right_cluster_centroid['xyz'] + np.array([0.0, 0.85, 0.0]),
                    'label':'top_right',
                    'cls':top_right_det['class_name']
                }
                top_left_cluster_centroid = {
                    'xyz':bottom_left_cluster_centroid['xyz'] + np.array([0.0, 0.85, 0.0]),
                    'label':'top_left',
                    'cls':top_left_det['class_name']
                }
                clusters_centroids = [
                    top_left_cluster_centroid,
                    top_right_cluster_centroid,
                    bottom_left_cluster_centroid,
                    bottom_right_cluster_centroid
                ]

                created_array = self.create_3d_marker(
                    clusters_centroids,
                    msg.header.frame_id
                )
                
                for m in created_array.markers:
                    m.id = marker_id
                    marker_array.markers.append(m)
                    marker_id += 1

        #发布Marker
        self.marker_pub.publish(marker_array)
        self.get_logger().info(
            f"Published MarkerArray with {marker_id} 3D boxes"
        )

        if not matched_any:
            self.get_logger().warn("No Bottom-Right matches found in this frame")

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
        boxes = results.boxes

        num = 0
        if boxes is not None:   
            num = len(boxes)
            self.get_logger().info(f"YOLO detected {num} objects")
        if num != 4:
            self.get_logger().warn("YOLO did not detect 4 objects as expected")
            return []
        
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        centroids = np.column_stack([
            (xyxy[:, 0] + xyxy[:, 2]) / 2.0,
            (xyxy[:, 1] + xyxy[:, 3]) / 2.0
        ])

        grid_idx = self.centroids_to_2x2_grid(centroids)
        pos_name = {
            (0, 0): "top_left",
            (0, 1): "top_right",
            (1, 0): "bottom_left",
            (1, 1): "bottom_right",
        }
        detections= []
        for row in range(2):
            for col in range(2):
                i = grid_idx[row][col]
                detections.append({
                    "bbox": xyxy[i].tolist(),          
                    "class_name": self.yolo.names[clss[i]],
                    "position": pos_name[(row, col)]
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
    
    def centroids_to_2x2_grid(self, centroids):
        # 1. 按 y 从小到大排序（上 -> 下）
        idx_y = np.argsort(centroids[:, 1])

        top_idx = idx_y[:2]
        bottom_idx = idx_y[2:]

        # 2. 每一行按 x 从小到大排序（左 -> 右）
        top = top_idx[np.argsort(centroids[top_idx, 0])]
        bottom = bottom_idx[np.argsort(centroids[bottom_idx, 0])]

        # 3. 组成 2x2 网格
        grid_idx = [
            [int(top[0]),    int(top[1])],
            [int(bottom[0]), int(bottom[1])]
        ]

        return grid_idx   

    # ================= Marker =================
    def create_3d_marker(self, clusters_centroids,frame_id):
        marker_array = MarkerArray()
        for centroid in clusters_centroids:
            mean_x, mean_y, mean_z = centroid['xyz']
            label = centroid['cls']

            bcube = Marker()
            bcube.header.frame_id = frame_id
            bcube.header.stamp = self.get_clock().now().to_msg()
            bcube.ns = "fusion_bcube"
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
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = centroid['xyz'][0]
            text.pose.position.y = centroid['xyz'][1]
            text.pose.position.z = centroid['xyz'][2] + 0.3
            text.pose.orientation.w = 1.0
            text.scale.z = 0.4
            text.color = bcube.color
            text.text = label

            marker_array.markers.append(bcube)
            marker_array.markers.append(text)

        return marker_array

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