import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2
from ultralytics import YOLO


class YoloGridNode(Node):

    def __init__(self):
        super().__init__('yolo_grid_node')

        self.sub_image = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.yolo = YOLO("/home/shepherd/ws_ROBOCON/models/bestx.pt")

        self.get_logger().info("YOLO Grid Node Started")

    # ================= Image Callback =================
    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        detections = self.run_yolo(image)
        if len(detections) != 4:
            self.get_logger().warn("YOLO detection != 4, skip drawing")
            return

        # 在图像上画框 + 文字
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = f"{det['class_name']} | {det['position']}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow("YOLO 2x2 Grid", image)
        cv2.waitKey(1)

    # ================= YOLO =================
    def run_yolo(self, image):
        results = self.yolo(image, verbose=False)[0]
        boxes = results.boxes

        if boxes is None or len(boxes) != 4:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        # box 中心点（像素坐标）
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

        detections = []
        for row in range(2):
            for col in range(2):
                i = grid_idx[row][col]
                detections.append({
                    "bbox": xyxy[i].tolist(),
                    "class_name": self.yolo.names[clss[i]],
                    "position": pos_name[(row, col)]
                })

        return detections

    # ================= 排序逻辑（完全复用你的） =================
    def centroids_to_2x2_grid(self, centroids):
        # 按 y 从小到大（上 → 下）
        idx_y = np.argsort(centroids[:, 1])
        top_idx = idx_y[:2]
        bottom_idx = idx_y[2:]

        # 每一行按 x 从小到大（左 → 右）
        top = top_idx[np.argsort(centroids[top_idx, 0])]
        bottom = bottom_idx[np.argsort(centroids[bottom_idx, 0])]

        return [
            [int(top[0]), int(top[1])],
            [int(bottom[0]), int(bottom[1])]
        ]


def main():
    rclpy.init()
    node = YoloGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
