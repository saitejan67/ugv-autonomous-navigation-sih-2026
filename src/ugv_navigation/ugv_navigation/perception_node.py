import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from ugv_navigation.perception import process_bgr_image


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10
        )
        self.traversability_publisher = self.create_publisher(
            Image, '/perception/traversability', 10)
        self.obstacle_publisher = self.create_publisher(
            Detection2DArray, '/perception/obstacles', 10)
        self.debug_publisher = self.create_publisher(
            Image, '/perception/debug_image', 10)

        self.get_logger().info('UGV Perception Node started')
        self.get_logger().info(
            'Waiting for camera: /camera'
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

            result = process_bgr_image(frame)
            traversability = self.bridge.cv2_to_imgmsg(
                result.traversability_mask, encoding='mono8')
            traversability.header = msg.header
            self.traversability_publisher.publish(traversability)

            obstacles = Detection2DArray()
            obstacles.header = msg.header
            for index, (x, y, width, height) in enumerate(result.obstacles):
                detection = Detection2D()
                detection.header = msg.header
                detection.id = str(index)
                detection.bbox.center.position.x = x + width / 2.0
                detection.bbox.center.position.y = y + height / 2.0
                detection.bbox.size_x = float(width)
                detection.bbox.size_y = float(height)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = 'obstacle_like_region'
                hypothesis.hypothesis.score = min(
                    1.0, (width * height) / float(frame.shape[0] * frame.shape[1]))
                detection.results.append(hypothesis)
                obstacles.detections.append(detection)
            self.obstacle_publisher.publish(obstacles)

            debug = self.bridge.cv2_to_imgmsg(result.debug_image, encoding='bgr8')
            debug.header = msg.header
            self.debug_publisher.publish(debug)

        except (ValueError, cv2.error) as error:
            self.get_logger().error(
                f'Perception image processing failed: {error}'
            )


def main(args=None):
    rclpy.init(args=args)

    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # ros2 launch can send a second SIGINT during teardown.
            pass


if __name__ == '__main__':
    main()
