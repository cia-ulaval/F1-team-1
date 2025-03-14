"""
ros2_controller.py

Publishes Twist commands to /cmd_vel in ROS2 to control a robot or sim.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class GestureROSController(Node):
    """
    ROS2 node that publishes Twist to /cmd_vel.
    """

    def __init__(self):
        super().__init__('gesture_ros_controller')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("GestureROSController node initialized.")

    def publish_action(self, action: str) -> None:
        """
        Convert an action string ("FORWARD", "LEFT", "RIGHT", "STOP")
        into a Twist message, publish to /cmd_vel.
        """
        twist = Twist()

        if action == Movement.FORWARD:
            twist.linear.x = 1.0
        elif action == Movement.LEFT:
            twist.angular.z = 1.0
        elif action == Movement.RIGHT:
            twist.angular.z = -1.0
        elif action == Movement.STOP:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Action {action} => {twist}")

def create_ros_node():
    """
    Convenience function you can call from main.py to create & return the node.
    """
    rclpy.init()
    node = GestureROSController()
    return node

def shutdown_ros_node(node: GestureROSController):
    """
    Cleanly shutdown the node and rclpy.
    """
    node.destroy_node()
    rclpy.shutdown()
