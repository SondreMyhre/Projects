import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

class GNSSListener(Node):

    def __init__(self):
        super().__init__('gnss_listener_node')
        self.subscription = self.create_subscription(
            NavSatFix,
            '/senti/gnss_pos_llh',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(
            f'Latitude: {msg.latitude}, Longitude: {msg.longitude}')

def main(args=None):
    rclpy.init(args=args)
    gnss_listener = GNSSListener()
    rclpy.spin(gnss_listener)
    gnss_listener.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
