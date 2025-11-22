import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix

class GNSSPublisher(Node):

    def __init__(self):
        super().__init__('gnss_publisher_node')
        self.publisher_ = self.create_publisher(NavSatFix, '/senti/gnss_pos_llh', 10)
        self.timer = self.create_timer(1.0, self.publish_gnss_data)
        self.get_logger().info('GNSS Publisher Node Initialized')

    def publish_gnss_data(self):
        gnss_data = NavSatFix()
        gnss_data.latitude = 37.7749  # Example latitude value
        gnss_data.longitude = -122.4194  # Example longitude value
        gnss_data.altitude = 0.0  # Example altitude value
        self.publisher_.publish(gnss_data)
        self.get_logger().info('Published GNSS data')

def main(args=None):
    rclpy.init(args=args)
    gnss_publisher = GNSSPublisher()
    rclpy.spin(gnss_publisher)
    gnss_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

