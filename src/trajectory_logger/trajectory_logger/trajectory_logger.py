import rclpy
from rclpy.node import Node
import json
import math
import csv
import os
from std_msgs.msg import String  # Assuming the topic publishes JSON as a String message
from visualization_msgs.msg import MarkerArray

class TrajectoryLogger(Node):
    def __init__(self):
        super().__init__('trajectory_logger')

        # Declare and get parameter for CSV filename (without extension)
        self.declare_parameter('csv_filename', 'trajectory_log')
        filename_param = self.get_parameter('csv_filename').get_parameter_value().string_value

        # Construct full path to CSV file
        home_dir = os.path.expanduser("~")
        csv_dir = os.path.join(home_dir, 'human-trajectory-prediction', 'htp_ws', 'src', 'trajectory_data')
        os.makedirs(csv_dir, exist_ok=True)  # Ensure directory exists
        csv_filepath = os.path.join(csv_dir, f"{filename_param}.csv")
        
        # Throw an error if the file already exists
        if os.path.exists(csv_filepath):
            self.get_logger().error(f"File {csv_filepath} already exists. Exiting.")
            # Exit the node immediately
            raise FileExistsError(f"File {csv_filepath} already exists.")
        
        # Create a subscription to the "trajectory_source" topic
        self.subscription = self.create_subscription(
            MarkerArray,
            '/cm_mot/track_markers',
            self.listener_callback,
            10
        )

        # Open a CSV file for logging
        self.csv_file = open(csv_filepath, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['id_prefix', 'id', 'x', 'y', 'velocity_scalar', 'orientation', 'timestamp'])  # Write CSV header

        self.get_logger().info('Trajectory Logger Node started.')

    def listener_callback(self, msg):
        for marker in msg.markers:
            try:
                object_id = str(marker.id)
                
                # Filter by the first digit of the ID
                if not object_id.startswith('5') and not object_id.startswith('4'):
                    continue

                # Extract position, velocity, and orientation
                x = marker.pose.position.x
                y = marker.pose.position.y
                vel = marker.scale.x

                # Orientation quaternion
                q_x = marker.pose.orientation.x
                q_y = marker.pose.orientation.y
                q_z = marker.pose.orientation.z
                q_w = marker.pose.orientation.w

                # Compute yaw from quaternion
                yaw = math.atan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2))
                yaw_degrees = math.degrees(yaw)
                if yaw_degrees < 0:
                    yaw_degrees += 360

                # Extract timestamp from the marker's header
                timestamp = str(marker.header.stamp.sec)
                timestamp += str(marker.header.stamp.nanosec)

                # Write the parsed data to the CSV file
                self.csv_writer.writerow([object_id[0], object_id[1] + object_id[2], x, y, vel, yaw_degrees, timestamp])

                self.get_logger().info(f"Logged: id={object_id}, x={x}, y={y}, vel={vel}, yaw={yaw_degrees}, timestamp={timestamp}")

            except Exception as e:
                self.get_logger().error(f"Error processing marker with id={marker.id}: {e}")


    def destroy_node(self):
        # Close the CSV file on node shutdown
        self.csv_file.close()
        self.get_logger().info('CSV file closed.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
