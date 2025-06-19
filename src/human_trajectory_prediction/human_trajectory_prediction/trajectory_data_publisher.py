import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import math
import time
from visualization_msgs.msg import Marker, MarkerArray

class TrajectoryDataPublisherNode(Node):
    def __init__(self):
        super().__init__('trajectory_data_publisher')
        
        self.df = pd.read_csv(f'~/turtlebot3_ws/src/human_trajectory_prediction/data/real_data_cleaned.csv')
        features = list(self.df.columns)
        features.remove("id_prefix")
        features.remove("workstation")
        features.remove("start")
        features.remove("goal")
        self.df = self.df[features]

        self.publisher = self.create_publisher(MarkerArray, '/cm_mot/track_markers', 10)

        self.current_row = 0

        self.timer = self.create_timer(0.1, self.publish_next_point)

        self.get_logger().info(f'Starting trajectory publisher with {len(self.df)} points')

    def publish_next_point(self):
        row = self.df.iloc[self.current_row]
        
        marker = Marker()
        
        marker.id = int(float("4" + str(row['trajectory_id'])))
        
        marker.pose.position.x = float(row['x'])
        marker.pose.position.y = float(row['y'])
        marker.pose.position.z = 0.0
        
        vel = marker.scale.x = float(row['velocity_scalar'])

        # Convert yaw angle to quaternion
        yaw_radians = math.radians(float(row['orientation']))
        q_x = 0.0
        q_y = 0.0
        q_z = math.sin(yaw_radians / 2.0)
        q_w = math.cos(yaw_radians / 2.0)
        marker.pose.orientation.x = q_x
        marker.pose.orientation.y = q_y
        marker.pose.orientation.z = q_z
        marker.pose.orientation.w = q_w

        timestamp_ns = row['timestamp']
        marker.header.stamp.sec = timestamp_ns // 1_000_000_000  # Integer division for seconds
        marker.header.stamp.nanosec = timestamp_ns % 1_000_000_000  # Remainder for nanoseconds

        markers = MarkerArray()

        markers.markers.append(marker)

        # Publish the point
        self.publisher.publish(markers)
        
        self.get_logger().info(f'Published point {self.current_row}: x={marker.pose.position.x:.2f}, y={marker.pose.position.y:.2f}')
        
        # Move to next row, loop back to start when reaching the end
        self.current_row = (self.current_row + 1) % len(self.df)

def main(args=None):
    rclpy.init(args=args)
    
    node = TrajectoryDataPublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()