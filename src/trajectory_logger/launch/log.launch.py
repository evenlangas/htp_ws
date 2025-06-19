from launch import LaunchDescription
from launch_ros.actions import Node
from datetime import datetime
import os
import subprocess

def generate_launch_description():
    # Generate a timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"log_{timestamp}"

    # Full path for the rosbag and CSV
    filepath = "/home/cogniman/human-trajectory-prediction/htp_ws/src/trajectory_data/"
    full_bag_path = os.path.join(filepath, base_filename)

    # Launch the static transform publisher
    static_transform_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="os_pose",
        output="screen",
        arguments=["0", "0", "14.10", "0.0", "0.0", "3.145", "world", "os_sensor"],
    )

    # Launch the trajectory logger
    trajectory_logger_node = Node(
        package="trajectory_logger",
        executable="trajectory_logger",
        name="trajectory_logger",
        output="screen",
        parameters=[{"csv_filename": f"{base_filename}"}],
    )

    # Run ros2 bag record via subprocess
    def start_rosbag():
        subprocess.Popen([
            "ros2", "bag", "record",
            "-o", full_bag_path,
            "/cm_od/cloud",
            "/cm_mot/object_detection_pose_array",
            "/tf",
            "/tf_static",
            "-d 60"
        ])

    # Start the ros2 bag recording
    start_rosbag()

    return LaunchDescription([
        static_transform_node,
        trajectory_logger_node,
    ])
