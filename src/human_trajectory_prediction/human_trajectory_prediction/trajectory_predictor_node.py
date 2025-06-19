#!/home/even/miniconda3/envs/ros2_humble/bin/python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler # Scikitlearn 1.3.0, tensorflow 2.13.0
from collections import deque
import pandas as pd
import os


class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__('trajectory_predictor_node')
        
        # Parameters
        self.declare_parameter('model_path', 'models/htp_model_transformer.h5')
        self.declare_parameter('input_window', 30)
        self.declare_parameter('output_window', 30)
        self.declare_parameter('input_topic', '/human_pose')
        self.declare_parameter('prediction_topic', '/predicted_trajectory')
        self.declare_parameter('current_pose_topic', '/current_human_pose')
        
        # Get parameters
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.data_path = self.get_parameter('data_path').get_parameter_value().string_value
        self.input_window = self.get_parameter('input_window').get_parameter_value().integer_value
        self.output_window = self.get_parameter('output_window').get_parameter_value().integer_value
        
        # Initialize data structures
        self.pose_buffer = deque(maxlen=self.input_window)
        self.scaler = MinMaxScaler()
        self.model = None
        self.model_loaded = False
        
        # Initialize scaler with real data
        self.initialize_scaler()
        
        # Publishers
        self.prediction_publisher = self.create_publisher(
            Path, 
            self.get_parameter('prediction_topic').get_parameter_value().string_value, 
            10
        )
        self.current_pose_publisher = self.create_publisher(
            PoseStamped,
            self.get_parameter('current_pose_topic').get_parameter_value().string_value,
            10
        )
        
        # Subscriber
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            self.get_parameter('input_topic').get_parameter_value().string_value,
            self.pose_callback,
            10
        )
        
        # Load model
        self.load_model()
        
        self.get_logger().info(f'Trajectory Predictor Node initialized')
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Data path: {self.data_path}')
        self.get_logger().info(f'Input window: {self.input_window}, Output window: {self.output_window}')

    def initialize_scaler(self):
        """Initialize the scaler using real data from CSV file"""
        try:
            # Check if data file exists
            if not os.path.exists(self.data_path):
                self.get_logger().error(f'Data file not found: {self.data_path}')
                # Fallback to dummy data
                dummy_data = np.array([[-4, -4], [12, 12]])
                self.scaler.fit(dummy_data)
                self.get_logger().warn('Using dummy data for scaler initialization')
                return
            
            # Read the CSV file
            df = pd.read_csv(self.data_path)
            
            # Extract x, y coordinates
            coordinates = df[['x', 'y']].values
            
            # Fit the scaler with real data
            self.scaler.fit(coordinates)
            
            # Log data statistics
            x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
            y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
            
            self.get_logger().info(f'Scaler initialized with real data from {self.data_path}')
            self.get_logger().info(f'Data range - X: [{x_min:.3f}, {x_max:.3f}], Y: [{y_min:.3f}, {y_max:.3f}]')
            self.get_logger().info(f'Total data points: {len(coordinates)}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load data from {self.data_path}: {str(e)}')
            # Fallback to dummy data
            dummy_data = np.array([[-4, -4], [12, 12]])
            self.scaler.fit(dummy_data)
            self.get_logger().warn('Using dummy data for scaler initialization due to error')

    def load_model(self):
        """Load the trained model from H5 file"""
        try:
            self.model = keras.models.load_model(
                self.model_path, 
                custom_objects={'TransformerBlock': TransformerBlock}
            )
            self.model_loaded = True
            self.get_logger().info(f'Model loaded successfully from {self.model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            self.model_loaded = False

    def pose_callback(self, msg):
        """Callback for incoming pose messages"""
        if not self.model_loaded:
            return
            
        # Extract x, y from pose
        x = msg.pose.position.x
        y = msg.pose.position.y
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Add to buffer
        self.pose_buffer.append([x, y, timestamp])
        
        # Publish current pose
        self.publish_current_pose(msg)
        
        # Make prediction if we have enough data
        if len(self.pose_buffer) >= self.input_window:
            self.make_prediction(msg.header)

    def publish_current_pose(self, pose_msg):
        """Publish the current human pose"""
        self.current_pose_publisher.publish(pose_msg)

    def make_prediction(self, header):
        """Make trajectory prediction using the loaded model"""
        try:
            # Prepare input data
            input_data = np.array([[pos[0], pos[1]] for pos in self.pose_buffer])
            
            # Normalize input data
            input_normalized = self.scaler.transform(input_data)
            
            # Reshape for model input (1, input_window, num_features)
            input_sequence = input_normalized.reshape(1, self.input_window, 2)
            
            # Make prediction
            prediction = self.model.predict(input_sequence, verbose=0)
            
            # Denormalize prediction
            prediction_reshaped = prediction.reshape(-1, 2)
            prediction_denormalized = self.scaler.inverse_transform(prediction_reshaped)
            
            # Create and publish path message
            self.publish_prediction(prediction_denormalized, header)
            
        except Exception as e:
            self.get_logger().error(f'Prediction failed: {str(e)}')

    def publish_prediction(self, prediction, header):
        """Publish the predicted trajectory as a Path message"""
        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = 'map'  # Adjust frame_id as needed
        
        # Create poses for each predicted point
        for i, (x, y) in enumerate(prediction):
            pose_stamped = PoseStamped()
            pose_stamped.header = header
            pose_stamped.header.frame_id = 'map'
            
            # Set position
            pose_stamped.pose.position.x = float(x)
            pose_stamped.pose.position.y = float(y)
            pose_stamped.pose.position.z = 0.0
            
            # Set orientation (identity quaternion)
            pose_stamped.pose.orientation.x = 0.0
            pose_stamped.pose.orientation.y = 0.0
            pose_stamped.pose.orientation.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose_stamped)
        
        self.prediction_publisher.publish(path_msg)
        
        self.get_logger().debug(f'Published prediction with {len(prediction)} points')


def main(args=None):
    rclpy.init(args=args)
    
    node = TrajectoryPredictorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()