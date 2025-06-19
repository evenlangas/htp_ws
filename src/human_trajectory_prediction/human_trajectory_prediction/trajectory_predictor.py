import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
import math
import time
from visualization_msgs.msg import Marker, MarkerArray
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler # Scikitlearn 1.3.0, tensorflow 2.13.0
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
        super().__init__('trajectory_data_publisher')
        self.last_process_time = 0
        self.publisher = self.create_publisher(MarkerArray, '/cm_mot/predicted_markers', 10)
        self.subscriber = self.create_publisher(MarkerArray, '/cm_mot/predicted_markers', 10)
        # self.model_path = "~/htp_ws/src/human_trajectory_prediction/models/htp_model_transformer.h5"
        self.model_path = os.path.expanduser("~/htp_ws/src/human_trajectory_prediction/models/htp_model_transformer.h5")
        self.model = None
        self.model_loaded = False
        self.max_prediction_freq = 15 # Hz
        self.load_model()

        # Create a subscription to the "trajectory_source" topic
        self.subscriber = self.create_subscription(
            MarkerArray,
            '/cm_mot/track_markers',
            self.listener_callback,
            10
        )

        self.publisher = self.create_publisher(MarkerArray, '/cm_mot/predicted_markers', 10)

        self.trajectory_window = MarkerArray()

        self.get_logger().info(f'Starting trajectory predictor')

    def listener_callback(self, msg):
        for marker in msg.markers:
            if str(marker.id).startswith('4'):
                self.trajectory_window.markers.append(marker)
                # Maintain sliding window of 30 markers
                if len(self.trajectory_window.markers) > 30:
                    # Remove the oldest marker (first in the list)
                    self.trajectory_window.markers.pop(0)
        self.trajectory_window.markers.sort(key=lambda marker: marker.header.stamp.sec + marker.header.stamp.nanosec * 1e-9)
        current_time = time.time()
    
        # Throttle to max 10 Hz
        if current_time - self.last_process_time < 1/self.max_prediction_freq:
            return  # Skip this callback
        
        self.last_process_time = current_time
        self.predict_markers()

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

    def predict_markers(self):
        """Predict future trajectory markers using the loaded transformer model"""
        
        # Check if we have enough markers and model is loaded
        if len(self.trajectory_window.markers) < 30:
            self.get_logger().warn(f'Not enough markers for prediction: {len(self.trajectory_window.markers)}/30')
            return
        
        if self.model is None:
            self.get_logger().warn('Model not loaded. Loading model now...')
            self.load_model()
            if not self.model_loaded:
                return
        
        try:
            # Extract x, y positions from the last 30 markers
            positions = []
            for marker in self.trajectory_window.markers[-30:]:  # Get last 30 markers
                positions.append([marker.pose.position.x, marker.pose.position.y])
            
            # Convert to numpy array and reshape for model input
            # Shape: (1, 30, 2) - batch_size=1, sequence_length=30, features=2
            input_data = np.array(positions).reshape(1, 30, 2)
            
            # Optional: Normalize the data if your model was trained with normalization
            # You might need to store the scaler used during training
            # input_data = self.scaler.transform(input_data.reshape(-1, 2)).reshape(1, 30, 2)
            
            # Make prediction
            prediction = self.model.predict(input_data, verbose=0)
            
            # prediction shape depends on your model output
            # Assuming it predicts next N positions: (1, N, 2)
            predicted_positions = prediction[0]  # Remove batch dimension
            
            # Create predicted markers
            predicted_markers = MarkerArray()
            
            # Get the latest marker for reference (timestamp, frame_id, etc.)
            latest_marker = self.trajectory_window.markers[-1]
            
            # Create predicted markers
            for i, (pred_x, pred_y) in enumerate(predicted_positions):
                predicted_marker = Marker()
                
                predicted_marker.header.frame_id = "map"
                # Copy header info from latest marker
                predicted_marker.header = latest_marker.header
                predicted_marker.header.stamp = self.get_clock().now().to_msg()
                
                # Set marker properties
                predicted_marker.id = int(f"4{latest_marker.id}00{i}")  # Unique ID for predicted markers
                predicted_marker.type = Marker.LINE_STRIP
                predicted_marker.action = Marker.ADD
                
                # Set position (predicted)
                predicted_marker.pose.position.x = float(pred_x)
                predicted_marker.pose.position.y = float(pred_y)
                predicted_marker.pose.position.z = 0.0
                
                # Copy orientation from latest marker
                predicted_marker.pose.orientation = latest_marker.pose.orientation
                
                # Set scale (make predicted markers slightly smaller/different)
                predicted_marker.scale.x = 0.3
                
                # Set color (different color for predictions, e.g., red with transparency)
                predicted_marker.color.r = 1.0
                predicted_marker.color.g = 0.0
                predicted_marker.color.b = 0.0
                predicted_marker.color.a = 0.7  # Semi-transparent
                
                # Set lifetime
                predicted_marker.lifetime.sec = 1  # Markers disappear after 1 second
                
                predicted_markers.markers.append(predicted_marker)
            
            # Publish predicted markers
            self.publisher.publish(predicted_markers)
            
            self.get_logger().info(f'Published {len(predicted_markers.markers)} predicted markers')
            
        except Exception as e:
            self.get_logger().error(f'Error in prediction: {str(e)}')
            import traceback
            self.get_logger().error(f'Traceback: {traceback.format_exc()}')

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