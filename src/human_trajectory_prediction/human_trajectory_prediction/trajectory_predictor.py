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
        
        self.publisher = self.create_publisher(MarkerArray, '/cm_mot/predicted_markers', 10)
        
        self.subscriber = self.create_publisher(MarkerArray, '/cm_mot/predicted_markers', 10)

        # Create a subscription to the "trajectory_source" topic
        self.subscriber = self.create_subscription(
            MarkerArray,
            '/cm_mot/track_markers',
            self.listener_callback,
            10
        )

        self.trajectory_window = MarkerArray()

        self.get_logger().info(f'Starting trajectory predictor')

    def listener_callback(self, msg):
        for marker in msg.markers:
            id = str(marker.id)
            # Append to markerarray 

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