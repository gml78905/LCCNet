#!/bin/bash

# ============================================================================
# Quick Start: Modify this command and run
# ============================================================================
python3 train_with_sacred.py with \
    sensor_mode='lidar' \
    epochs=120 \
    max_t=0.5 \
    max_r=5.0 \
    checkpoint_name='lidar_r5_t_0.5'
    
python3 train_with_sacred.py with \
    sensor_mode='radar' \
    epochs=120 \
    max_t=0.5 \
    max_r=5.0 \
    checkpoint_name='radar_r5_t_0.5'

python3 train_with_sacred.py with \
    sensor_mode='both' \
    epochs=120 \
    max_t=0.5 \
    max_r=5.0 \
    checkpoint_name='both_r5_t_0.5'
    

# ============================================================================
# LCCNet Training Script Examples
# ============================================================================
# 
# Usage:
#   - Uncomment the example you want to run, or modify it for your needs
#   - Run: bash run_train_examples.sh
#   - Or run directly: python3 train_with_sacred.py with <config>
#
# Note: For lists in Sacred, you can use JSON format like '["item1", "item2"]'
# ============================================================================

# Example 1: Train with lidar sensor, specific train and val scenes
# python3 train_with_sacred.py with \
#     sensor_mode='lidar' \
#     train_scene='["parking_lot_2", "parking_lot_4", "parking_lot_5"]' \
#     val_scene='["library_1"]' \
#     epochs=120 \
#     batch_size=120

# Example 2: Train with radar sensor, specific train and val scenes  
# python3 train_with_sacred.py with \
#     sensor_mode='radar' \
#     train_scene='["parking_lot_1", "parking_lot_2"]' \
#     val_scene='["parking_lot_4"]' \
#     epochs=120 \
#     batch_size=120

# Example 3: Train with lidar sensor, using all scenes except val_scene for training
# (train_scene=None means use all scenes except val_scene)
# python3 train_with_sacred.py with \
#     sensor_mode='lidar' \
#     train_scene=None \
#     val_scene='["library_1"]' \
#     epochs=120

# Example 4: Train with custom checkpoint name
# python3 train_with_sacred.py with \
#     sensor_mode='lidar' \
#     train_scene='["parking_lot_2", "parking_lot_4"]' \
#     val_scene='["library_1"]' \
#     checkpoint_name='my_custom_name' \
#     epochs=120

# Example 5: Train with different hyperparameters
# python3 train_with_sacred.py with \
#     sensor_mode='lidar' \
#     train_scene='["parking_lot_2"]' \
#     val_scene='["library_1"]' \
#     max_r=10.0 \
#     max_t=1.0 \
#     BASE_LEARNING_RATE=3e-4 \
#     epochs=200

# Example 6: Single scene for training (as string, will be converted to list)
# python3 train_with_sacred.py with \
#     sensor_mode='lidar' \
#     train_scene='parking_lot_2' \
#     val_scene='library_1' \
#     epochs=120

# Example 7: Train with both lidar and radar sensors
# python3 train_with_sacred.py with \
#     sensor_mode='both' \
#     train_scene='["parking_lot_2", "parking_lot_4"]' \
#     val_scene='["library_1"]' \
#     epochs=120 \
#     batch_size=120



