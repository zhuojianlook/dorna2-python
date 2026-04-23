#!/usr/bin/env python3
"""
Live 3D point-cloud visualization for Intel RealSense D405 with high-accuracy preset.

Controls:
  - Press 'S' in the Open3D window to save a snapshot to 'snapshot.ply'.
  - Press 'Q' in the Open3D window to quit.
"""
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Globals to hold the latest frames and pointcloud helper
_pc = None
_last_depth = None
_last_color = None


def save_snapshot(vis):
    global _pc, _last_depth, _last_color
    if _pc and _last_color:
        _pc.map_to(_last_color)
        points = _pc.calculate(_last_depth)
        _pc.export_to_ply('snapshot.ply', _last_color)
        print('Saved snapshot.ply')
    return False


def quit_vis(vis):
    vis.close()
    return False


def main():
    global _pc, _last_depth, _last_color

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth,  640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color,  640, 480, rs.format.bgr8, 15)
    profile = pipeline.start(config)

    # High-accuracy preset
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset,
                            rs.rs400_visual_preset.high_accuracy)

    # PointCloud helper
    _pc = rs.pointcloud()

    # Set up Open3D visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='RealSense Point Cloud')
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # Register key callbacks
    vis.register_key_callback(ord('S'), save_snapshot)
    vis.register_key_callback(ord('Q'), quit_vis)

    try:
        while vis.poll_events():  # Loop until window closed
            # Wait for frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Update global frames
            _last_depth = depth_frame
            _last_color = color_frame

            # Compute point cloud vertices
            _pc.map_to(color_frame)
            points = _pc.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            pcd.points = o3d.utility.Vector3dVector(verts)

            # Update visualization
            vis.update_geometry(pcd)
            vis.update_renderer()
    finally:
        pipeline.stop()
        vis.destroy_window()


if __name__ == '__main__':
    main()
