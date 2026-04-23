#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

def main():
    # 1. Configure RealSense pipeline
    pipeline = rs.pipeline()
    config   = rs.config()

    # If you have multiple devices connected, uncomment and set your serial:
    # config.enable_device('YOUR_DEVICE_SERIAL')

    # Enable the color stream at 640×480 @ 30fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    try:
        pipeline.start(config)
    except Exception as e:
        print("Failed to start RealSense pipeline:", e, file=sys.stderr)
        sys.exit(1)

    cv2.namedWindow('D405 Color Stream', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # 2. Wait for a coherent pair of frames: color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 3. Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # 4. (Insert your processing here)
            #    e.g. gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 5. Display
            cv2.imshow('D405 Color Stream', frame)

            # 6. Exit on 'q' or ESC
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        # 7. Clean up
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
