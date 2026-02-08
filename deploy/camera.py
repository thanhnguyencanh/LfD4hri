import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    def __init__(self,
                 device_id,
                 width=640,
                 height=480,
                 fps=6): # D455 15 fps
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.scale = None
        self.intrinsics = None

    def connect(self):
        # Start and configure
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(str(self.device_id))
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        cfg = self.pipeline.start(config)

        # Determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()

        # Determine depth scale
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        return self.intrinsics

    def get_image_bundle(self):
        frames = self.pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.first(rs.stream.color)
        aligned_depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asarray(aligned_depth_frame.get_data(), dtype=np.float32)
        depth_image *= self.scale

        # depth hole(0) inpaint
        depth_image = cv2.copyMakeBorder(depth_image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (depth_image == 0).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(depth_image).max()
        depth_image = depth_image.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        depth_image = cv2.inpaint(depth_image, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        depth_image = depth_image[1:-1, 1:-1]
        depth_image = depth_image * scale
        # depth_image = np.expand_dims(depth_image, axis=-1)

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.expand_dims(depth_image, axis=2)

        return {
            'rgb': color_image,
            'aligned_depth': depth_image,
            'aligned_depth_frame': aligned_depth_frame,
        }

    def plot_image_bundle(self):
        images = self.get_image_bundle()

        rgb = images['rgb']
        depth = images['aligned_depth']

        fig, ax = plt.subplots(1, 2, squeeze=False)
        ax[0, 0].imshow(rgb)
        m, s = np.nanmean(depth), np.nanstd(depth)
        ax[0, 1].imshow(depth.squeeze(axis=2), vmin=m - s, vmax=m + s, cmap=plt.cm.gray)
        ax[0, 0].set_title('rgb')
        ax[0, 1].set_title('aligned_depth')

        plt.show()

    def show_realtime(self):
        """Show RGB and depth streams in realtime using OpenCV. Press 'q' to quit."""
        logger.info("Starting realtime view. Press 'q' to quit.")
        try:
            while True:
                images = self.get_image_bundle()
                rgb = images['rgb']
                depth = images['aligned_depth'].squeeze(axis=2)

                # Convert RGB to BGR for OpenCV display
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Normalize depth using mean ± std for better contrast
                m, s = np.nanmean(depth), np.nanstd(depth)
                depth_vis = np.clip(depth, m - s, m + s)
                depth_vis = ((depth_vis - (m - s)) / (2 * s) * 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # Stack side by side
                combined = np.hstack((bgr, depth_colormap))
                cv2.imshow('RealSense - RGB | Depth (press q to quit)', combined)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    cam = RealSenseCamera(device_id=943222070907)
    cam.connect()
    while True:
        # cam.plot_image_bundle()
        cam.show_realtime()
