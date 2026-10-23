"""Shared numerical helpers for the three packaged SWT workflows."""
import shutil
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter, PillowWriter

def animation_writer(video_format, fps):
    """Return a portable GIF writer or an FFmpeg-backed MP4 writer."""
    if video_format == "gif":
        return PillowWriter(fps=fps)
    if video_format != "mp4":
        raise ValueError(f"Unknown video format: {video_format}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError(
                "MP4 output requires ffmpeg. Use VIDEO_FORMAT=gif or install "
                "ffmpeg/imageio-ffmpeg in the cluster environment."
            ) from error
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    return FFMpegWriter(
        fps=fps,
        metadata={"title": "TTO spin-wave dynamic structure factor"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
