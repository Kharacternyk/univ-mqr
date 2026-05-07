from subprocess import run


def convert_to_opus(*, source_filename: str, target_filename: str, bitrate: str):
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            source_filename,
            "-c:a",
            "libopus",
            "-b:a",
            bitrate,
            target_filename,
        ],
        check=True,
        capture_output=True,
    )
