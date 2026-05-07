from subprocess import run


def convert_to_opus(*, source_filename, target_filename, bitrate: str):
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
