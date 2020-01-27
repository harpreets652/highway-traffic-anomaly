import cv2
import os
import argparse
import skvideo.io

"""
Script to fix video orientations.
"""


def all_videos(input_args):
    video_counter = 0
    for root, sub_dirs, video_files in os.walk(input_args.source_dir):
        for video_file in video_files:
            if not video_file.endswith(".mov"):
                continue

            input_video_path = os.path.join(root, video_file)
            output_video_path = os.path.join(input_args.destination, video_file)

            process_video(input_video_path, output_video_path)

            video_counter += 1
            if video_counter % 50 == 0:
                print(f"{video_counter} videos processed")

        print(f"{video_counter} total videos processed")
    return


def specific_video(input_args):
    video_file = os.path.basename(input_args.source_video)
    output_video_path = os.path.join(input_args.destination, video_file)

    process_video(input_args.source_video, output_video_path)
    return


def process_video(input_video_path, output_video_path):
    input_video = cv2.VideoCapture(input_video_path)
    if not input_video.isOpened():
        print(f"Unable to open input video {input_video_path}")
        return

    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(input_video.get(cv2.CAP_PROP_FPS))

    rotation = get_rotation(input_video_path)

    if rotation in (270, 90):
        frame_size = (frame_height, frame_width)
    else:
        frame_size = (frame_width, frame_height)

    output_video = cv2.VideoWriter(output_video_path,
                                   cv2.VideoWriter_fourcc('a', 'v', 'c', '1'),
                                   frame_rate,
                                   frame_size)

    while True:
        frame_exists, frame = input_video.read()

        if not frame_exists:
            break

        if rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        output_video.write(frame)

    input_video.release()
    output_video.release()

    return


def get_rotation(input_video_path):
    meta_data = skvideo.io.ffprobe(input_video_path)

    # meta_data['video']['tag'][0]['@key'] == 'rotate'
    for tag in meta_data['video']['tag']:
        if tag['@key'] == 'rotate':
            return int(tag['@value'])

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=False, help="Directory containing videos to process")
    parser.add_argument("--source_video", required=False, help="Path to specific video to process")
    parser.add_argument("--destination", required=True, help="Destination folder to save processed videos")

    input_arguments = parser.parse_args()

    if input_arguments.source_dir:
        all_videos(input_arguments)
    elif input_arguments.source_video:
        specific_video(input_arguments)
    else:
        raise NotImplementedError("Either provide a source directory or specific video.")
