import os
from pathlib import Path
import argparse
import shutil

"""
Script to search for train/test sets in DeepDrive and move to separate directories 
"""


def main(input_args):
    Path(input_args.destination).mkdir(parents=True, exist_ok=True)

    with open(input_args.query_list, 'r') as query_file:
        file_lines = query_file.read().splitlines()
        query_files_set = set(file_lines)

    video_counter = 0
    total_video_counter = 0
    for root, sub_dirs, files in os.walk(input_args.source_dir):
        for file in files:
            total_video_counter += 1
            if file in query_files_set:
                video_counter += 1
                dest_loc = shutil.copy(os.path.join(root, file), input_args.destination)
                if input_args.verbose:
                    print(f"copied: {dest_loc}")

    if input_args.verbose:
        print(f"Total # of videos: {total_video_counter}, Number of videos copied: {video_counter}")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True, help="Directory containing videos to search")
    parser.add_argument("--query_list", required=True, help="List of filenames to search in source_dir")
    parser.add_argument("--destination", required=True, help="Destination folder for copies")
    parser.add_argument("--verbose", action="store_true", help="Print copy filenames")

    input_arguments = parser.parse_args()
    main(input_arguments)
