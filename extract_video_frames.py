'''
python extract_video_frames.py --video_path ./inputVideo/studetClass.mp4 --out_frames_path ./detect_frames --frame_rate 1
该代码的作用是对视频进行抽帧，
三个输入参数，
第一个video_path，是输入视频的路径，
第二个out_frames_path，是裁剪的帧输出的位置
第三个frame_rate，是每秒裁剪的帧数。
'''

import argparse
import os
import json


# 视频抽帧
def extract_video_frames(video_path, out_frames_path, frame_rate):
    # 创建输出目录（如果不存在）
    if not os.path.exists(out_frames_path):
        print(f"{out_frames_path} doesn't exist. Creating it.")
        os.makedirs(out_frames_path)

    # 提取视频名字
    # 利用os.path模块获取文件名和后缀名
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 抽帧并保存到输出目录中
    out_name = os.path.join(out_frames_path, f"{video_name}_%06d.jpg")
    command = f'ffmpeg -i "{video_path}" -r "{frame_rate}" -q:v 1 "{out_name}"'
    os.system(command)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="path to input video file")
    parser.add_argument("--out_frames_path", type=str, required=True, help="path to out frames file")
    parser.add_argument("--frame_rate", type=str, required=True, help="number of cropped video frames per second")
    args = parser.parse_args()

    extract_video_frames(args.video_path, args.out_frames_path, args.frame_rate)


