
'''
python yolo2ava.py  --video_path ./inputVideo/studetClass.mp4 --threshold_change 0.3 --out_crop_video_path ./out_crop_video --out_frames_path ./outFrames --annotations_path ./annotations --weights_HRW yolov7_4.2k_HRW.pt  --weights_yolo yolov7.pt --conf_HRW 0.8 --conf_yolo 0.25 --project_HRW ./runs/detect/HRW --project_yolo ./runs/detect/person --frame_rate 1 --detect_frames_path ./detect_frames

python yolo2ava.py  --video_path ./inputVideo/studetClass.mp4 \
                    --detect_frames_path ./detect_frames \
                    --frame_rate 1 \
                    --weights_HRW yolov7_4.2k_HRW.pt  \
                    --weights_yolo yolov7.pt \
                    --conf_HRW 0.8 \
                    --conf_yolo 0.25 \
                    --project_HRW ./runs/detect/HRW \
                    --project_yolo ./runs/detect/person \
                    --threshold_change 0.3 \
                    --out_crop_video_path ./out_crop_video \
                    --out_frames_path ./outFrames \
                    --annotations_path ./annotations

'''
import argparse
import os
import json
import csv
import pickle
import numpy as np
import random


# 视频抽帧 用于检测
def extract_video_detect_frames(video_path, detect_frames_path, frame_rate):
    # 创建输出目录（如果不存在）
    if not os.path.exists(detect_frames_path):
        print(f"{detect_frames_path} doesn't exist. Creating it.")
        os.makedirs(detect_frames_path)

    # 提取视频名字
    # 利用os.path模块获取文件名和后缀名
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 抽帧并保存到输出目录中
    out_name = os.path.join(detect_frames_path, f"{video_name}_%06d.jpg")
    command = f'ffmpeg -i "{video_path}" -r "{frame_rate}" -q:v 1 "{out_name}"'
    os.system(command)

# 转换yolo格式的候选框为左上(x1,y1)，右下角的坐标值(x2,y2)
def yolo_to_xywh(bbox):
    x, y, w, h = float(bbox['center_x']), float(bbox['center_y']), float(bbox['width']), float(bbox['height'])
    x1 = (x - w / 2)
    # 保证x1，y1不小于0
    x1 = x1 if x1 > 0 else 0
    y1 = (y - h / 2)
    y1 = y1 if y1 > 0 else 0
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2

def get_image_nums(txt_dir):
    # 获取最后一个txt文件的文件名和编号
    txt_list = sorted(os.listdir(txt_dir))
    last_txt = txt_list[-1]
    last_num = int(last_txt.split("_")[-1].split(".")[0])
    return last_num

def generate_index_list(last_num):
    # 生成需要查找的索引列表
    index_list = []
    for n in range(1, last_num+1):
        index = 7*(n-1) + 4
        if index > last_num-3:
            break
        index_list.append(index)
    return index_list

# 人数改变判断或者画面跳转判断
def number_of_people_chang(file_name, pesons_json, threshold_change):

    base_num = int(file_name.split("_")[1].split(".")[0])
    num_list = [-3, -2, -1, 0, 1, 2, 3]

    temp_num = -1
    for num in num_list:
        new_num = base_num + num
        new_str = f"studetClass_{new_num:06d}.txt"



        # pesons_json.get(file_name, {})。
        # 如果file_name在pesons_json中存在，则返回该值；否则返回一个空字典{}。
        # 这样即使file_name在pesons_json中不存在也不会抛出异常了。
        # 这样即使在pesons_json中没有找到对应的filename，也可以返回{'persons': '0'}的结果了。?
        peson_num = int(pesons_json.get(new_str, {}).get('persons', '0'))

        # 第一次进来 temp_num 为 -1，先把第一个值赋给 temp_num
        if temp_num == -1:
            temp_num = peson_num
            continue

        # 判断前一个数字和当前的数字差值不超过threshold_change(30%)
        elif abs(peson_num - temp_num) > threshold_change * max(temp_num, peson_num):
            # 返回 False 代表 画面跳转
            return False

        temp_num = peson_num
    # 返回 True 代表 没有出现画面跳转
    return True


def pesons_info_extract(persons_txt_path):
    results = {}

    for filename in sorted(os.listdir(persons_txt_path)):
        if filename.endswith('.txt'):
            filepath = os.path.join(persons_txt_path, filename)
            with open(filepath, 'r') as f:
                count = 0
                for line in f:
                    line = line.strip().split()
                    if int(line[0]) == 0:
                        count += 1
                results[filename] = {'persons': str(count)}

    pesons_json = json.dumps(results, indent=4)
    return pesons_json



def HRW_info_extract(video_path, txt_path, index_list, pesons_json, threshold_change):
    
    # 提取视频名字
    # 利用os.path模块获取文件名和后缀名
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # generate file names using the indexes provided
    file_names = [f'{video_name}_{index:06d}.txt' for index in index_list]
    
    pesons_json = json.loads(pesons_json)
    
    # loop through the file_names list and check if each file exists in the directory
    label_dict = {}

    index_list_filter = []

    for file_name in file_names:
        if not os.path.isfile(os.path.join(txt_path, file_name)):
            continue
        
        # 判断当前帧和前后3帧的人数是否存大的波动（人数变化超过40%），
        # 如果波动大，那么就舍弃该帧
        if not number_of_people_chang(file_name, pesons_json, threshold_change):
            print(file_name,"人数波动大，舍弃帧")
            continue
        

        with open(os.path.join(txt_path, file_name), 'r') as f:
            label_content = []
            for line in f:
                class_id, center_x, center_y, width, height = line.strip().split(' ')
                label_content.append({
                    'class_id': int(class_id),
                    'center_x': float(center_x),
                    'center_y': float(center_y),
                    'width': float(width),
                    'height': float(height),
                })
            image_idx = int(os.path.splitext(os.path.basename(file_name))[0].split('_')[1])
            label_dict[file_name] = {
                'image_idx': image_idx,
                'label_content': label_content,
            }
            index_list_filter.append(image_idx)
    HRW_json = json.dumps(label_dict, indent=4)
    return HRW_json, index_list_filter


# 裁剪视频
def crop_videos(video_path, HRW_json, out_crop_video_path):

    # 创建输出目录（如果不存在）
    if not os.path.exists(out_crop_video_path):
        print(f"{out_crop_video_path} doesn't exist. Creating it.")
        os.makedirs(out_crop_video_path)

    HRW_json = json.loads(HRW_json)
    # 遍历json文件
    for key, value in HRW_json.items():
        # 获取帧名，作为裁剪的视频名
        file_name = os.path.splitext(os.path.basename(key))[0]
        out_name = f"{out_crop_video_path}/{file_name}.mp4"

        start_time = int(value['image_idx']) - 3

        command = f'ffmpeg -ss {start_time} -t 7 -i "{video_path}" "{out_name}" '
		
        os.system(command)




# 视频抽帧
def extract_video_frames(crop_video_path, out_frames_path):
    # 创建输出目录（如果不存在）
    if not os.path.exists(out_frames_path):
        print(f"{out_frames_path} doesn't exist. Creating it.")
        os.makedirs(out_frames_path)

    # 遍历视频文件夹中的所有视频
    for video_name in os.listdir(crop_video_path):
        video_path = os.path.join(crop_video_path, video_name)

        # 提取视频名字
        if video_name.endswith(".webm"):
            video_name = video_name[:-5]
        else:
            video_name = video_name[:-4]

        # 创建输出目录（如果不存在）
        out_video_dir = os.path.join(out_frames_path, video_name)
        if not os.path.exists(out_video_dir):
            os.makedirs(out_video_dir)

        # 抽帧并保存到输出目录中
        out_name = os.path.join(out_video_dir, f"{video_name}_%06d.jpg")
        command = f'ffmpeg -i "{video_path}" -r 30 -q:v 1 "{out_name}"'
        os.system(command)

# 生成train.csv或者val.csv
def train_val_csv(HRW_json, annotations_path):
    # 遍历json数据并将其转换为csv格式

    HRW_json = json.loads(HRW_json)
    
    # 生成 0 到 9999 之间的随机整数
    rand_num = random.randint(0, 9999)
    # 格式化为四位数格式的字符串
    four_digit_num = '{:04d}'.format(rand_num)
    train_val_file = 'train_val_'+ four_digit_num + '.csv'
    train_val_csv_path = os.path.join(annotations_path, train_val_file)
    with open(train_val_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in HRW_json.items():
            video_name = os.path.splitext(os.path.basename(key))[0]

            frame_ids = [2,3,4]
            for frame_id in frame_ids:
                for i, label in enumerate(value['label_content'], start=1):
                    x1, y1, x2, y2 = yolo_to_xywh(label)
                    person_id = i
                    class_id = label['class_id']
                    writer.writerow([video_name, frame_id, x1, y1, x2, y2, class_id, person_id])

# 生成 proposals_train.pkl 或者 proposals_val.pkl
def proposals_train_val(HRW_json, annotations_path):
    # 遍历json数据并将其转换为pkl格式

    HRW_json = json.loads(HRW_json)
    # 生成 0 到 9999 之间的随机整数
    rand_num = random.randint(0, 9999)
    # 格式化为四位数格式的字符串
    four_digit_num = '{:04d}'.format(rand_num)
    proposals_train_val_file = 'proposals_train_val_'+ four_digit_num + '.pkl'
    proposals_train_val_file_path = os.path.join(annotations_path, proposals_train_val_file)

    output_dict = dict()

    for filename, label_info in HRW_json.items():
        video_name = os.path.splitext(os.path.basename(filename))[0]
        frame_ids = [2,3,4]
        for frame_id in frame_ids:
            value = []

            for label in label_info["label_content"]:
                x1, y1, x2, y2 = yolo_to_xywh(label)
                key = f"{video_name},{frame_id}"
                value.append([x1, y1, x2, y2])
            
            output_dict[key] = np.array(value)

    with open(proposals_train_val_file_path, "wb") as f:
        pickle.dump(output_dict, f)



if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="path to input video file")
    parser.add_argument("--persons_txt_path", type=str, required=True, help="path to persons txt")
    parser.add_argument("--threshold_change", type=float, required=True, help="filter out fluctuations in the number of people or screen switching thresholds")
    parser.add_argument("--out_crop_video_path", type=str, required=True, help="path to out video file")
    parser.add_argument("--out_frames_path", type=str, required=True, help="path to out frames file")
    parser.add_argument("--annotations_path", type=str, required=True, help="path to annotations file")
    parser.add_argument('--weights_HRW', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--weights_yolo', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--conf_HRW', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--conf_yolo', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--project_HRW', default='runs/detect', help='save results to project/name')
    parser.add_argument('--project_yolo', default='runs/detect', help='save results to project/name')
    parser.add_argument("--frame_rate", type=str, required=True, help="number of cropped video frames per second")
    parser.add_argument("--detect_frames_path", type=str, required=True, help="path to detect frames file")
    args = parser.parse_args()

    input("请手动清空：\n "\
          + args.persons_txt_path +  "\n"\
          + args. +  "\n"\
          + args.out_crop_video_path +  "\n"\
          + args.out_frames_path +  "\n"\
          + args.annotations_path +  "\n"\
          + args.project_HRW +  "\n"\
          + args.project_yolo +  "\n"
          + args.detect_frames_path +  "\n"
          )

    # 
    extract_video_detect_frames(args.video_path, args.detect_frames_path, args.frame_rate)

    # HRW 对视频帧进行检测 
    command_HRW = 'python detect.py --weights ' + str(args.weights_HRW[0]) + ' --conf ' + str(args.conf_HRW) + ' --img-size 640 ' + ' --source ' + str(args.detect_frames_path) + ' --save-txt ' + ' --project ' + str(args.project_HRW)
    
    # 执行命令
    os.system(command_HRW)

    #yolo  对视频帧进行检测
    command_yolo = 'python detect.py --weights ' + str(args.weights_yolo[0]) + ' --conf ' + str(args.conf_yolo) + ' --img-size 640 ' + ' --source ' + str(args.detect_frames_path) + ' --save-txt ' + ' --project ' + str(args.project_yolo)
    # 执行命令

    os.system(command_yolo)

    # 提取出每一帧的人数
    persons_txt_path = os.path.join(args.project_yolo,'exp/labels')
    pesons_json = pesons_info_extract(persons_txt_path)

    # 获取最后一个txt文件编号和需要查找的索引列表
    # last_num 是获取 HRW检测结果中最后一帧的名字
    HRW_txt_path = os.path.join(args.project_HRW, 'exp/labels')
    last_num = get_image_nums(HRW_txt_path)

    # index_list是一个列表，7n+4,n=0,1,2,3...，直到last_num-3
    index_list = generate_index_list(last_num)
    print("index_list:",index_list)

    # 处理所有要查找的HRW txt文件，并保存结果到json文件
    HRW_json, index_list_filter = HRW_info_extract(args.video_path, HRW_txt_path, index_list, pesons_json, args.threshold_change)

    print("index_list_filter:",index_list_filter)
    print("HRW_json:",HRW_json)


    # 裁剪视频
    crop_videos(args.video_path, HRW_json, args.out_crop_video_path)

    # 视频抽帧
    extract_video_frames(args.out_crop_video_path, args.out_frames_path)

    # 生成annotations相关文件
    train_val_csv(HRW_json, args.annotations_path)
    proposals_train_val(HRW_json, args.annotations_path)
    # touch train_excluded_timestamps.csv
    # touch val_excluded_timestamps.csv
    # touch test_excluded_timestamps.csv
    '''
    touch included_timestamps.txt
    2
    3
    4 
    '''
    '''
    touch action_list.pbtxt
    item {
        name: "hand_raising"
        id: 1
    }
    item {
        name: "reading"
        id: 2
    }
    item {
        name: "writing"
        id: 3
    }
    '''
    
    

    

