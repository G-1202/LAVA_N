import os
import subprocess
import shutil
import math
import json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from modules import format_conversion, read_gt, read_detections, calculate_f1_for_class
from adaptation import get_video_bit

re = [[1920, 1080], [1280, 720], [720, 480], [320, 240]]
# resolutions = ['1080p', '720p', '480p', '240p']
skip = [0, 1, 2, 5]
QP = [23, 28, 33, 38, 43]
button = 80
fps = 30

Frame = 'dataset/AD'
Video = 'pareto/video'


def encoder(image_folder, video_name, start, frames, w, h, fps, skip, qp):
    f = fps / (skip + 1)
    directory = os.path.dirname(video_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-start_number', str(start),
        '-i', os.path.join(image_folder, '%02d.jpg'),
        '-vf', f"select='not(mod(n\,{skip + 1}))',setpts=N/({f}*TB),scale={w}:{h}",
        '-frames:v', str(frames // (skip + 1)),
        '-r', str(f),
        '-x264-params', f'qp={qp}',  # Set the QP value
        video_name
    ]
    subprocess.run(ffmpeg_command, check=True)


def detect_data(video_name, output_folder, num, frame):
    if os.path.isdir(f'runs/detect/{output_folder}/{num}'):
        shutil.rmtree(f'runs/detect/{output_folder}/{num}')
    if os.path.isdir(f'pareto/input/{num}'):
        shutil.rmtree(f'pareto/input/{num}')
    yolo = YOLO('yolov8s.pt', task="detect")
    yolo(source=video_name, save_txt=True, save_conf=True, name=f'{output_folder}/{num}')
    new_p = f'pareto/input/{num}'
    folder = os.path.exists(new_p)
    if not folder:
        os.makedirs(new_p)
    format_conversion(f'runs/detect/{output_folder}/{num}/labels', f'{new_p}/', 1920, 1080)
    parent_directory = f'pareto/input'
    rewrite(parent_directory, num, frame)


def calculate_marco_f1(gt_name, det_name, num):
    gt_path = f'pareto/input/{gt_name}'
    det_path = f'pareto/input/{det_name}'
    gt_files = [os.path.join(gt_path, f"{gt_name}_{i}.txt") for i in range(1, 61)]
    det_files = [os.path.join(det_path, f"{num}_{i}.txt") for i in range(1, 61)]
    # gt_files = get_files(gt_path, f"{gt_name}_")
    # det_files = get_files(det_path, f"{det_name}_")

    all_gt_boxes = defaultdict(list)
    all_pred_boxes = defaultdict(list)

    for gt_file, det_file in zip(gt_files, det_files):
        for box in read_gt(gt_file):
            all_gt_boxes[box[0]].append(box[1:])
        for box in read_detections(det_file):
            all_pred_boxes[box[0]].append(box[1:])

    f1_scores = []
    for cls in all_gt_boxes:
        gt_boxes = all_gt_boxes[cls]
        pred_boxes = all_pred_boxes[cls]
        f1 = calculate_f1_for_class(gt_boxes, pred_boxes)
        f1_scores.append(f1)

    marco_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return marco_f1


# def skip_truth(s, t, skip):
#     source = f'pareto/input/{s}'
#     target = f'pareto/input/{t}'
#     if os.path.isdir(target):
#         shutil.rmtree(target)
#     os.makedirs(target)
#     source_files = os.listdir(source)
#     target_files = []
#     num = 1
#     for i, source_file in enumerate(source_files):
#         if i % skip == 0:
#             target_file = f'{t}_{num}.txt'
#             shutil.copy(os.path.join(source, f'{s}_{i + 1}.txt'), os.path.join(target, target_file))
#             num = num + 1

def recovery_label(num, skip):
    source = f'pareto/input/{num}'
    target = f'pareto/input/{num}_r'
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target)
    for i in range(1, math.ceil(60/(skip+1))+1):
        src_file_name = f"{num}_{i}.txt"  # Original file name
        shutil.copy(os.path.join(source, src_file_name), os.path.join(target, f'{num}_{1+(skip+1)*(i-1)}.txt'))
        for j in range(1, skip + 1):
            if 1+(skip+1)*(i-1)+j > 60:
                break
            dst_file_name = f"{num}_{1+(skip+1)*(i-1)+j}.txt"  # New file name
            src_file_path = os.path.join(source, src_file_name)
            dst_file_path = os.path.join(target, dst_file_name)
            shutil.copy(src_file_path, dst_file_path)


def rewrite(v_path, num, frame):
    folder_name = os.path.join(v_path, str(num))
    files = [f for f in os.listdir(folder_name) if f.endswith('.txt')]
    if len(files) < frame:
        for j in range(1, frame+1):
            file_name = os.path.join(folder_name, f"{num}_{j}.txt")
            if not os.path.exists(file_name):
                with open(file_name, 'w') as fp:
                    pass


def base_label(folder_path):
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    for file_name in txt_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
        column_to_delete = 1
        for i in range(len(lines)):
            columns = lines[i].split()
            del columns[column_to_delete]
            lines[i] = ' '.join(columns) + '\n'
        with open(file_path, 'w') as file:
            file.writelines(lines)


def creat_pareto(matrix):
    lst = [(i, j, matrix[i][j][0], matrix[i][j][1]) for i in range(len(matrix)) for j in range(len(matrix[0]))]
    lst.sort(key=lambda x: x[2], reverse=True)
    return lst


def pareto_main():
    # c = np.empty((4, 4), dtype=object)
    c = np.empty((button, 5), dtype=object)
    train_c = np.empty((300, button), dtype=object)
    for k in range(1, 301):
        if os.path.isdir(Video):
            shutil.rmtree(Video)
        num = 1
        for k in range(5):
            for i in range(4):
                for j in range(4):
                    encoder(f'{Frame}/{k}', f'{Video}/{num}.mp4',
                            0, 60, re[i][0], re[i][1], fps, skip[j], QP[k])
                    f = math.floor(60 / (skip[j] + 1))
                    bit = get_video_bit(f'{Video}/{num}.mp4')
                    detect_data(f'{Video}/{num}.mp4', 'pareto', num, f)
                    if i == 0 and j == 0:
                        base_label(f"pareto/input/1")
                        train_c[k - 1][4 * i + j] = [bit, 1]
                        # c[i][j] = [bit, 1]
                    else:
                        if j == 0:
                            f = calculate_marco_f1('1', f'{num}', num)
                            train_c[k - 1][4 * i + j] = [bit, f]
                        else:
                            recovery_label(num, skip[j])
                            f = calculate_marco_f1('1', f'{num}_r', num)
                            train_c[k - 1][4 * i + j] = [bit, f]
                    num = num + 1

    train_c_list = train_c.tolist()
    train_c_json = json.dumps(train_c_list)
    file_path_json = 'train_c.json'
    with open(file_path_json, 'w') as file:
        file.write(train_c_json)

    for k in range(5):

        for i in range(4):
            for j in range(4):
                first_items = np.array([item[0] for item in train_c[:, K*16+4 * i + j]])
                second_items = np.array([item[1] for item in train_c[:, K*16+4 * i + j]])
                bit_mean = np.mean(first_items)
                f_mean = np.mean(second_items)
                c[K*16+4 * i + j][0] = k
                c[K*16+4 * i + j][1] = j
                c[K*16+4 * i + j][2] = i
                c[K*16+4 * i + j][3] = bit_mean
                c[K*16+4 * i + j][4] = f_mean

    c_sorted = sorted(c, key=lambda x: x[2], reverse=True)
    c_sorted_np = np.array(c_sorted)
    c_list = c_sorted_np.tolist()
    c_json = json.dumps(c_list)
    with open('pareto_list.txt', 'w') as file:
        file.write(c_json)


if __name__ == "__main__":
    pareto_main()