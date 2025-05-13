%%writefile /kaggle/working/Retinaface_custome/widerface_evaluate/evaluation.py 

import os
import tqdm
import pickle
import argparse
import numpy as np
from bbox import bbox_overlaps

def get_gt_boxes_from_txt(gt_path, cache_dir):
    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            boxes = pickle.load(f)
        return boxes

    with open(gt_path, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    state = 0
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '.jpg' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue
        if state == 2 and '.jpg' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float64')
            current_name = line
            current_boxes = []
            continue
        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    if current_name and current_boxes:
        boxes[current_name] = np.array(current_boxes).astype('float64')

    with open(cache_file, 'wb') as f:
        pickle.dump(boxes, f)
    return boxes

def read_pred_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes, dtype=np.float64)
    img_name = os.path.basename(filepath).rstrip('.txt')
    return img_name, boxes

def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    for event in events:
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname] = _boxes
        boxes[event] = current_event
    return boxes

def norm_score(pred):
    max_score = 0
    min_score = 1
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)
    diff = max_score - min_score
    if diff == 0:
        return
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff

def image_eval(pred, gt, ignore, iou_thresh):
    _pred = pred.copy().astype(np.float64)
    _gt = gt.copy().astype(np.float64)
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh and recall_list[max_idx] == 0:
            recall_list[max_idx] = 1
            proposal_list[h] = 1
        else:
            proposal_list[h] = 0
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list

def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):
        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info

def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0] if pr_curve[i, 0] != 0 else 0
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face if count_face != 0 else 0
    return _pr_curve

def voc_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluation(pred, gt_path, iou_thresh=0.3):
    pred = get_preds(pred)
    norm_score(pred)
    gt_boxes = get_gt_boxes_from_txt(gt_path, os.path.dirname(gt_path))
    count_face = 0
    thresh_num = 1000
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    
    for img_path in gt_boxes.keys():
        gt_img_boxes = gt_boxes.get(img_path, np.array([]))
        event_name = img_path.split('/')[0]
        img_name = img_path.split('/')[-1].rstrip('.jpg')
        pred_event = pred.get(event_name, {})
        pred_info = pred_event.get(img_name, np.array([]))
        
        count_face += len(gt_img_boxes)
        
        if len(gt_img_boxes) == 0 or len(pred_info) == 0:
            continue
        ignore = np.zeros(gt_img_boxes.shape[0])
        pred_recall, proposal_list = image_eval(pred_info, gt_img_boxes, ignore, iou_thresh)
        _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
        pr_curve += _img_pr_info
    
    if count_face == 0:
        return 0.0
    
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    ap = voc_ap(recall, propose)
    
    print(ap)

if __name__ == '__main__':
    import sys
    sys.argv = [arg for arg in sys.argv if not arg.endswith('.json')]
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./widerface_txt/")
    parser.add_argument('-g', '--gt', default='./ground_truth/')
    args = parser.parse_args()
    evaluation(args.pred, args.gt)