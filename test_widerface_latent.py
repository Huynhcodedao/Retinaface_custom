from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.init import xavier_uniform_

parser = argparse.ArgumentParser(description='Retinaface with JPEG AI Latents')
parser.add_argument('-m', '--trained_model', default='/kaggle/working/Retinaface_custom/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--latent_folder', default='/kaggle/input/jpegai/latents/', type=str, help='Folder containing latent_*.npy files')
parser.add_argument('--save_folder', default='/kaggle/working/Retinaface_custom/widerface_evaluate/widerfacejpegai_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_list', default='/kaggle/working/Retinaface_custom/data/widerfacejpegai/val/wider_val.txt', type=str, help='Path to wider_val.txt')
parser.add_argument('--confidence_threshold', default=0.01, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

class BridgeModule(nn.Module):
    def __init__(self):
        super(BridgeModule, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(160, 160), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        return x

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformableConv2d, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        offsets = self.offset_conv(x)
        x = torchvision.ops.deform_conv2d(x, offsets, self.conv.weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)
        return x

class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super(ContextModule, self).__init__()
        self.dcn1 = DeformableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.dcn2 = DeformableConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.dcn1(x))
        x = F.relu(self.dcn2(x))
        x = self.conv(x)
        return x

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        self.p6_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_uniform_(m.weight)

    def forward(self, inputs):
        laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]
        for i in range(len(laterals)-2, -1, -1):
            laterals[i] += F.interpolate(laterals[i+1], size=laterals[i].shape[2:], mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        p6 = self.p6_conv(outs[-1])
        return outs + [p6]

class DetectionHead(nn.Module):
    def __init__(self, num_anchors=3, num_classes=2):
        super(DetectionHead, self).__init__()
        self.cls_head = nn.Conv2d(256, num_anchors * num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(256, num_anchors * 4, kernel_size=1)
        self.landm_head = nn.Conv2d(256, num_anchors * 10, kernel_size=1)

    def forward(self, x):
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        landm = self.landm_head(x)
        return cls, reg, landm

class PrunedRetinaFace(nn.Module):
    def __init__(self, cfg, phase='test'):
        super(PrunedRetinaFace, self).__init__()
        self.phase = phase
        self.bridge = BridgeModule()
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(self.backbone.children())[5:8])  # Stage 2-4
        self.fpn = FPN(in_channels_list=[512, 1024, 2048], out_channels=256)
        self.context_modules = nn.ModuleList([ContextModule(256) for _ in range(5)])
        self.detection_head = DetectionHead(num_anchors=3, num_classes=2)
        self.cfg = cfg

    def forward(self, x):
        x = self.bridge(x)
        features = []
        for layer in self.backbone:
            x = layer(x)
            features.append(x)
        fpn_features = self.fpn(features)
        fpn_features = [self.context_modules[i](f) for i, f in enumerate(fpn_features)]
        cls_outputs, reg_outputs, landm_outputs = [], [], []
        for f in fpn_features:
            cls, reg, landm = self.detection_head(f)
            cls_outputs.append(cls.permute(0, 2, 3, 1).contiguous())
            reg_outputs.append(reg.permute(0, 2, 3, 1).contiguous())
            landm_outputs.append(landm.permute(0, 2, 3, 1).contiguous())
        cls_outputs = torch.cat([o.view(o.size(0), -1, 2) for o in cls_outputs], 1)
        reg_outputs = torch.cat([o.view(o.size(0), -1, 4) for o in reg_outputs], 1)
        landm_outputs = torch.cat([o.view(o.size(0), -1, 10) for o in landm_outputs], 1)
        return reg_outputs, cls_outputs, landm_outputs

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage, weights_only=True)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device), weights_only=True)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    net = PrunedRetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    with open(args.dataset_list, 'r') as fr:
        test_dataset = fr.read().split()
    num_images = len(test_dataset)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    for i, img_name in enumerate(test_dataset):
        latent_path = os.path.join(args.latent_folder, f'latent_{img_name.split("/")[-1][:-4]}.npy')
        if not os.path.exists(latent_path):
            print(f"Warning: Latent file {latent_path} not found, skipping...")
            continue
        latent = np.load(latent_path)
        latent = torch.from_numpy(latent).float().to(device)

        _t['forward_pass'].tic()
        loc, conf, landms = net(latent)
        _t['forward_pass'].toc()
        _t['misc'].tic()

        priorbox = PriorBox(cfg, image_size=(640, 640))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * torch.Tensor([640, 640, 640, 640]).to(device)
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([640, 640, 640, 640, 640, 640, 640, 640, 640, 640]).to(device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))

        if args.save_image:
            img_raw = np.zeros((640, 640, 3), dtype=np.uint8)  # Placeholder for visualization
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            if not os.path.exists("/kaggle/working/Retinaface_custom/results/"):
                os.makedirs("/kaggle/working/Retinaface_custom/results/")
            name = "/kaggle/working/Retinaface_custom/results/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)