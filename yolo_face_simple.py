import cv2
import torch
from pathlib import Path
import sys

# Tambahkan root directory ke sys.path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.torch_utils import select_device

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale landmark coordinates
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]
    coords[:, :10] /= gain
    coords[:, :10] = coords[:, :10].clamp(min=0)
    return coords

def detect_and_draw(model, img, device):
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5

    h0, w0 = img.shape[:2]
    r = img_size / max(h0, w0)
    img0 = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)

    imgsz = check_img_size(img_size, s=model.stride.max())
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device).float() / 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], img0.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:15].view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()

                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)

                for j in range(5):
                    point_x = int(landmarks[2 * j])
                    point_y = int(landmarks[2 * j + 1])
                    cv2.circle(img0, (point_x, point_y), 3, (0, 0, 255), -1)

                cv2.putText(img0, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return img0

if __name__ == '__main__':
    weights = 'yolov5m-face.pt'  # Ganti dengan path ke model weights Anda
    image_path = './George_W_Bush/George_W_Bush_0004.jpg'  # Ganti dengan path ke gambar Anda

    device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(weights, device)

    img = cv2.imread(image_path)
    result = detect_and_draw(model, img, device)

    cv2.imshow('Face Detection', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()