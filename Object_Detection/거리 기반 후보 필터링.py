import cv2
import os
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import time

# 이미지에 텍스트를 출력하는 함수 (한글 지원)
def put_text_on_image(img, text, position, font_size=30, font_color=(0, 255, 0)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=font_color[::-1])
    return np.array(img_pil)

# 두 바운딩 박스 사이의 IOU 계산 함수
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0 # 겹치지 않음
    inter_area = (x2 - x1) * (y2 - y1) # 교집합 넓이 
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]) # 첫 박스 넓이
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]) # 두 번째 박스 넓이
    union = area1 + area2 - inter_area # 합집합 넓이
    return inter_area / union # IOU 계산

# 중심 거리 계산 함수
def center_distance(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

# === 설정 ===
video_path = r"C:\workspace\list\0_8_IPC1_20230108162038.mp4"
model_path = r"C:\workspace\list\best.pt"
grid_size = 3
selected_cell = (2, 0)
scale_factor = 1.0
conf_threshold = 0.25
DIST_THRESHOLD = 50

# === 초기화 ===
if not os.path.exists(video_path):
    print(f"\u274c 파일이 존재하지 않습니다: {video_path}")
    exit()

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

last_detections = []
frame_memory = []  # 최근 N프레임 저장
MAX_MEMORY = 10
tracked_ids_per_frame = []
start_time = time.time()

# === 메인 루프 ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    row, col = selected_cell
    cell_h = height // grid_size
    cell_w = width // grid_size
    x, y = col * cell_w, row * cell_h
    cell_frame = frame[y:y+cell_h, x:x+cell_w]

    results = model.track(cell_frame, conf=conf_threshold, persist=True, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().tolist()
    ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else [-1] * len(boxes)

    new_detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        obj_id = ids[i]

        if obj_id == -1:
            for memory in reversed(frame_memory):
                for last_box, last_id in memory:
                    if center_distance(box, last_box) < DIST_THRESHOLD:
                        iou = calculate_iou(box, last_box)
                        if iou > 0.5:
                            obj_id = last_id
                            break
                if obj_id != -1:
                    break

        new_detections.append(((x1, y1, x2, y2), obj_id))

    plotted = cell_frame.copy()
    for (x1, y1, x2, y2), obj_id in new_detections:
        cv2.rectangle(plotted, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(plotted, f"ID {obj_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    if scale_factor != 1.0:
        h, w = plotted.shape[:2]
        new_size = (int(w * scale_factor), int(h * scale_factor))
        plotted = cv2.resize(plotted, new_size)

    chicken_count = len(new_detections)
    plotted = put_text_on_image(plotted, f"(2,0) 셀 - 닭 {chicken_count}마리", (10, 30), 25)

    cv2.imshow("Selected Cell Detection with ID Recovery", plotted)

    last_detections = [((x1, y1, x2, y2), obj_id) for (x1, y1, x2, y2), obj_id in new_detections]
    frame_memory.append(last_detections)
    if len(frame_memory) > MAX_MEMORY:
        frame_memory.pop(0)

    frame_ids = [obj_id for (_, obj_id) in new_detections if obj_id != -1]
    tracked_ids_per_frame.append(set(frame_ids))

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time - start_time

id_switch_count = 0
for i in range(1, len(tracked_ids_per_frame)):
    prev_ids = tracked_ids_per_frame[i-1]
    curr_ids = tracked_ids_per_frame[i]
    new_ids = curr_ids - prev_ids
    id_switch_count += len(new_ids)

print("\n=== 성능 측정 결과 ===")
print(f"총 프레임 수: {len(tracked_ids_per_frame)}")
print(f"ID Switching 횟수: {id_switch_count}")
print(f"총 실행 시간: {total_time:.2f}초")
