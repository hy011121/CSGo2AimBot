import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import win32gui
import win32api
import win32con
import time
from PIL import ImageGrab
import math
import threading
import concurrent.futures

def capture_window_screen(window_title, size_scale=1):
    hwnd = win32gui.FindWindow(None, window_title)
    rect = win32gui.GetWindowRect(hwnd)
    region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]
    ori_img = np.array(ImageGrab.grab(bbox=region))
    ori_img = cv2.resize(ori_img, (ori_img.shape[1] // size_scale, ori_img.shape[0] // size_scale))
    return ori_img

def move_mouse_to(dx, dy):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

model = YOLO(r"Her you must input Your pt file")

screen_width = 1920
screen_height = 1080

window_title = 'Counter-Strike 2'
size_scale = 1  # Größenanpassung des Bildschirmshots (optional)

warmup_image = np.zeros((screen_height // size_scale, screen_width // size_scale, 3), dtype=np.uint8)
model(warmup_image)

def process_frame(screen_image):
    # Convert the image to the format required for object detection
    image = Image.fromarray(cv2.cvtColor(screen_image, cv2.COLOR_BGR2RGB))
    
    results = model(image)
    
    detected_boxes = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        classes = result.boxes.cls.cpu().numpy()  # Extract class labels

        for box, cls in zip(boxes, classes):
            # Calculate the bounding box coordinates
            left, top, right, bottom = map(int, box)
            detected_boxes.append((left, right, top, bottom))

    return detected_boxes

def find_closest_target(detected_boxes):
    if not detected_boxes:
        return None, None

    min_dist = float('inf')
    closest_box_index = 0
    centers = []
    for i, box in enumerate(detected_boxes):
        x1, x2, y1, y2 = box
        c_x = (x2 + x1) / 2
        c_y = (y2 + y1) / 2
        centers.append((c_x, c_y))
        dist = math.hypot(screen_width / 2 - c_x, screen_height / 2 - c_y)
        if dist < min_dist:
            min_dist = dist
            closest_box_index = i

    target_center_x = centers[closest_box_index][0]
    target_center_y = centers[closest_box_index][1]
    target_height = detected_boxes[closest_box_index][3] - detected_boxes[closest_box_index][2]

    y_adjustment = 0.3 * target_height  # Adjusted value
    x = target_center_x - screen_width / 2
    y = target_center_y - screen_height / 2 - y_adjustment

    scale_factor = 1  # You can adjust this if needed to fine-tune the movements
    dx = int(x * scale_factor)
    dy = int(y * scale_factor)

    return dx, dy

def main_loop():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            screen_image = capture_window_screen(window_title, size_scale)
            future = executor.submit(process_frame, screen_image)
            detected_boxes = future.result()
            dx, dy = find_closest_target(detected_boxes)

            if dx is not None and dy is not None:
                print(f"Moving mouse by: dx={dx}, dy={dy}")
                move_mouse_to(dx, dy)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

            time.sleep(0.1)

if __name__ == "__main__":
    main_loop()
