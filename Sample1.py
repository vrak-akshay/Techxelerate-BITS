import cv2
import torch
import numpy as np
import random

model = torch.hub.load('ultralytics/yolov5', 'yolov5m')

COCO_LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'traffic light',
]

NUM_STATES = 3
NUM_ACTIONS = 4
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 1.0
MIN_EXPLORATION_RATE = 0.01
EXPLORATION_DECAY = 0.995

Q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

def calculate_traffic_density(detections):
    vehicles = [obj for obj in detections if obj[-1] == 2]
    return len(vehicles)

def get_state(traffic_density):
    if traffic_density < 5:
        return 0
    elif 5 <= traffic_density <= 10:
        return 1
    else:
        return 2

def choose_action(state):
    if random.uniform(0, 1) < EXPLORATION_RATE:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        return np.argmax(Q_table[state])

def update_q_table(state, action, reward, next_state):
    Q_table[state, action] = Q_table[state, action] + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * np.max(Q_table[next_state]) - Q_table[state, action]
    )

def get_reward(traffic_density, action):
    if traffic_density > 10 and action == 2:
        return 1.0
    elif 5 <= traffic_density <= 10 and action == 1:
        return 0.5
    else:
        return -0.1

cap = cv2.VideoCapture('vehicles2.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = results.xyxy[0].numpy()

    traffic_density = calculate_traffic_density(detections)

    state = get_state(traffic_density)

    action = choose_action(state)

    if action == 0:
        signal_state = "Green (Lane 1)"
    elif action == 1:
        signal_state = "Green (Lane 2)"
    elif action == 2:
        signal_state = "Green (Lane 3)"
    else:
        signal_state = "Green (Lane 4)"

    reward = get_reward(traffic_density, action)

    next_state = state

    update_q_table(state, action, reward, next_state)

    EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * EXPLORATION_DECAY)

    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        if class_id == 2:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{COCO_LABELS[int(class_id)]} {confidence:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Vehicles: {traffic_density}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Signal: {signal_state}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Traffic Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Final Q-table:")
print(Q_table)