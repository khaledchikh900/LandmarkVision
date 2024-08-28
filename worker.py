import threading
import json
import os
import cv2
import numpy as np
import mediapipe as mp
from playsound import playsound
import queue
import time
from threading import Event
from PIL import Image, ImageTk  # Use PIL for image processing
import pickle

class Worker(threading.Thread):
    def __init__(self, config, update_status_callback, update_frame_callback):
        super().__init__()
        self.config = config
        self.update_status = update_status_callback
        self.update_frame = update_frame_callback
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.cap = None
        self.holistic = None
        self.collecting = False
        self.paused = False
        self.data_buffer = []
        self.images_buffer = []
        self.current_action = None
        self.stop_event = Event()
        self.pause_event = Event()
        self.queue = queue.Queue()
        self.DATA_PATH, self.actions, self.start_folder, self.no_sequences, self.sequence_length = self.folder_creation()

    def folder_creation(self):
        DATA_PATH = self.config['data_path']
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

        actions = np.array(self.config['actions'])
        no_sequences = self.config['no_sequences']
        sequence_length = self.config['sequence_length']
        start_folder = self.config['start_folder']

        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            if not os.path.exists(action_path):
                os.makedirs(action_path)
            dirmax = 0
            if os.listdir(action_path):
                dirmax = np.max(np.array(os.listdir(action_path)).astype(int))
            for sequence in range(start_folder, start_folder + no_sequences):
                try:
                    os.makedirs(os.path.join(action_path, str(sequence)))
                except:
                    pass

        return DATA_PATH, actions, start_folder, no_sequences, sequence_length

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_styled_landmarks(self, image, results):
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

    def run(self):
        while not self.stop_event.is_set():
            try:
                task = self.queue.get(timeout=0.1)
                if task == 'collect':
                    self.collect_data()
                elif task == 'save':
                    self.save_data()
            except queue.Empty:
                continue

    def collect_data(self):
        if not self.cap or not self.holistic:
            self.queue.put(('status', "Error: Collection has not started properly."))
            return

        for i in range(3, 0, -1):
            if self.stop_event.is_set():
                return
            self.queue.put(('status', f"Starting in {i}..."))
            time.sleep(1)
            playsound('./note.wav')

        for sequence in range(self.start_folder, self.start_folder + self.no_sequences):
            for frame_num in range(self.sequence_length):
                if self.stop_event.is_set():
                    return

                while self.paused and not self.stop_event.is_set():
                    self.pause_event.wait(timeout=0.1)

                ret, frame = self.cap.read()
                if not ret:
                    continue

                image, results = self.mediapipe_detection(frame, self.holistic)
                self.draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {self.current_action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    self.queue.put(('frame', image))
                    time.sleep(0.5)
                else:
                    cv2.putText(image, f'Collecting frames for {self.current_action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    self.queue.put(('frame', image))

                keypoints = self.extract_keypoints(results)
                self.data_buffer.append(keypoints)
                self.images_buffer.append(frame)

                time.sleep(0.01)  # Small delay to allow for interruption

        if self.cap:
            self.cap.release()
        self.queue.put(('frame', None))
        self.queue.put(('status', "Collection Completed"))


    def start_collection(self, action):
        self.current_action = action
        self.cap = cv2.VideoCapture(self.config['camera_index'])
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence'],
            refine_face_landmarks=True
        )
        self.collecting = True
        self.paused = False
        self.data_buffer = []
        self.images_buffer = []
        self.stop_event.clear()
        self.pause_event.clear()
        self.queue.put('collect')

    def pause_collection(self):
        if self.collecting and not self.paused:
            self.paused = True
            self.update_status("Collection paused.")

    def resume_collection(self):
        if self.collecting and self.paused:
            self.paused = False
            self.pause_event.set()
            self.update_status("Collection resumed.")

    def save_data(self):
        if not self.current_action:
            self.update_status("No action selected.")
            return

        file_format = self.config.get('file_format', 'npy')
        save_images = self.config.get('save_images', False)  # Default to False for GDPR compliance

        for idx, keypoints in enumerate(self.data_buffer):
            folder_idx = self.start_folder + idx // self.sequence_length
            if folder_idx >= self.start_folder + self.no_sequences:
                break

            if save_images:
                image_path = os.path.join(self.DATA_PATH, self.current_action, str(folder_idx), f'{idx % self.sequence_length}.jpg')
                cv2.imwrite(image_path, self.images_buffer[idx])
            
            file_path = os.path.join(self.DATA_PATH, self.current_action, str(folder_idx), str(idx % self.sequence_length))
            if file_format == 'pickle':
                with open(file_path + '.pkl', 'wb') as f:
                    pickle.dump(keypoints, f)
            else:  # default to npy
                np.save(file_path + '.npy', keypoints)

        self.data_buffer = []
        self.images_buffer = []
        self.update_status("Data Saved")



    def stop_collection(self):
        self.stop_event.set()
        if self.collecting:
            self.save_data()  # Save data when stopping the collection
        self.collecting = False
        self.update_status("Collection Stopped")

    def reset(self):
        self.stop_event.set()
        self.pause_event.set()
        if self.cap:
            self.cap.release()
        self.collecting = False
        self.paused = False
        self.data_buffer = []
        self.images_buffer = []
        self.current_action = None
        self.queue.put(('status', "Reset Completed"))
