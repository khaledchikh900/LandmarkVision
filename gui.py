import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk  # Use PIL for image processing
import json
from worker import Worker
import cv2

class FacialLandmarkCollectorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Landmark Collector")

        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        # Status Label
        self.status_label = ttk.Label(self.root, text="Status: Idle")
        self.status_label.grid(row=0, column=0, columnspan=4, pady=5)

        # Action Label
        self.action_label = ttk.Label(self.root, text="Select Action:")
        self.action_label.grid(row=1, column=0, pady=5)

        # Action Dropdown
        self.action_dropdown = ttk.Combobox(self.root, state="readonly")
        self.action_dropdown.grid(row=1, column=1, pady=5)
        self.action_dropdown.bind("<<ComboboxSelected>>", self.update_action)

        # Buttons
        self.start_button = ttk.Button(self.root, text="Start Collection", command=self.start_collection)
        self.save_button = ttk.Button(self.root, text="Save Data", command=self.save_data)
        self.stop_button = ttk.Button(self.root, text="Stop Collection", command=self.stop_collection)
        self.reset_button = ttk.Button(self.root, text="Reset", command=self.reset)
        self.pause_button = ttk.Button(self.root, text="Pause Collection", command=self.pause_collection)
        self.resume_button = ttk.Button(self.root, text="Resume Collection", command=self.resume_collection)
        self.exit_button = ttk.Button(self.root, text="Exit", command=self.exit_app)

        # Button Grid
        self.start_button.grid(row=2, column=0, padx=5, pady=5)
        self.save_button.grid(row=2, column=1, padx=5, pady=5)
        self.stop_button.grid(row=2, column=2, padx=5, pady=5)
        self.reset_button.grid(row=2, column=3, padx=5, pady=5)
        self.pause_button.grid(row=3, column=0, padx=5, pady=5)
        self.resume_button.grid(row=3, column=1, padx=5, pady=5)
        self.exit_button.grid(row=3, column=2, padx=5, pady=5)

        # Video Label
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=4, column=0, columnspan=4, pady=5)

        # Load actions from the configuration file
        self.load_actions()

    def load_actions(self):
        # Load actions from the configuration file
        with open('config.json', 'r') as f:
            config = json.load(f)
            actions = config.get('actions', [])
        
        if actions:
            self.action_dropdown['values'] = actions
            if actions:
                self.action_dropdown.set(actions[0])
        else:
            self.action_dropdown.set("No Actions Available")

    def update_action(self, event):
        # Handle any additional logic needed when the action changes
        pass

    def start_collection(self):
        selected_action = self.action_dropdown.get()
        if not selected_action:
            messagebox.showwarning("Warning", "Please select an action from the dropdown.")
            return
        
        if not self.worker or not self.worker.is_alive():
            with open('config.json', 'r') as f:
                config = json.load(f)
            self.worker = Worker(config, self.update_status, self.update_frame)
            self.worker.start()

        self.worker.start_collection(selected_action)
        self.update_ui_state(starting=True)

    def save_data(self):
        if self.worker:
            self.worker.save_data()

    def stop_collection(self):
        if self.worker:
            self.worker.stop_collection()
        self.update_ui_state(starting=False)

    def pause_collection(self):
        if self.worker:
            self.worker.pause_collection()
            self.update_ui_state(pause_resume=True)

    def resume_collection(self):
        if self.worker:
            self.worker.resume_collection()
            self.update_ui_state(pause_resume=True)

    def reset(self):
        if self.worker:
            self.worker.reset()
            self.worker = None
        self.update_ui_state(starting=False)

    def exit_app(self):
        if self.worker:
            self.worker.reset()
        self.root.destroy()

    def update_ui_state(self, starting=False, pause_resume=False):
        if starting:
            self.start_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.resume_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        elif pause_resume:
            self.pause_button.config(state=tk.NORMAL if not self.worker.paused else tk.DISABLED)
            self.resume_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            self.resume_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")

    def update_frame(self, frame):
        if frame is None:
            self.video_label.config(image='')
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

