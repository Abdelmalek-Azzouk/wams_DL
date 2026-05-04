import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading


class VideoEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VideoEfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x = x.view(B * F, C, H, W)
        out = self.backbone(x)
        out = out.view(B, F, -1)
        out = out.mean(dim=1)
        return out


class ShopliftingDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shoplifting Video Detector")
        self.root.geometry("400x350")
        self.root.resizable(False, False)

        self.video_path = None
        self.model_path = "C:/Users/AmaTek/Documents/M1/S2/AW/project/best_shoplifting_efficientnet.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = 10
        self.img_size = 224

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.build_ui()
        self.model = None
        self.load_model()

    def build_ui(self):
        tk.Label(self.root, text="Shoplifting Detector", font=("Arial", 16, "bold"), pady=20).pack()

        self.btn_select = tk.Button(
            self.root,
            text="1. Select MP4 Video",
            command=self.select_video,
            width=20,
            font=("Arial", 12)
        )
        self.btn_select.pack(pady=10)

        self.lbl_file = tk.Label(
            self.root,
            text="No video selected",
            fg="gray",
            wraplength=350
        )
        self.lbl_file.pack(pady=5)

        self.btn_analyze = tk.Button(
            self.root,
            text="2. Analyze Video",
            command=self.start_analysis,
            width=20,
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            state=tk.DISABLED
        )
        self.btn_analyze.pack(pady=20)

        self.lbl_result = tk.Label(
            self.root,
            text="",
            font=("Arial", 14, "bold")
        )
        self.lbl_result.pack(pady=10)

    def load_model(self):
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Error",
                f"Model file '{self.model_path}' not found in the current folder!"
            )
            return

        self.lbl_result.config(text="Loading model...", fg="blue")
        self.root.update()

        try:
            self.model = VideoEfficientNet(num_classes=2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.lbl_result.config(text="Model ready.", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.lbl_result.config(text="")

    def select_video(self):
        filepath = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )

        if filepath:
            self.video_path = filepath
            self.lbl_file.config(text=f"...{filepath[-40:]}")
            self.btn_analyze.config(state=tk.NORMAL)
            self.lbl_result.config(text="")

    def start_analysis(self):
        self.btn_select.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)
        self.lbl_result.config(text="Analyzing... Please wait.", fg="blue")

        threading.Thread(target=self.analyze_video, daemon=True).start()

    def analyze_video(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if frame_count > 0:
                frame_indices = np.linspace(
                    0,
                    frame_count - 1,
                    self.num_frames,
                    dtype=int
                )
            else:
                frame_indices = np.zeros(self.num_frames, dtype=int)

            frames = []
            current_frame = 0
            grabbed_count = 0

            while cap.isOpened() and grabbed_count < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame == frame_indices[grabbed_count]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.transform(frame)
                    frames.append(frame_tensor)
                    grabbed_count += 1

                current_frame += 1

            cap.release()

            while len(frames) < self.num_frames:
                frames.append(torch.zeros((3, self.img_size, self.img_size)))

            video_tensor = torch.stack(frames).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(video_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, 0)

            is_shoplifting = predicted_class.item() == 1
            conf_percent = confidence.item() * 100

            result_text = (
                f"Result: {'SHOPLIFTING' if is_shoplifting else 'NORMAL'}\n"
                f"Confidence: {conf_percent:.2f}%"
            )

            color = "red" if is_shoplifting else "green"

            self.root.after(0, self.update_result, result_text, color)

        except Exception as e:
            self.root.after(0, self.update_result, "Error during analysis", "red")
            print(e)

    def update_result(self, text, color):
        self.lbl_result.config(text=text, fg=color)
        self.btn_select.config(state=tk.NORMAL)
        self.btn_analyze.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = ShopliftingDetectorApp(root)
    root.mainloop()
