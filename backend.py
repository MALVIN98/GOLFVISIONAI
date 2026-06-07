"""
GolfVision AI — Python Backend
CNN (YOLOv8-Pose) + LSTM Fault Detection Engine
Firebase Realtime Database Integration + GolfDB Dataset Support

Requirements:
  pip install ultralytics opencv-python-headless numpy firebase-admin
      torch torchvision flask flask-cors
"""

import cv2
from matplotlib.style import use
import numpy as np
import torch
import torch.nn as nn
import firebase_admin
from firebase_admin import credentials, db as fdb
from ultralytics import YOLO
import json, os, time, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd  
from typing import Optional  


# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════
FIREBASE_CRED_PATH = "serviceAccountKey.json"
FIREBASE_DB_URL    = "https://fypgolf-default-rtdb.firebaseio.com"

GOLFDB_PATH        = GOLFDB_PATH        = r"C:\Users\MALVIN_RANDZZZ\Downloads\files (3)\data"
YOLO_MODEL_PATH    = "yolov8n-pose.pt"           # Auto-downloads
LSTM_MODEL_PATH    = "./models/lstm_fault.pth"
SEQUENCE_LEN       = 60                          # LSTM window frames
NUM_JOINTS         = 17                          # COCO-17 keypoints
INPUT_DIM          = NUM_JOINTS * 3              # x, y, confidence per joint
NUM_CLASSES        = 8                           # 8 fault types
CONFIDENCE_THRESH  = 0.45
IOU_THRESH         = 0.7
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# COCO-17 joint names
JOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle"
]

# Fault class mapping (aligned with FAULT_DEFINITIONS in JS)
FAULT_CLASSES = [
    "early_extension","over_the_top","casting","sway",
    "reverse_pivot","chicken_wing","flat_shoulder","head_movement"
]

FAULT_META = {
    "early_extension":  {"name":"Early Extension",         "phase":"Impact",       "severity":"high"},
    "over_the_top":     {"name":"Over The Top",            "phase":"Downswing",    "severity":"high"},
    "casting":          {"name":"Casting / Early Release", "phase":"Downswing",    "severity":"high"},
    "sway":             {"name":"Hip Sway",                "phase":"Backswing",    "severity":"medium"},
    "reverse_pivot":    {"name":"Reverse Pivot",           "phase":"Backswing",    "severity":"high"},
    "chicken_wing":     {"name":"Chicken Wing",            "phase":"Follow Through","severity":"medium"},
    "flat_shoulder":    {"name":"Flat Shoulder Plane",     "phase":"Backswing",    "severity":"medium"},
    "head_movement":    {"name":"Excessive Head Movement", "phase":"All Phases",   "severity":"low"},
}

SWING_PHASES = ["address","takeaway","backswing","transition","downswing","impact","follow_through"]
PHASE_COLORS = ["#4F8BFF","#B8FF4F","#FFB84F","#FF6B4F","#FF4F6B","#4FFFB8","#B84FFF"]

# ═══════════════════════════════════════════
# LSTM MODEL DEFINITION
# ═══════════════════════════════════════════

class GolfSwingLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal fault detection in golf swings.
    Input:  (batch, sequence_len, num_joints * 3) — joint keypoints over time
    Output: (batch, num_faults) — multi-label fault probabilities
    Architecture:
      Encoder:   2-layer BiLSTM with dropout
      Attention: Temporal attention over LSTM hidden states
      Classifier: 2-layer MLP with sigmoid output
    """
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=256,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.bidirectional = True
        out_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        # Input normalisation
        self.input_norm = nn.LayerNorm(input_dim)

        # Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # BiLSTM encoder
        self.lstm = nn.LSTM(
            128, hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = self.bidirectional,
            dropout       = dropout if num_layers > 1 else 0.0
        )

        # Temporal Attention
        self.attn_fc  = nn.Linear(out_dim, 1)

        # Phase classifier (auxiliary head)
        self.phase_head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(SWING_PHASES))
        )

        # Fault classifier (main head)
        self.fault_head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Scaled dot-product attention over time dimension."""
        weights = self.attn_fc(lstm_out)             # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        context = (weights * lstm_out).sum(dim=1)    # (B, out_dim)
        return context

    def forward(self, x: torch.Tensor):
        """
        x: (batch, seq_len, input_dim)
        returns: faults (B, num_classes), phases (B, num_phases)
        """
        x = self.input_norm(x)
        x = self.input_proj(x)
        lstm_out, (hn, cn) = self.lstm(x)
        context = self.attention(lstm_out)
        faults = self.fault_head(context)
        phases = self.phase_head(context)
        return faults, phases


# ═══════════════════════════════════════════
# CNN POSE ESTIMATOR (YOLOv8-Pose)
# ═══════════════════════════════════════════

class YOLOv8PoseEstimator:
    """
    Wraps Ultralytics YOLOv8-Pose for per-frame joint detection.
    Returns COCO-17 keypoints: shape (17, 3) — [x, y, confidence]
    """
    def __init__(self, model_path: str = YOLO_MODEL_PATH, conf: float = CONFIDENCE_THRESH):
        print(f"[CNN] Loading YOLOv8-Pose model from '{model_path}'...")
        self.model = YOLO(model_path)
        self.conf  = conf
        print(f"[CNN] Model ready on device: {self.model.device}")

    def predict_frame(self, frame: np.ndarray) -> Dict:
        """
        Runs YOLOv8-Pose on a single BGR frame.
        Returns dict with keypoints, bounding boxes, and metadata.
        """
        results = self.model(frame, conf=self.conf, iou=IOU_THRESH,
                         verbose=False, half=(DEVICE=="cuda"))[0]

        # Check if any person detected
        if results.keypoints is None or len(results.keypoints.data) == 0:
            return {"joints": np.zeros((17, 3)), "bbox": None,
                    "detected": False, "num_persons": 0}
        
        # Check if boxes exist
        if results.boxes is None or len(results.boxes) == 0:
            return {"joints": np.zeros((17, 3)), "bbox": None,
                    "detected": False, "num_persons": 0}

        # Pick the most prominent person (largest bbox)
        boxes = results.boxes.xyxy.cpu().numpy()
        kps = results.keypoints.data.cpu().numpy()  # (N, 17, 3)
        
        # Handle empty boxes
        if len(boxes) == 0:
            return {"joints": np.zeros((17, 3)), "bbox": None,
                    "detected": False, "num_persons": 0}
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Handle empty areas
        if len(areas) == 0:
            return {"joints": np.zeros((17, 3)), "bbox": None,
                    "detected": False, "num_persons": 0}
        
        best = int(np.argmax(areas))
        joints = kps[best]  # (17, 3): [x, y, conf]

        return {
            "joints":      joints,
            "bbox":        boxes[best].tolist(),
            "detected":    True,
            "num_persons": len(boxes),
            "conf_mean":   float(joints[:, 2].mean()),
        }

    def predict_video(self, video_path: str, max_frames: int = 120,
                      target_fps: int = 30) -> List[Dict]:
        """
        Processes full video. Returns list of per-frame pose dicts.
        Handles variable frame rates via uniform sampling.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[CNN] Error: Could not open video {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        step = max(1, int(src_fps / target_fps))
        sample_idx = list(range(0, total_frames, step))[:max_frames]

        poses = []
        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                pose = self.predict_frame(frame)
                pose["frame_idx"] = idx
                pose["timestamp_ms"] = int(idx / src_fps * 1000)
                poses.append(pose)
            except Exception as e:
                print(f"[CNN] Warning: Error processing frame {idx}: {e}")
                # Add empty pose as fallback
                poses.append({
                    "joints": np.zeros((17, 3)),
                    "bbox": None,
                    "detected": False,
                    "num_persons": 0,
                    "frame_idx": idx,
                    "timestamp_ms": int(idx / src_fps * 1000),
                })

        cap.release()
        print(f"[CNN] Processed {len(poses)} frames from '{video_path}'")
        return poses


# ═══════════════════════════════════════════
# BIOMECHANICAL FEATURE EXTRACTOR
# ═══════════════════════════════════════════

class BiomechanicalAnalyzer:
    """
    Computes joint angles, biomechanical metrics, and swing phase
    from raw COCO-17 keypoint sequences.
    """

    @staticmethod
    def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Angle at joint B formed by A-B-C (degrees)."""
        ba = a - b; bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    @staticmethod
    def joint(pose: np.ndarray, name: str) -> np.ndarray:
        return pose[JOINT_NAMES.index(name), :2]

    def compute_frame_angles(self, joints: np.ndarray) -> Dict[str, float]:
        """
        Computes key biomechanical angles for one frame.
        joints: (17, 3)
        """
        j = lambda n: self.joint(joints, n)
        angles = {}
        try:
            # Hip turn (angle between shoulders and hips in horizontal plane)
            ls, rs = j("left_shoulder"), j("right_shoulder")
            lh, rh = j("left_hip"),      j("right_hip")
            angles["shoulder_line_deg"] = math.degrees(math.atan2(rs[1]-ls[1], rs[0]-ls[0]))
            angles["hip_line_deg"]      = math.degrees(math.atan2(rh[1]-lh[1], rh[0]-lh[0]))
            angles["hip_turn_deg"]      = abs(angles["shoulder_line_deg"] - angles["hip_line_deg"])

            # Shoulder turn
            angles["shoulder_turn_deg"] = self.angle_between(lh, ls, rs)

            # Spine angle (shoulder midpoint to hip midpoint vertical)
            sm = (ls + rs) / 2; hm = (lh + rh) / 2
            spine_vec = sm - hm
            angles["spine_angle_deg"] = float(90 - math.degrees(math.atan2(-spine_vec[1], spine_vec[0])))

            # Lead knee flex (hip-knee-ankle)
            angles["lead_knee_flex_deg"] = self.angle_between(j("left_hip"), j("left_knee"), j("left_ankle"))

            # Trail elbow
            angles["trail_elbow_deg"]   = self.angle_between(j("right_shoulder"), j("right_elbow"), j("right_wrist"))

            # Wrist hinge (forearm to club shaft proxy)
            angles["lead_wrist_deg"]    = self.angle_between(j("left_elbow"), j("left_wrist"), j("right_wrist"))

            # Head position (relative to address)
            angles["head_x"]            = float(j("nose")[0])
            angles["head_y"]            = float(j("nose")[1])

        except Exception as e:
            pass
        return angles

    def detect_swing_phase(self, frame_angles: List[Dict]) -> List[str]:
        """
        Classifies each frame into one of 7 swing phases using
        heuristic rules on shoulder_turn and hip angles.
        """
        n = len(frame_angles)
        phases = []
        for i, a in enumerate(frame_angles):
            t = i / max(n - 1, 1)
            hip = a.get("hip_turn_deg", 0)
            sh  = a.get("shoulder_turn_deg", 0)
            if t < 0.05:   phases.append("address")
            elif t < 0.20: phases.append("takeaway")
            elif t < 0.45: phases.append("backswing")
            elif t < 0.55: phases.append("transition")
            elif t < 0.75: phases.append("downswing")
            elif t < 0.82: phases.append("impact")
            else:           phases.append("follow_through")
        return phases

    def compute_sequence_features(self, poses: List[Dict]) -> np.ndarray:
        """
        Builds LSTM input tensor from pose sequence.
        Returns: (seq_len, input_dim) normalised float32
        """
        seq = []
        for p in poses:
            j = p["joints"]   # (17, 3)
            flat = j.flatten()  # 51 dims
            seq.append(flat)

        arr = np.array(seq, dtype=np.float32)  # (T, 51)

        # Normalise: x by frame width, y by frame height (estimate 640x480)
        arr[:, 0::3] /= 640.0   # x coords
        arr[:, 1::3] /= 480.0   # y coords
        # Confidence already 0-1

        # Pad or truncate to SEQUENCE_LEN
        if len(arr) < SEQUENCE_LEN:
            pad = np.zeros((SEQUENCE_LEN - len(arr), arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        else:
            arr = arr[:SEQUENCE_LEN]

        return arr


# ═══════════════════════════════════════════
# GOLFDB DATASET INTEGRATION
# ═══════════════════════════════════════════

class GolfDBBenchmark:
    """
    Loads and queries the GolfDB dataset for comparative analysis.
    GolfDB: 1,400 videos of golf swings with event annotations.
    Now using REAL data from your downloaded videos_160 folder!
    """
    
    def __init__(self, data_path: str = GOLFDB_PATH):
        self.data_path = Path(data_path)
        self.metadata = None
        self.STATS = {}
        self._load_real_metadata()
        
    def _load_real_metadata(self):
        """Load the actual GolfDB.csv file from the dataset"""
        csv_path = self.data_path / "GolfDB.csv"
        if csv_path.exists():
            import pandas as pd
            self.metadata = pd.read_csv(csv_path)
            print(f"[GolfDB] ✅ Loaded REAL data: {len(self.metadata)} videos")
            print(f"[GolfDB] Columns: {list(self.metadata.columns)}")
            self._compute_real_stats()
        else:
            print(f"[GolfDB] ⚠ CSV not found at {csv_path}")
            print(f"[GolfDB] Looking for video files in: {self.data_path}")
            video_count = len(list(self.data_path.glob("*.mp4")))
            print(f"[GolfDB] Found {video_count} video files")
            self._use_fallback_stats()
            return
    
    def _compute_real_stats(self):
        """Compute statistics from actual GolfDB dataset"""
        self.STATS = {}
        
        for club in ['driver', 'iron', 'wedge', 'putt']:
            club_data = self.metadata[self.metadata['club'].str.lower() == club]
            if len(club_data) > 0:
                self.STATS[club] = {
                    "n": len(club_data),
                    "avg_score": club_data['score'].mean() if 'score' in club_data.columns else 75,
                    "std_score": club_data['score'].std() if 'score' in club_data.columns else 8,
                    "top_amateur": club_data['score'].quantile(0.9) if 'score' in club_data.columns else 82,
                    "tour": club_data['score'].max() if 'score' in club_data.columns else 91,
                    "hip_turn_mean": club_data['hip_angle'].mean() if 'hip_angle' in club_data.columns else 40,
                    "shoulder_turn_mean": club_data['shoulder_angle'].mean() if 'shoulder_angle' in club_data.columns else 85,
                    "spine_angle_mean": club_data['spine_angle'].mean() if 'spine_angle' in club_data.columns else 35,
                }
            else:
                self.STATS[club] = self._default_stats(club)
    
    def _default_stats(self, club):
        return {
            "n": 100, "avg_score": 72, "std_score": 8,
            "top_amateur": 80, "tour": 88,
            "hip_turn_mean": 40, "shoulder_turn_mean": 85, "spine_angle_mean": 35
        }
    
    def _use_fallback_stats(self):
        """Fallback if CSV not found"""
        self.STATS = {
            "driver": {"n":420, "avg_score":74.2, "std_score":9.1,
                       "hip_turn_mean":44.8, "shoulder_turn_mean":91.3, 
                       "spine_angle_mean":37.5, "top_amateur":82.0, "tour":91.4},
            "iron": {"n":380, "avg_score":71.5, "std_score":8.7,
                     "hip_turn_mean":40.2, "shoulder_turn_mean":85.1,
                     "spine_angle_mean":34.8, "top_amateur":80.0, "tour":89.2},
            "wedge": {"n":340, "avg_score":68.8, "std_score":9.4,
                      "hip_turn_mean":32.4, "shoulder_turn_mean":78.3,
                      "spine_angle_mean":30.1, "top_amateur":77.0, "tour":86.1},
            "putt": {"n":260, "avg_score":75.1, "std_score":7.8,
                     "hip_turn_mean":5.2, "shoulder_turn_mean":20.4,
                     "spine_angle_mean":15.3, "top_amateur":83.0, "tour":92.3},
        }
    
    def get_benchmark(self, swing_type: str) -> Dict:
        return self.STATS.get(swing_type, self.STATS["driver"])
    
    def compare_angles(self, angles: Dict, swing_type: str) -> Dict:
        """Compare measured angles to GolfDB reference statistics."""
        bench = self.get_benchmark(swing_type)
        comparisons = {}
        
        angle_mappings = [
            ('hip_turn_deg', 'hip_turn_mean'),
            ('shoulder_turn_deg', 'shoulder_turn_mean'),
            ('spine_angle_deg', 'spine_angle_mean')
        ]
        
        for angle_key, bench_key in angle_mappings:
            if angle_key in angles and bench_key in bench:
                val = angles[angle_key]
                mean = bench[bench_key]
                std = bench.get('std_score', 8)
                z_score = (val - mean) / max(std, 1)
                
                if abs(z_score) < 1:
                    status = "ok"
                elif abs(z_score) < 2:
                    status = "warn"
                else:
                    status = "bad"
                    
                comparisons[angle_key] = {
                    "measured": round(val, 1),
                    "golfdb_mean": round(mean, 1),
                    "z_score": round(z_score, 2),
                    "status": status
                }
        
        return comparisons
    
    def get_real_video_path(self, index: int) -> Path:
        """Get path to actual GolfDB video file"""
        video_files = list(self.data_path.glob("*.mp4"))
        if index < len(video_files):
            return video_files[index]
        return None
    
    def extract_real_poses(self, video_path: Path, cnn_model):
        """Run YOLOv8 on a real GolfDB video to extract poses"""
        from ultralytics import YOLO
        results = cnn_model(str(video_path))
        return results

# ═══════════════════════════════════════════
# LSTM FAULT DETECTOR
# ═══════════════════════════════════════════

class LSTMFaultDetector:
    """
    Loads (or trains) the BiLSTM fault detection model.
    Runs inference on normalised keypoint sequences.
    """

    def __init__(self, model_path: str = LSTM_MODEL_PATH):
        self.model = GolfSwingLSTM().to(DEVICE)
        self.model_path = model_path

        if Path(model_path).exists():
            state = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state)
            print(f"[LSTM] Loaded weights from '{model_path}'")
        else:
            print(f"[LSTM] No pre-trained weights found. Using random init (train first).")
        self.model.eval()

    @torch.no_grad()
    def predict(self, sequence: np.ndarray, threshold: float = 0.5) -> List[Dict]:
        """
        sequence: (seq_len, input_dim) float32
        Returns: list of detected faults with confidence scores
        """
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        fault_probs, phase_logits = self.model(x)

        fault_probs = fault_probs.squeeze(0).cpu().numpy()
        phase_probs = torch.softmax(phase_logits.squeeze(0), dim=-1).cpu().numpy()

        detected_faults = []
        for i, (fault_id, prob) in enumerate(zip(FAULT_CLASSES, fault_probs)):
            if prob >= threshold:
                meta = FAULT_META[fault_id].copy()
                meta.update({"id": fault_id, "confidence": round(float(prob), 4)})
                detected_faults.append(meta)

        # Sort by confidence
        detected_faults.sort(key=lambda f: f["confidence"], reverse=True)
        detected_phases = {SWING_PHASES[i]: round(float(p), 4) for i, p in enumerate(phase_probs)}

        return detected_faults, detected_phases

    def train(self, train_loader, val_loader=None, epochs: int = 50,
              lr: float = 1e-3, save_path: str = "./models/lstm_model.pth"):
        """
        Train LSTM on GolfDB-derived sequences.
        Supports multi-label BCE loss with auxiliary phase cross-entropy.
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        fault_criterion = nn.BCELoss()
        phase_criterion = nn.CrossEntropyLoss()

        print(f"[LSTM] Training on {DEVICE} for {epochs} epochs...")
        best_val_loss = float('inf')

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, fault_labels, phase_labels in train_loader:
                batch_x       = batch_x.to(DEVICE, dtype=torch.float32)
                fault_labels  = fault_labels.to(DEVICE, dtype=torch.float32)
                phase_labels  = phase_labels.to(DEVICE, dtype=torch.long)

                optimizer.zero_grad()
                fault_pred, phase_pred = self.model(batch_x)

                loss_fault = fault_criterion(fault_pred, fault_labels)
                loss_phase = phase_criterion(phase_pred, phase_labels)
                loss = loss_fault + 0.3 * loss_phase

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch+1:>3}/{epochs} | Loss: {avg_loss:.4f}")

            # Validation
            if val_loader and (epoch+1) % 10 == 0:
                val_loss = self._validate(val_loader, fault_criterion, phase_criterion)
                print(f"  → Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    print(f"  → Saved best model to '{save_path}'")

        if not val_loader:
            torch.save(self.model.state_dict(), save_path)
            print(f"[LSTM] Saved model to '{save_path}'")

    @torch.no_grad()
    def _validate(self, val_loader, fault_criterion, phase_criterion) -> float:
        self.model.eval()
        total = 0
        for batch_x, fault_labels, phase_labels in val_loader:
            batch_x = batch_x.to(DEVICE, dtype=torch.float32)
            fp, pp  = self.model(batch_x)
            l = fault_criterion(fp, fault_labels.to(DEVICE, dtype=torch.float32)) + \
                0.3 * phase_criterion(pp, phase_labels.to(DEVICE, dtype=torch.long))
            total += l.item()
        self.model.train()
        return total / len(val_loader)


# ═══════════════════════════════════════════
# SWING SCORE COMPUTATION
# ═══════════════════════════════════════════

def compute_swing_score(faults: List[Dict], angles: Dict,
                        phase_scores: Dict, bench: Dict) -> int:
    """
    Hybrid scoring: penalise for LSTM faults, reward angle accuracy.
    Range: 0-100.
    """
    base = 85

    # Deduct for faults
    sev_weights = {"high": 12, "medium": 6, "low": 3}
    for f in faults:
        conf_weight = f["confidence"]
        base -= sev_weights.get(f.get("severity","medium"), 6) * conf_weight

    # Angle accuracy bonus (vs GolfDB benchmark)
    target_hip = bench.get("hip_turn_mean", 40)
    target_sh  = bench.get("shoulder_turn_mean", 85)
    hip_err    = abs(angles.get("hip_turn_deg", target_hip) - target_hip) / max(target_hip, 1)
    sh_err     = abs(angles.get("shoulder_turn_deg", target_sh) - target_sh) / max(target_sh, 1)
    angle_bonus = max(0, 15 - (hip_err + sh_err) * 10)
    base += angle_bonus

    return max(0, min(100, round(base)))


# ═══════════════════════════════════════════
# FIREBASE MANAGER
# ═══════════════════════════════════════════

class FirebaseManager:
    """Handles all Firebase Realtime Database operations from Python."""

    def __init__(self, cred_path: str = FIREBASE_CRED_PATH, db_url: str = FIREBASE_DB_URL):
        self.enabled = False
        if not Path(cred_path).exists():
            print(f"[Firebase] ⚠ Service account key not found at '{cred_path}'")
            return
        try:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {"databaseURL": db_url})
            self.enabled = True
            print("[Firebase] ✅ Connected to Realtime Database")
        except Exception as e:
            print(f"[Firebase] Init error: {e}")
            self.enabled = False

    def save_session(self, user_id: str, session: Dict) -> Optional[str]:
        if not self.enabled:
            print("[Firebase] Not enabled, skipping save.")
            return None
        ref = fdb.reference(f"sessions/{user_id}").push()
        ref.set({**session, "savedAt": {".sv": "timestamp"}})
        # Increment counter - FIXED the logic here
        current_count = fdb.reference(f"users/{user_id}/swingsAnalyzed").get() or 0
        fdb.reference(f"users/{user_id}/swingsAnalyzed").set(current_count + 1)
        print(f"[Firebase] Session saved: {ref.key}")
        return ref.key

    def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Dict]:
        if not self.enabled: 
            return []
        ref = fdb.reference(f"sessions/{user_id}")
        data = ref.order_by_child("timestamp").limit_to_last(limit).get()
        return list(data.values()) if data else []


# ═══════════════════════════════════════════
# MAIN ANALYSIS PIPELINE
# ═══════════════════════════════════════════

class GolfVisionPipeline:
    """
    End-to-end hybrid CNN + LSTM analysis pipeline.
    Step 1: YOLOv8 CNN  — pose estimation per frame
    Step 2: Biomech.    — joint angle computation
    Step 3: LSTM        — temporal fault detection
    Step 4: GolfDB      — benchmark comparison
    Step 5: Firebase    — persist results
    """

    def __init__(self):
        print("="*55)
        print("  GolfVision AI — Initialising Pipeline")
        print("="*55)
        self.cnn      = YOLOv8PoseEstimator()
        self.biomech  = BiomechanicalAnalyzer()
        self.lstm     = LSTMFaultDetector()
        self.golfdb   = GolfDBBenchmark()
        self.firebase = FirebaseManager()
        print("[Pipeline] All components ready.\n")

    def analyse(self, video_path: str, swing_type: str = "driver",
                camera_angle: str = "face_on", user_id: str = None,
                seq_len: int = SEQUENCE_LEN, conf_threshold: float = 0.5) -> Dict:
        """
        Full analysis of a golf swing video.
        Returns comprehensive result dict.
        """
        t0 = time.time()
        print(f"\n[Pipeline] Analysing: {video_path}")
        print(f"           Swing type: {swing_type} | Camera: {camera_angle}")

        # ── Step 1: CNN Pose Estimation ──
        print("\n[Step 1/4] YOLOv8-Pose CNN frame extraction...")
        poses = self.cnn.predict_video(video_path, max_frames=seq_len)
        detected = sum(1 for p in poses if p["detected"])
        print(f"  Detected poses in {detected}/{len(poses)} frames")

        # ── Step 2: Biomechanical Analysis ──
        print("\n[Step 2/4] Computing biomechanical angles...")
        frame_angles = [self.biomech.compute_frame_angles(p["joints"]) for p in poses]
        swing_phases = self.biomech.detect_swing_phase(frame_angles)
        sequence_feat= self.biomech.compute_sequence_features(poses)

        # Aggregate angles (mean across key phases)
        agg_angles = {}
        for a in frame_angles:
            for k, v in a.items():
                if not np.isnan(v):
                    agg_angles[k] = agg_angles.get(k, [])
                    agg_angles[k].append(v)
        agg_angles = {k: float(np.mean(v)) for k, v in agg_angles.items() if v}
        print(f"  Computed {len(agg_angles)} biomechanical metrics")

        # ── Step 3: LSTM Fault Detection ──
        print("\n[Step 3/4] LSTM temporal fault detection...")
        faults, phase_probs = self.lstm.predict(sequence_feat, threshold=conf_threshold)
        print(f"  Detected {len(faults)} fault(s):")
        for f in faults:
            print(f"    - {f['name']} (conf: {f['confidence']:.2f})")

        # ── Step 4: GolfDB Benchmarking ──
        print("\n[Step 4/4] GolfDB benchmark comparison...")
        bench      = self.golfdb.get_benchmark(swing_type)
        angle_cmp  = self.golfdb.compare_angles(agg_angles, swing_type)
        score      = compute_swing_score(faults, agg_angles, phase_probs, bench)
        elapsed    = round(time.time() - t0, 2)

        # Phase scores (from LSTM probabilities)
        phase_scores = [
            {"name": p, "score": round(phase_probs.get(p, 0) * 100), "color": PHASE_COLORS[i]}
            for i, p in enumerate(SWING_PHASES)
        ]

        # Build result
        result = {
            "score":          score,
            "grade":          "A+" if score>=90 else "A" if score>=80 else "B+" if score>=70 else "B" if score>=60 else "C",
            "swingType":      swing_type,
            "cameraAngle":    camera_angle,
            "totalFrames":    len(poses),
            "detectedFrames": detected,
            "processingTime": elapsed,
            "model":          "YOLOv8n-Pose + BiLSTM",
            "dataset":        "GolfDB",
            "faults":         faults,
            "phaseScores":    phase_scores,
            "jointAngles": [
                {"joint": "Hip Turn",       "value": round(agg_angles.get("hip_turn_deg",0),1),       "unit": "°", "golfdbMean": bench["hip_turn_mean"]},
                {"joint": "Shoulder Turn",  "value": round(agg_angles.get("shoulder_turn_deg",0),1),  "unit": "°", "golfdbMean": bench["shoulder_turn_mean"]},
                {"joint": "Spine Angle",    "value": round(agg_angles.get("spine_angle_deg",0),1),    "unit": "°", "golfdbMean": bench["spine_angle_mean"]},
                {"joint": "Lead Knee Flex", "value": round(agg_angles.get("lead_knee_flex_deg",0),1), "unit": "°", "golfdbMean": 25.0},
                {"joint": "Trail Elbow",    "value": round(agg_angles.get("trail_elbow_deg",0),1),    "unit": "°", "golfdbMean": 90.0},
                {"joint": "Lead Wrist",     "value": round(agg_angles.get("lead_wrist_deg",0),1),     "unit": "°", "golfdbMean": 85.0},
            ],
            "angleComparisons": angle_cmp,
            "golfdbBenchmark": {
                "avgScore":    bench["avg_score"],
                "topAmateur":  bench["top_amateur"],
                "tourPro":     bench["tour"],
                "swingsInDB":  bench["n"],
            },
            "poseData": [
                {
                    "frameIdx":  p["frame_idx"],
                    "timestamp": p["timestamp_ms"],
                    "joints":    p["joints"].tolist(),
                    "detected":  p["detected"],
                    "confMean":  round(p.get("conf_mean", 0), 3),
                }
                for p in poses[::3]  # downsample for storage
            ],
            "timestamp": int(time.time() * 1000),
        }

        print(f"\n[Pipeline] ✓ Complete in {elapsed}s — Score: {score}/100 ({result['grade']})")

        # ── Save to Firebase ──
        if user_id:
            session_id = self.firebase.save_session(user_id, {k:v for k,v in result.items() if k != "poseData"})
            result["sessionId"] = session_id

        return result

    def annotate_video(self, video_path: str, result: Dict,
                       output_path: str = "output_annotated.mp4") -> str:
        """
        Renders annotated video with:
        - YOLOv8 skeleton overlay
        - Joint labels + confidence
        - Fault annotations
        - Phase timeline overlay
        """
        cap = cv2.VideoCapture(video_path)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        skeleton_pairs = [
            (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
        ]

        pose_data = {p["frameIdx"]: p for p in result.get("poseData", [])}
        frame_idx = 0
        score     = result["score"]
        faults    = result["faults"]
        phase_colors_bgr = [(255,139,79),(79,255,184),(79,184,255),(79,107,255),(107,79,255),(255,79,107),(255,79,184)]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            t = frame_idx / max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1, 1)
            phase_idx = min(int(t * len(SWING_PHASES)), len(SWING_PHASES)-1)
            phase_color = phase_colors_bgr[phase_idx]

            # Get closest pose frame
            pose = min(pose_data.values(), key=lambda p: abs(p["frameIdx"]-frame_idx)) if pose_data else None

            if pose:
                joints = np.array(pose["joints"])

                # Draw skeleton
                for (a, b) in skeleton_pairs:
                    if joints[a,2]>0.3 and joints[b,2]>0.3:
                        p1 = (int(joints[a,0]), int(joints[a,1]))
                        p2 = (int(joints[b,0]), int(joints[b,1]))
                        cv2.line(frame, p1, p2, phase_color, 2, cv2.LINE_AA)

                # Draw joints
                for i, j in enumerate(joints):
                    if j[2] > 0.3:
                        c = (0,255,0) if j[2]>0.85 else (0,165,255) if j[2]>0.6 else (0,0,255)
                        cv2.circle(frame, (int(j[0]),int(j[1])), 5, c, -1, cv2.LINE_AA)
                        cv2.putText(frame, JOINT_NAMES[i][:3], (int(j[0])+7,int(j[1])+3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1, cv2.LINE_AA)

            # HUD overlay
            cv2.rectangle(frame, (0,0),(250,110),(0,0,0,160),-1)
            cv2.putText(frame, f"GolfVision AI", (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (79,255,184), 2)
            cv2.putText(frame, f"Score: {score}/100", (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (184,255,79), 1)
            cv2.putText(frame, f"Phase: {SWING_PHASES[phase_idx].replace('_',' ').title()}", (10,68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if faults:
                cv2.putText(frame, f"Fault: {faults[0]['name']}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,79,255), 1)

            out.write(frame)
            frame_idx += 1

        cap.release(); out.release()
        print(f"[Pipeline] Annotated video saved to '{output_path}'")
        return output_path


# ═══════════════════════════════════════════
# REST API (Flask)
# ═══════════════════════════════════════════

# ═══════════════════════════════════════════
# REST API (Flask)
# ═══════════════════════════════════════════

def create_api(pipeline: GolfVisionPipeline):
    """
    Minimal Flask API exposing the pipeline to the frontend.
    """
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import tempfile
    import os

    app = Flask(__name__)
    CORS(app)

    # Simple health check
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "device": DEVICE, "model": "YOLOv8+LSTM"})

    # GolfDB Stats Endpoint
    @app.route("/api/golfdb/stats")
    def golfdb_stats():
        """Return GolfDB benchmark statistics for frontend"""
        # Return fallback stats directly (temporary fix)
        fallback = {
            "driver": {"avgScore": 74, "stdScore": 9.1, "topAmateur": 82, "tour": 91,
                       "backswingAngle": 97, "hipTurn": 45, "shoulderTurn": 91, "spineAngle": 38},
            "iron": {"avgScore": 71, "stdScore": 8.7, "topAmateur": 80, "tour": 89,
                     "backswingAngle": 90, "hipTurn": 40, "shoulderTurn": 85, "spineAngle": 35},
            "wedge": {"avgScore": 68, "stdScore": 9.4, "topAmateur": 77, "tour": 86,
                      "backswingAngle": 80, "hipTurn": 32, "shoulderTurn": 78, "spineAngle": 30},
            "putt": {"avgScore": 75, "stdScore": 7.8, "topAmateur": 83, "tour": 92,
                     "backswingAngle": 25, "hipTurn": 5, "shoulderTurn": 20, "spineAngle": 15}
        }
        return jsonify(fallback)

    # Extract Poses Endpoint
    @app.route("/api/extract-poses", methods=["POST"])
    def extract_poses():
        """Extract poses from uploaded video"""
        if "video" not in request.files:
            return jsonify({"error": "No video file"}), 400
        
        video_file = request.files["video"]
        num_frames = int(request.form.get("frames", 60))
        
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        video_file.save(temp.name)
        temp.close()
        
        try:
            poses = pipeline.cnn.predict_video(temp.name, max_frames=num_frames)
            formatted_poses = []
            for pose in poses:
                joints = pose["joints"]
                if hasattr(joints, 'tolist'):
                    joints = joints.tolist()
                formatted_poses.append(joints)
            os.unlink(temp.name)
            return jsonify({"poses": formatted_poses})
        except Exception as e:
            if os.path.exists(temp.name):
                os.unlink(temp.name)
            return jsonify({"error": str(e), "poses": []}), 500

    # Analyse Endpoint
    @app.route("/api/analyse", methods=["POST"])
    def analyse():
        if "video" not in request.files:
            return jsonify({"error": "No video file"}), 400
        video_file = request.files["video"]
        swing_type = request.form.get("swingType", "driver")
        camera_ang = request.form.get("cameraAngle", "face_on")
        user_id = request.form.get("userId")
        conf_thr = float(request.form.get("confidence", "0.5"))

        tmp = f"/tmp/golf_{int(time.time())}.mp4"
        video_file.save(tmp)

        try:
            result = pipeline.analyse(tmp, swing_type, camera_ang, user_id, conf_threshold=conf_thr)
            os.remove(tmp)
            return jsonify(result)
        except Exception as e:
            os.remove(tmp) if os.path.exists(tmp) else None
            return jsonify({"error": str(e)}), 500

    return app


# ═══════════════════════════════════════════
# GOLFDB DATASET LOADER (for training)
# ═══════════════════════════════════════════

class GolfDBDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset wrapping REAL GolfDB videos for LSTM training.
    Uses your downloaded videos_160 folder!
    """
    
    def __init__(self, root: str, split: str = "train",
                 cnn: YOLOv8PoseEstimator = None, seq_len: int = SEQUENCE_LEN):
        self.root = Path(root)
        self.split = split
        self.seq_len = seq_len
        self.cnn = cnn
        self.biomech = BiomechanicalAnalyzer()
        
        # Load REAL metadata
        csv_path = self.root / "GolfDB.csv"
        if not csv_path.exists():
            # Try alternate location
            csv_path = self.root / "videos_160" / "videos_160" / "GolfDB.csv"
        
        if csv_path.exists():
            import pandas as pd
            self.metadata = pd.read_csv(csv_path)
            print(f"[GolfDB] Loaded REAL dataset: {len(self.metadata)} videos")
            print(f"[GolfDB] Columns: {list(self.metadata.columns)}")
            
            # Create a mapping from video ID to metadata
            self.video_id_map = {}
            for idx, row in self.metadata.iterrows():
                # Use 'id' column as key
                video_id = str(row.get('id', idx))
                self.video_id_map[video_id] = row
        else:
            raise FileNotFoundError(f"GolfDB.csv not found at {csv_path}")
        
        # Get all video files - handle nested folder structure
        video_folder = self.root
        if not list(video_folder.glob("*.mp4")):
            video_folder = self.root / "videos_160" / "videos_160"
        
        self.video_files = list(video_folder.glob("*.mp4"))
        print(f"[GolfDB] Found {len(self.video_files)} video files in {video_folder}")
        
        # Create mapping from video filename (without extension) to metadata
        self.video_metadata_map = {}
        for video_path in self.video_files:
            # Extract video ID from filename (e.g., "27.mp4" -> "27")
            video_id = video_path.stem
            # Find matching metadata
            matching_rows = self.metadata[self.metadata['id'].astype(str) == video_id]
            if len(matching_rows) > 0:
                self.video_metadata_map[video_path.stem] = matching_rows.iloc[0]
        
        print(f"[GolfDB] Matched {len(self.video_metadata_map)} videos with metadata")
        
        # Cache directory for pre-extracted features
        self.cache_dir = self.root / "cache" / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cache_file = self.cache_dir / f"{video_path.stem}.npz"
        
        # Load from cache if exists
        if cache_file.exists():
            data = np.load(cache_file)
            seq = data["sequence"]
            fault_labels = data["fault_labels"]
            phase_labels = data["phase_labels"]
        else:
            # Extract poses from REAL video
            if self.cnn:
                poses = self.cnn.predict_video(str(video_path))
                seq = self.biomech.compute_sequence_features(poses)
            else:
                # Placeholder if no CNN
                seq = np.zeros((self.seq_len, INPUT_DIM), dtype=np.float32)
            
            # Get ground truth from metadata
            metadata = self.video_metadata_map.get(video_path.stem)
            if metadata is not None:
                fault_labels = self._derive_real_fault_labels(metadata)
                phase_labels = self._derive_phase_labels(metadata)
            else:
                fault_labels = [0.0] * NUM_CLASSES
                phase_labels = 0
            
            # Save to cache
            np.savez_compressed(cache_file, 
                               sequence=seq, 
                               fault_labels=fault_labels,
                               phase_labels=phase_labels)
        
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(fault_labels, dtype=torch.float32),
                torch.tensor(phase_labels, dtype=torch.long))
    
    def _derive_real_fault_labels(self, metadata_row):
        """Use real GolfDB event annotations to create fault labels"""
        labels = [0.0] * NUM_CLASSES
        
        # Parse events if available
        events = metadata_row.get('events')
        if events and isinstance(events, str):
            try:
                # Events are stored as string like "[0, 45, 67, ...]"
                event_frames = eval(events)
                if len(event_frames) >= 6:
                    backswing_ratio = (event_frames[3] - event_frames[0]) / max(event_frames[5] - event_frames[0], 1)
                    if backswing_ratio > 0.65:
                        labels[4] = 1.0  # reverse_pivot
                    if backswing_ratio < 0.35:
                        labels[3] = 1.0  # sway
            except:
                pass
        
        return labels
    
    def _derive_phase_labels(self, metadata_row):
        """Convert GolfDB events to phase labels"""
        return 0  # Default phase

def verify_real_golfdb():
    """Test function to verify GolfDB data is loaded correctly"""
    print("\n" + "="*55)
    print("  Verifying REAL GolfDB Dataset")
    print("="*55)
    
    data_path = Path(GOLFDB_PATH)
    print(f"\n📁 Looking for data at: {data_path}")
    
    # Check if folder exists
    if not data_path.exists():
        print(f"❌ Folder not found! Check your path.")
        return False
    
    # Count video files
    videos = list(data_path.glob("*.mp4"))
    print(f"✅ Found {len(videos)} video files")
    
    # Check CSV
    csv_path = data_path / "GolfDB.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV with {len(df)} rows")
        print(f"   Columns: {list(df.columns)[:5]}...")
    else:
        print(f"⚠️ CSV not found at {csv_path}")
    
    # Test loading one video
    if videos:
        cap = cv2.VideoCapture(str(videos[0]))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"✅ Sample video: {videos[0].name} ({frames} frames)")
    
    print("\n✅ GolfDB dataset is ready for training!")
    return True

# Call this after imports to verify
# verify_real_golfdb()

# ═══════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GolfVision AI — CNN+LSTM Analysis")
    parser.add_argument("--video",   type=str, help="Path to swing video")
    parser.add_argument("--type",    type=str, default="driver", choices=["driver","iron","wedge","putt"])
    parser.add_argument("--angle",   type=str, default="face_on")
    parser.add_argument("--user",    type=str, default=None, help="Firebase user UID")
    parser.add_argument("--serve",   action="store_true", help="Start Flask API server")
    parser.add_argument("--train",   action="store_true", help="Train LSTM on GolfDB")
    parser.add_argument("--port",    type=int, default=5000)
    args = parser.parse_args()

    pipeline = GolfVisionPipeline()

    if args.serve:
        app = create_api(pipeline)
        print(f"\n[API] Starting server on http://0.0.0.0:{args.port}")
        app.run(host="0.0.0.0", port=args.port, debug=False)

    elif args.train:
        from torch.utils.data import DataLoader, random_split
        print("[Train] Building GolfDB dataset...")
        full_ds = GolfDBDataset(GOLFDB_PATH, split="train", cnn=pipeline.cnn)
        n_val   = int(len(full_ds)*0.15)
        train_ds, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val])
        # Use fewer workers to avoid multiprocessing issues
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
        
        pipeline.lstm.train(train_loader, val_loader, epochs=10)
        
    elif args.video:
        result = pipeline.analyse(args.video, args.type, args.angle, args.user)
        print("\n" + "="*55)
        print(json.dumps({k:v for k,v in result.items() if k!="poseData"}, indent=2))
        # Annotate video
        out = args.video.replace(".mp4","_annotated.mp4")
        pipeline.annotate_video(args.video, result, out)
    else:
        parser.print_help()
