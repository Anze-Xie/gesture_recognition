#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
本科生人工智能实验课：基于 OpenCV 与 MediaPipe 的实时手势识别与人机交互综合系统
Course Demo: Real-Time Hand Gesture Recognition & HCI System
================================================================================

功能概述 (Features):
1. [第 1 课对应模块] 视频流采集与 MediaPipe 21 个手部三维关键点 (Hand Landmarks) 实时检测、
   镜像翻转、BGR->RGB 空间转换、归一化坐标映射与 FPS 性能监控。
2. [第 2 课对应模块 - 路径 A 几何规则法] 基于向量夹角余弦定理 (Vector Angle) 与欧氏距离
   (Euclidean Distance) 的手指伸直/弯曲判定，支持识别：
   - 数字手势 0 (Fist/拳头/石头), 1, 2 (Yeah/剪刀), 3, 4, 5 (Open Palm/布)
   - 语义手势 OK、Thumbs Up (点赞)、Pinch (拇指食指捏合)
3. [第 3 课对应模块 - 综合交互应用与路径 C 混合进阶] 通过键盘按键切换 4 大实验模式：
   - 模式 1 [按键 '1']: 基础静态手势识别与关节角度/距离可视化仪表盘 (Lesson 1 & 2)
   - 模式 2 [按键 '2']: 隔空音量与屏幕亮度控制器 (Volume & Brightness HCI Control)
       * 右手拇指(4)与食指(8)捏合间距 -> 线性插值 + EMA 平滑滤波 -> 控制系统音量
       * 左手拇指(4)与食指(8)捏合间距 -> 线性插值 + EMA 平滑滤波 -> 控制屏幕亮度
       * 自动兼容 Windows (pycaw / screen_brightness_control)、macOS (osascript) 及模拟 HUD 模式
   - 模式 3 [按键 '3']: 趣味互动：人机“石头剪刀布”实时对战游戏 (Rock-Paper-Scissors AI Game)
       * 按 [SPACE] 空格键触发 3 秒倒计时，自动抓取玩家手势 (0=石头, 2=剪刀, 5=布) 并判分
   - 模式 4 [按键 '4']: 路径 C 混合进阶：在线采集手势样本 + KNN 机器学习分类器实时训练与推理
       * 按 ['c'] 键进入采集状态，按数字键 ['0'-'3'] 录制自定义手势的 42 维腕部相对归一化特征
       * 按 ['t'] 键一键训练 Scikit-Learn KNN 分类器 (或内置 NumPy KNN)，实时输出预测类别与置信度

环境依赖安装 (Installation):
    python -m pip install -r requirements.txt
    # 使用 Python 3.12 虚拟环境，详见同目录 README_ENV_ZH.md。
    # 固定 MediaPipe Solutions 兼容版本；只安装一个 OpenCV 包。
    # Windows 系统若需真实控制硬件音量与屏幕亮度，可选安装：
    pip install pycaw comtypes screen-brightness-control

运行方式 (Usage):
    python3 gesture_recognition_demo.py              # 启动电脑自带摄像头实时交互
    python3 gesture_recognition_demo.py --self-test  # 离线算法自检模式（无需摄像头硬件）
================================================================================
"""

import argparse
import csv
import math
import os
import platform
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 尝试导入 MediaPipe（若学生环境尚未安装，提供清晰的中文报错提示）
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# 尝试导入 Scikit-Learn KNN（若未安装，自动回退到本脚本内置的纯 NumPy KNN 实现）
try:
    from sklearn.neighbors import KNeighborsClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ==============================================================================
# 第一部分：手部几何特征计算与规则分类器 (对应第 2 课：路径 A 核心算法)
# ==============================================================================

class HandGeometryAnalyzer:
    """
    手部 21 关键点几何特征提取与规则分类器
    MediaPipe Hands 关键点索引定义:
      0: WRIST (手腕)
      1-4: THUMB (拇指: CMC, MCP, IP, TIP)
      5-8: INDEX (食指: MCP, PIP, DIP, TIP)
      9-12: MIDDLE (中指: MCP, PIP, DIP, TIP)
      13-16: RING (无名指: MCP, PIP, DIP, TIP)
      17-20: PINKY (小指: MCP, PIP, DIP, TIP)
    """

    FINGER_TIPS = [4, 8, 12, 16, 20]
    FINGER_PIPS = [2, 6, 10, 14, 18]
    FINGER_MCPS = [1, 5, 9, 13, 17]

    @staticmethod
    def calc_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算二维平面两点之间的欧氏距离 d = sqrt((x1-x2)^2 + (y1-y2)^2)"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def calc_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        利用向量点积余弦定理计算以 b 为顶点的夹角 ∠ABC（单位：度，范围 0~180°）
        向量 u = A - B, 向量 v = C - B
        cos(theta) = (u · v) / (||u|| * ||v||)
        """
        u = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
        v = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < 1e-6 or norm_v < 1e-6:
            return 0.0
        cos_theta = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    @classmethod
    def get_finger_states(cls, pts: List[Tuple[int, int]]) -> Tuple[List[int], List[float]]:
        """
        判定五根手指的伸直(1) / 弯曲(0) 状态，并返回五指关键关节角度列表。
        采用关节夹角法（比单纯比较 y 坐标具有更好的手掌旋转鲁棒性）：
          - 拇指：计算 (CMC_1, MCP_2, TIP_4) 夹角及指尖(4)到小指根部(17)相对跨度
          - 其余四指：计算 (MCP, PIP, TIP) 夹角，大于 150° 且指尖到手腕距离大于 PIP 到手腕距离则判定为伸直
        """
        if len(pts) < 21:
            return [0, 0, 0, 0, 0], [0.0] * 5

        wrist = pts[0]
        states = []
        angles = []

        # 1. 拇指判定：结合关节夹角 ∠1-2-4 与拇指尖(4)到小指MCP(17)的距离对比
        thumb_angle = cls.calc_angle(pts[1], pts[2], pts[4])
        angles.append(thumb_angle)
        dist_tip_pinky = cls.calc_distance(pts[4], pts[17])
        dist_ip_pinky = cls.calc_distance(pts[3], pts[17])
        thumb_open = (thumb_angle > 150.0) and (dist_tip_pinky > dist_ip_pinky * 1.08)
        states.append(1 if thumb_open else 0)

        # 2. 食指、中指、无名指、小指判定
        finger_joints = [
            (5, 6, 8),    # 食指 MCP, PIP, TIP
            (9, 10, 12),  # 中指 MCP, PIP, TIP
            (13, 14, 16), # 无名指 MCP, PIP, TIP
            (17, 18, 20)  # 小指 MCP, PIP, TIP
        ]
        for mcp_idx, pip_idx, tip_idx in finger_joints:
            ang = cls.calc_angle(pts[mcp_idx], pts[pip_idx], pts[tip_idx])
            angles.append(ang)
            dist_tip_wrist = cls.calc_distance(pts[tip_idx], wrist)
            dist_pip_wrist = cls.calc_distance(pts[pip_idx], wrist)
            is_straight = (ang > 150.0) and (dist_tip_wrist > dist_pip_wrist)
            states.append(1 if is_straight else 0)

        return states, angles

    @classmethod
    def classify_rule_gesture(cls, pts: List[Tuple[int, int]]) -> Tuple[str, List[int], Dict[str, float]]:
        """
        基于五指状态向量与相对几何距离的静态手势规则分类器
        返回: (手势名称, 五指状态列表 [拇,食,中,无,小], 几何指标字典)
        """
        states, angles = cls.get_finger_states(pts)
        # 计算手掌参考尺度（手腕 0 到中指根部 9 的像素距离，用于尺度归一化）
        palm_scale = max(cls.calc_distance(pts[0], pts[9]), 1.0)
        # 计算拇指尖(4)与食指尖(8)的欧氏距离及归一化比值
        pinch_dist = cls.calc_distance(pts[4], pts[8])
        pinch_ratio = pinch_dist / palm_scale

        metrics = {
            "palm_scale": palm_scale,
            "pinch_dist": pinch_dist,
            "pinch_ratio": pinch_ratio,
            "index_angle": angles[1],
        }

        # 特殊手势优先判定：OK 手势（拇指与食指捏合，且中指、无名指、小指伸直）
        if pinch_ratio < 0.35 and states[2] == 1 and states[3] == 1 and states[4] == 1:
            return "OK", states, metrics

        # 标准五指状态编码映射
        up_count = sum(states)
        if states == [0, 0, 0, 0, 0]:
            return "0 / Fist (Rock)", states, metrics
        elif states == [1, 0, 0, 0, 0]:
            return "Thumbs Up (Good)", states, metrics
        elif states == [0, 1, 0, 0, 0]:
            return "1 / One", states, metrics
        elif states == [0, 1, 1, 0, 0]:
            return "2 / Yeah (Scissors)", states, metrics
        elif states in ([0, 1, 1, 1, 0], [1, 1, 1, 0, 0]):
            return "3 / Three", states, metrics
        elif states == [0, 1, 1, 1, 1]:
            return "4 / Four", states, metrics
        elif states == [1, 1, 1, 1, 1]:
            return "5 / Open Palm (Paper)", states, metrics
        elif states == [1, 1, 0, 0, 1]:
            return "Spider-Man / Love", states, metrics
        elif states == [1, 0, 0, 0, 1]:
            return "6 / Call Me", states, metrics
        else:
            return f"Custom ({up_count} Fingers)", states, metrics

    @classmethod
    def extract_normalized_feature_vector(cls, pts: List[Tuple[int, int]]) -> np.ndarray:
        """
        [路径 C 混合进阶核心函数]
        将 21 个像素坐标转化为具有【平移不变性】和【尺度不变性】的 42 维特征向量：
        1. 平移归一化：以手腕关节点 (Landmark 0) 为坐标原点，计算其余关节点相对偏移 (dx, dy)
        2. 尺度归一化：除以手部最大跨度 max_dist，消除手掌离摄像头远近造成的大小差异
        """
        arr = np.array(pts, dtype=np.float64)  # shape: (21, 2)
        wrist = arr[0]
        rel_coords = arr - wrist  # 平移不变性
        max_val = np.max(np.abs(rel_coords))
        if max_val < 1e-6:
            max_val = 1.0
        norm_coords = rel_coords / max_val  # 尺度不变性，范围 [-1, 1]
        return norm_coords.flatten()  # shape: (42,)


# ==============================================================================
# 第二部分：跨平台系统硬件控制与平滑滤波器 (对应第 3 课：音量与屏幕亮度控制)
# ==============================================================================

class ExponentialMovingAverage:
    """
    指数移动平均滤波器 (EMA Smoothing Filter)
    公式: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
    用于消除摄像头关键点高频抖动，使音量/亮度推子丝滑稳定
    """
    def __init__(self, alpha: float = 0.25, initial_value: float = 50.0):
        self.alpha = alpha
        self.value = initial_value

    def update(self, measurement: float) -> float:
        self.value = self.alpha * measurement + (1.0 - self.alpha) * self.value
        return self.value


class SystemHardwareController:
    """
    跨平台系统音量与屏幕亮度控制器（支持硬件真实调用与安全模拟回退模式）
    - Windows: 自动尝试调用 pycaw 控制音量，调用 screen_brightness_control 控制亮度
    - macOS: 可选调用 osascript 调节系统音量
    - 若未安装硬件库或在虚拟机中运行，自动保持 HUD 视觉反馈，绝不崩溃报错
    """
    def __init__(self, enable_hardware: bool = True):
        self.os_name = platform.system()
        self.volume = 50.0
        self.brightness = 70.0
        self.vol_smoother = ExponentialMovingAverage(alpha=0.25, initial_value=50.0)
        self.bri_smoother = ExponentialMovingAverage(alpha=0.25, initial_value=70.0)
        self.hw_vol_available = False
        self.hw_bri_available = False
        self._last_hw_update = 0.0

        if enable_hardware:
            self._init_hardware_interfaces()

    def _init_hardware_interfaces(self):
        # 1. 初始化 Windows Pycaw 音频接口
        if self.os_name == "Windows":
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._win_volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.hw_vol_available = True
            except Exception:
                self.hw_vol_available = False

        # 2. 初始化跨平台屏幕亮度接口 screen_brightness_control
        try:
            import screen_brightness_control as sbc
            self._sbc = sbc
            # 测试读取当前亮度
            cur = sbc.get_brightness()
            if isinstance(cur, list) and len(cur) > 0:
                self.brightness = float(cur[0])
            elif isinstance(cur, (int, float)):
                self.brightness = float(cur)
            self.hw_bri_available = True
        except Exception:
            self.hw_bri_available = False

    def set_volume_from_distance(self, pinch_dist: float, min_dist: float = 25.0, max_dist: float = 180.0) -> int:
        """将拇指与食指像素间距 [min_dist, max_dist] 线性映射到音量 [0, 100] 并滤波"""
        raw_vol = float(np.interp(pinch_dist, [min_dist, max_dist], [0.0, 100.0]))
        smooth_vol = int(round(self.vol_smoother.update(raw_vol)))
        self.volume = smooth_vol

        # 限制硬件 API 调用频率（最高约 15Hz），避免阻塞视频帧率
        now = time.time()
        if now - self._last_hw_update > 0.06:
            if self.hw_vol_available and self.os_name == "Windows":
                try:
                    self._win_volume.SetMasterVolumeLevelScalar(smooth_vol / 100.0, None)
                except Exception:
                    pass
        return smooth_vol

    def set_brightness_from_distance(self, pinch_dist: float, min_dist: float = 25.0, max_dist: float = 180.0) -> int:
        """将拇指与食指像素间距 [min_dist, max_dist] 线性映射到屏幕亮度 [10, 100] 并滤波"""
        raw_bri = float(np.interp(pinch_dist, [min_dist, max_dist], [10.0, 100.0]))
        smooth_bri = int(round(self.bri_smoother.update(raw_bri)))
        self.brightness = smooth_bri

        now = time.time()
        if now - self._last_hw_update > 0.08:
            self._last_hw_update = now
            if self.hw_bri_available:
                try:
                    self._sbc.set_brightness(smooth_bri, display=0)
                except Exception:
                    pass
        return smooth_bri


# ==============================================================================
# 第三部分：路径 C 轻量级 KNN 手势分类器与 CSV 数据采集器 (对应第 3 课进阶)
# ==============================================================================

class LightweightKNNClassifier:
    """
    轻量级手势机器学习分类器：
    优先使用 scikit-learn 的 KNeighborsClassifier；若环境未装 sklearn，则自动使用纯 NumPy 实现。
    支持课堂实时按 '0'~'3' 键采集样本保存至 CSV，并按 't' 键一键完成训练与实时推理。
    """
    DEFAULT_LABELS = {
        0: "Class 0 (Fist/Stop)",
        1: "Class 1 (Peace/Victory)",
        2: "Class 2 (Open Palm)",
        3: "Class 3 (Custom Gesture)"
    }

    def __init__(self, csv_path: str = "gesture_samples.csv", k: int = 5):
        self.csv_path = csv_path
        self.k = k
        self.X_train: List[np.ndarray] = []
        self.y_train: List[int] = []
        self.is_trained = False
        self.model = None
        self._load_existing_csv()

    def _load_existing_csv(self):
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 43:
                        self.y_train.append(int(row[0]))
                        self.X_train.append(np.array([float(x) for x in row[1:]], dtype=np.float64))
            if len(self.y_train) >= self.k:
                self.train()
        except Exception:
            pass

    def add_sample(self, feature_vec: np.ndarray, label: int):
        """添加一条 42 维归一化关键点样本并追加写入 CSV"""
        self.X_train.append(feature_vec)
        self.y_train.append(label)
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([label] + [round(float(x), 5) for x in feature_vec])
        except Exception:
            pass

    def train(self) -> Tuple[bool, str]:
        """训练 KNN 分类模型"""
        n_samples = len(self.y_train)
        if n_samples < 3:
            return False, f"Need >= 3 samples (current: {n_samples})"
        k_eff = min(self.k, n_samples)
        X_arr = np.vstack(self.X_train)
        y_arr = np.array(self.y_train, dtype=int)

        if HAS_SKLEARN:
            self.model = KNeighborsClassifier(n_neighbors=k_eff, weights="distance")
            self.model.fit(X_arr, y_arr)
            self.is_trained = True
            return True, f"Sklearn KNN (k={k_eff}) trained on {n_samples} samples"
        else:
            # 纯 NumPy 距离加权 KNN 备份实现
            self._X_np = X_arr
            self._y_np = y_arr
            self._k_eff = k_eff
            self.is_trained = True
            return True, f"NumPy KNN (k={k_eff}) trained on {n_samples} samples"

    def predict(self, feature_vec: np.ndarray) -> Tuple[str, float]:
        """实时预测手势类别与置信度"""
        if not self.is_trained:
            return "Untrained (Press 'c' then '0'-'3', then 't')", 0.0

        if HAS_SKLEARN and self.model is not None:
            probs = self.model.predict_proba(feature_vec.reshape(1, -1))[0]
            best_idx = int(np.argmax(probs))
            pred_label = int(self.model.classes_[best_idx])
            conf = float(probs[best_idx])
        else:
            dists = np.linalg.norm(self._X_np - feature_vec, axis=1)
            nearest_idx = np.argsort(dists)[:self._k_eff]
            nearest_labels = self._y_np[nearest_idx]
            weights = 1.0 / (dists[nearest_idx] + 1e-6)
            label_scores: Dict[int, float] = {}
            for lbl, w in zip(nearest_labels, weights):
                label_scores[int(lbl)] = label_scores.get(int(lbl), 0.0) + float(w)
            pred_label = max(label_scores, key=label_scores.get)
            conf = label_scores[pred_label] / sum(label_scores.values())

        label_name = self.DEFAULT_LABELS.get(pred_label, f"Class {pred_label}")
        return label_name, conf


# ==============================================================================
# 第四部分：趣味互动应用——人机“石头剪刀布”对战状态机 (对应第 3 课游戏模式)
# ==============================================================================

class RockPaperScissorsGame:
    """
    人机石头剪刀布对战系统 (Rock-Paper-Scissors AI Game)
    规则映射：
      - 0 / Fist (Rock)       -> Rock (石头)
      - 2 / Yeah (Scissors)   -> Scissors (剪刀)
      - 5 / Open Palm (Paper) -> Paper (布)
    """
    MOVES = ["Rock", "Scissors", "Paper"]

    def __init__(self):
        self.player_score = 0
        self.ai_score = 0
        self.state = "IDLE"  # IDLE, COUNTDOWN, RESULT
        self.countdown_start = 0.0
        self.result_start = 0.0
        self.last_player_move = "None"
        self.last_ai_move = "None"
        self.last_outcome = "Press [SPACE] to Start Round!"

    def start_round(self):
        self.state = "COUNTDOWN"
        self.countdown_start = time.time()

    @staticmethod
    def map_gesture_to_move(gesture_name: str, finger_states: List[int]) -> Optional[str]:
        up_cnt = sum(finger_states)
        if "Fist" in gesture_name or up_cnt == 0:
            return "Rock"
        elif "Yeah" in gesture_name or finger_states[1:3] == [1, 1] and sum(finger_states[3:]) == 0:
            return "Scissors"
        elif "Open Palm" in gesture_name or up_cnt >= 4:
            return "Paper"
        return None

    def update(self, current_gesture: str, finger_states: List[int]) -> Dict[str, str]:
        now = time.time()
        if self.state == "COUNTDOWN":
            elapsed = now - self.countdown_start
            remaining = 3.0 - elapsed
            if remaining > 0:
                return {
                    "banner": f"Get Ready! Show Hand in {remaining:.1f}s...",
                    "detail": f"Current Detected: {current_gesture}"
                }
            else:
                # 倒计时结束，锁定玩家手势与 AI 随机出招
                p_move = self.map_gesture_to_move(current_gesture, finger_states)
                ai_move = random.choice(self.MOVES)
                self.last_ai_move = ai_move
                if p_move is None:
                    self.last_player_move = "Invalid Gesture"
                    self.last_outcome = "Unrecognized Move! Try Rock(0), Scissors(2), or Paper(5)."
                else:
                    self.last_player_move = p_move
                    if p_move == ai_move:
                        self.last_outcome = "DRAW! (平局)"
                    elif (p_move == "Rock" and ai_move == "Scissors") or \
                         (p_move == "Scissors" and ai_move == "Paper") or \
                         (p_move == "Paper" and ai_move == "Rock"):
                        self.player_score += 1
                        self.last_outcome = "YOU WIN! (玩家获胜)"
                    else:
                        self.ai_score += 1
                        self.last_outcome = "AI WINS! (电脑获胜)"
                self.state = "RESULT"
                self.result_start = now

        if self.state == "RESULT":
            if now - self.result_start > 4.0:
                self.state = "IDLE"
            return {
                "banner": f"Player: {self.last_player_move}  vs  AI: {self.last_ai_move}  -->  {self.last_outcome}",
                "detail": f"Score | Player: {self.player_score}  :  AI: {self.ai_score}   (Press [SPACE] for Next Round)"
            }

        return {
            "banner": self.last_outcome,
            "detail": f"Score | Player: {self.player_score}  :  AI: {self.ai_score}   (Press [SPACE] to Play)"
        }


# ==============================================================================
# 第五部分：OpenCV HUD 界面渲染与主循环引擎 (整合第 1、2、3 课全部内容)
# ==============================================================================

def draw_hud_panel(
    frame: np.ndarray,
    mode: int,
    fps: float,
    gesture_text: str,
    finger_states: List[int],
    hw_ctrl: SystemHardwareController,
    rps_info: Optional[Dict[str, str]] = None,
    knn_info: Optional[str] = None
):
    """在视频帧上绘制半透明科技感教学 HUD 仪表盘"""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 顶部状态栏背景
    cv2.rectangle(overlay, (0, 0), (w, 88), (24, 32, 44), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    mode_titles = {
        1: "MODE 1: Geometric Gesture & Angle Inspector (Lesson 1 & 2)",
        2: "MODE 2: Contactless Volume & Brightness Controller (Lesson 3 HCI)",
        3: "MODE 3: Rock-Paper-Scissors AI Battle Game (Lesson 3 Game)",
        4: "MODE 4: Hybrid KNN Machine Learning Classifier (Path C Bonus)"
    }
    cv2.putText(frame, mode_titles.get(mode, ""), (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 220, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 135, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 140), 2)

    cv2.putText(frame, f"Gesture: {gesture_text} | Fingers[T,I,M,R,P]: {finger_states}",
                (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (240, 240, 240), 2)

    # 底部按键提示栏
    cv2.rectangle(frame, (0, h - 34), (w, h), (20, 24, 32), -1)
    help_str = "Keys: [1] Basic Demo  [2] Vol/Bright Control  [3] RPS Game ([SPACE] Play)  [4] KNN ML ([0-3] Sample, [t] Train)  [q] Quit"
    cv2.putText(frame, help_str, (12, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 210, 230), 1)

    # 模式 2 专属：绘制左右侧音量与亮度实时进度条
    if mode == 2:
        # 左侧亮度条 (Brightness)
        bx1, by1, bx2, by2 = 28, 130, 56, h - 70
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 2)
        fill_h_bri = int(np.interp(hw_ctrl.brightness, [0, 100], [by2, by1]))
        cv2.rectangle(frame, (bx1 + 2, fill_h_bri), (bx2 - 2, by2 - 2), (60, 210, 255), -1)
        cv2.putText(frame, f"BRI: {int(hw_ctrl.brightness)}%", (16, by1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 210, 255), 2)

        # 右侧音量条 (Volume)
        vx1, vy1, vx2, vy2 = w - 58, 130, w - 30, h - 70
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (200, 200, 200), 2)
        fill_h_vol = int(np.interp(hw_ctrl.volume, [0, 100], [vy2, vy1]))
        cv2.rectangle(frame, (vx1 + 2, fill_h_vol), (vx2 - 2, vy2 - 2), (100, 240, 120), -1)
        cv2.putText(frame, f"VOL: {int(hw_ctrl.volume)}%", (w - 115, vy1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 240, 120), 2)

    # 模式 3 专属：绘制石头剪刀布对战横幅
    elif mode == 3 and rps_info:
        cv2.rectangle(frame, (20, 98), (w - 20, 168), (35, 45, 65), -1)
        cv2.putText(frame, rps_info["banner"], (32, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100, 255, 220), 2)
        cv2.putText(frame, rps_info["detail"], (32, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)

    # 模式 4 专属：绘制 KNN 训练与推理状态栏
    elif mode == 4 and knn_info:
        cv2.rectangle(frame, (20, 98), (w - 20, 150), (45, 35, 65), -1)
        cv2.putText(frame, knn_info, (32, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 100), 2)


def run_live_camera_demo(camera_index: int = 0):
    """启动电脑自带摄像头，运行交互式手势识别主程序"""
    if not HAS_MEDIAPIPE:
        print("[错误] 当前环境尚未安装 mediapipe。请使用项目虚拟环境并执行: python -m pip install -r requirements.txt")
        sys.exit(1)

    mp_hands = mp.solutions.hands
    mp_Draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # 初始化摄像头
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 (Index={camera_index})，请检查电脑相机权限或连接状态。")
        sys.exit(1)

    hw_ctrl = SystemHardwareController(enable_hardware=True)
    rps_game = RockPaperScissorsGame()
    knn_clf = LightweightKNNClassifier()

    current_mode = 1
    prev_time = time.time()
    status_msg = "Press [0-3] to record sample, [t] to train KNN"

    print("=================================================================")
    print(" 实时手势识别教学实验系统已启动！")
    print(" 快捷键说明: [1] 几何特征仪表盘  [2] 隔空音量/亮度控制")
    print("             [3] 石头剪刀布游戏  [4] KNN 机器学习进阶  [q] 退出")
    print("=================================================================")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # 1. 水平镜像翻转（符合照镜子直觉）并转换 BGR -> RGB
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            gesture_label = "No Hand"
            finger_states = [0, 0, 0, 0, 0]
            latest_feature_vec = None

            # 2. 解析检测到的手部关键点
            if results.multi_hand_landmarks:
                for hand_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    # 绘制 21 个关节点与骨架连线
                    mp_Draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    # 将归一化坐标 (0~1) 映射回图像像素坐标 (px, py)
                    pts: List[Tuple[int, int]] = []
                    for idx, lm in enumerate(hand_lms.landmark):
                        px, py = int(lm.x * w), int(lm.y * h)
                        pts.append((px, py))

                    # 获取左右手标签
                    handedness_label = "Right"
                    if results.multi_handedness and hand_idx < len(results.multi_handedness):
                        handedness_label = results.multi_handedness[hand_idx].classification[0].label

                    # 规则分类器计算
                    gesture_label, finger_states, metrics = HandGeometryAnalyzer.classify_rule_gesture(pts)
                    latest_feature_vec = HandGeometryAnalyzer.extract_normalized_feature_vector(pts)

                    # 模式 1：在指尖旁显示关节夹角与关键点编号
                    if current_mode == 1:
                        cv2.putText(frame, f"IdxAngle:{metrics['index_angle']:.0f}deg",
                                    (pts[6][0] + 10, pts[6][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 80), 2)

                    # 模式 2：拇指(4)与食指(8)捏合线控制音量（右手）或屏幕亮度（左手）
                    elif current_mode == 2:
                        p_thumb, p_index = pts[4], pts[8]
                        cx, cy = (p_thumb[0] + p_index[0]) // 2, (p_thumb[1] + p_index[1]) // 2
                        pinch_d = metrics["pinch_dist"]
                        cv2.line(frame, p_thumb, p_index, (0, 255, 180), 3)
                        cv2.circle(frame, (cx, cy), 10, (0, 140, 255) if pinch_d < 35 else (0, 255, 180), -1)

                        if handedness_label == "Right":
                            vol_val = hw_ctrl.set_volume_from_distance(pinch_d)
                            cv2.putText(frame, f"VOL: {vol_val}% ({int(pinch_d)}px)", (cx + 14, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 120), 2)
                        else:
                            bri_val = hw_ctrl.set_brightness_from_distance(pinch_d)
                            cv2.putText(frame, f"BRI: {bri_val}% ({int(pinch_d)}px)", (cx + 14, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 210, 255), 2)

                    # 模式 4：若 KNN 已训练，用 KNN 推理结果覆盖显示
                    elif current_mode == 4 and knn_clf.is_trained:
                        knn_pred, knn_conf = knn_clf.predict(latest_feature_vec)
                        gesture_label = f"KNN: {knn_pred} ({knn_conf*100:.0f}%)"

            # 3. 更新模式 3 游戏状态机
            rps_info = rps_game.update(gesture_label, finger_states) if current_mode == 3 else None
            knn_info = f"Samples: {len(knn_clf.y_train)} | {status_msg}" if current_mode == 4 else None

            # 4. 计算实时帧率 FPS 并渲染 HUD
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-5)
            prev_time = curr_time

            draw_hud_panel(frame, current_mode, fps, gesture_label, finger_states, hw_ctrl, rps_info, knn_info)
            cv2.imshow("AI Lab: Hand Gesture Recognition & HCI System", frame)

            # 5. 键盘事件响应
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                current_mode = int(chr(key))
            elif key == ord(' ') and current_mode == 3:
                rps_game.start_round()
            elif current_mode == 4 and key in (ord('0'), ord('1'), ord('2'), ord('3')):
                if latest_feature_vec is not None:
                    lbl = int(chr(key))
                    knn_clf.add_sample(latest_feature_vec, lbl)
                    status_msg = f"Recorded Sample for Class {lbl}! Total: {len(knn_clf.y_train)}"
                else:
                    status_msg = "No hand detected! Cannot record sample."
            elif current_mode == 4 and key == ord('t'):
                ok, msg = knn_clf.train()
                status_msg = msg

    cap.release()
    cv2.destroyAllWindows()


# ==============================================================================
# 第六部分：离线算法自检与单元测试函数 (支持无摄像头环境自动验证全部核心算法)
# ==============================================================================

def run_offline_self_test():
    """在无摄像头环境下生成合成手部 21 关键点，验证几何规则分类、平滑控制与 KNN 训练流程"""
    print("=== Running Offline Self-Test for AI Gesture Recognition Demo ===")

    # 1. 构造合成的“张开五指 (Open Palm / 5)” 21 个像素坐标
    open_palm_pts = [
        (300, 500),  # 0: Wrist
        (240, 460), (200, 410), (165, 365), (130, 320),  # 1-4: Thumb (伸直向外)
        (260, 360), (255, 290), (252, 235), (250, 180),  # 5-8: Index (伸直向上)
        (300, 350), (300, 275), (300, 215), (300, 160),  # 9-12: Middle (伸直向上)
        (340, 365), (345, 295), (348, 240), (350, 190),  # 13-16: Ring (伸直向上)
        (375, 385), (385, 325), (392, 275), (400, 230),  # 17-20: Pinky (伸直向上)
    ]
    g_name, states, metrics = HandGeometryAnalyzer.classify_rule_gesture(open_palm_pts)
    print(f"[Test 1 - Open Palm] Gesture: {g_name} | Finger States: {states} | Index Angle: {metrics['index_angle']:.1f} deg")
    assert states == [1, 1, 1, 1, 1], f"Expected [1,1,1,1,1], got {states}"

    # 2. 构造合成的“剪刀手 (Yeah / 2)” 21 个像素坐标
    scissors_pts = list(open_palm_pts)
    # 弯曲拇指、无名指、小指
    scissors_pts[4] = (285, 420)
    scissors_pts[16] = (340, 380)
    scissors_pts[20] = (375, 400)
    g_name_2, states_2, _ = HandGeometryAnalyzer.classify_rule_gesture(scissors_pts)
    print(f"[Test 2 - Scissors]  Gesture: {g_name_2} | Finger States: {states_2}")
    assert states_2 == [0, 1, 1, 0, 0], f"Expected [0,1,1,0,0], got {states_2}"

    # 3. 构造合成的“握拳 (Fist / 0)” 21 个像素坐标
    fist_pts = list(open_palm_pts)
    fist_pts[4] = (285, 420)
    fist_pts[8] = (260, 390)
    fist_pts[12] = (300, 385)
    fist_pts[16] = (340, 390)
    fist_pts[20] = (375, 405)
    g_name_0, states_0, _ = HandGeometryAnalyzer.classify_rule_gesture(fist_pts)
    print(f"[Test 3 - Fist]      Gesture: {g_name_0} | Finger States: {states_0}")
    assert states_0 == [0, 0, 0, 0, 0], f"Expected [0,0,0,0,0], got {states_0}"

    # 4. 测试音量/亮度控制器与 EMA 滤波器
    hw = SystemHardwareController(enable_hardware=False)
    v1 = hw.set_volume_from_distance(150.0)
    b1 = hw.set_brightness_from_distance(90.0)
    print(f"[Test 4 - HCI EMA]   Smoothed Volume: {v1}% | Smoothed Brightness: {b1}%")

    # 5. 测试路径 C 归一化特征提取与 KNN 分类器
    tmp_csv = "/tmp/test_gesture_samples.csv"
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)
    knn = LightweightKNNClassifier(csv_path=tmp_csv, k=3)
    for _ in range(3):
        knn.add_sample(HandGeometryAnalyzer.extract_normalized_feature_vector(fist_pts), 0)
        knn.add_sample(HandGeometryAnalyzer.extract_normalized_feature_vector(scissors_pts), 1)
        knn.add_sample(HandGeometryAnalyzer.extract_normalized_feature_vector(open_palm_pts), 2)
    ok, msg = knn.train()
    pred_lbl, conf = knn.predict(HandGeometryAnalyzer.extract_normalized_feature_vector(scissors_pts))
    print(f"[Test 5 - Path C ML] {msg} -> Predicted: {pred_lbl} (Confidence: {conf*100:.1f}%)")
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    print("=== All Self-Tests Passed Successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Course Demo: Hand Gesture Recognition with OpenCV & MediaPipe")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--self-test", action="store_true", help="Run offline unit tests without opening webcam")
    args = parser.parse_args()

    if args.self_test:
        run_offline_self_test()
    else:
        run_live_camera_demo(camera_index=args.camera)
