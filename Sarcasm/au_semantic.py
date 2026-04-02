# au_semantic.py
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2

from feat import Detector


def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-6 or math.isnan(float(fps)):
        return 25.0
    return float(fps)


def _pick_au_series(df: pd.DataFrame, au_num: str) -> Optional[pd.Series]:
    au_num2 = au_num.zfill(2)
    candidates = [
        f"AU{au_num2}_r",
        f"AU{au_num2}",
        f"AU{au_num2}_c",
        f"AU{int(au_num)}_r",
        f"AU{int(au_num)}",
        f"AU{int(au_num)}_c",
    ]
    for c in candidates:
        if c in df.columns:
            return df[c].astype(float)
    return None


def extract_au_framewise(video_path: str, detector: Optional[Detector] = None) -> Tuple[pd.DataFrame, float]:
    fps = get_video_fps(video_path)
    detector = detector or Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model=None
    )

    fex = detector.detect_video(video_path)
    df = fex.to_dataframe() if hasattr(fex, "to_dataframe") else pd.DataFrame(fex)

    au_nums = ["1", "2", "4", "6", "7", "12", "14", "17", "23", "24", "46"]
    out = {}
    for au in au_nums:
        s = _pick_au_series(df, au)
        if s is None:
            continue
        out[f"AU{au.zfill(2)}"] = s.fillna(0.0).values

    au_df = pd.DataFrame(out).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return au_df, fps


@dataclass
class StateMachineConfig:
    onset_min_frames: int = 5
    offset_min_frames: int = 3
    min_dur_ms: int = 250
    max_dur_ms: int = 2000
    merge_gap_ms: int = 150


def _segments_from_mask(mask: np.ndarray, fps: float, cfg: StateMachineConfig) -> List[Tuple[int, int]]:
    n = len(mask)
    if n == 0:
        return []

    segments = []
    in_seg = False
    start = 0
    on_count = 0
    off_count = 0

    for i in range(n):
        if mask[i]:
            on_count += 1
            off_count = 0
        else:
            off_count += 1
            on_count = 0

        if (not in_seg) and on_count >= cfg.onset_min_frames:
            in_seg = True
            start = i - cfg.onset_min_frames + 1

        if in_seg and off_count >= cfg.offset_min_frames:
            end = i - cfg.offset_min_frames
            segments.append((start, max(start, end)))
            in_seg = False

    if in_seg:
        segments.append((start, n - 1))

    min_frames = int(math.ceil(cfg.min_dur_ms / 1000.0 * fps))
    max_frames = int(math.floor(cfg.max_dur_ms / 1000.0 * fps))

    out = []
    for s, e in segments:
        dur = e - s + 1
        if dur < max(1, min_frames):
            continue
        if max_frames > 0 and dur > max_frames:
            cur = s
            while cur <= e:
                cur_end = min(e, cur + max_frames - 1)
                out.append((cur, cur_end))
                cur = cur_end + 1
        else:
            out.append((s, e))

    gap_frames = int(math.floor(cfg.merge_gap_ms / 1000.0 * fps))
    if gap_frames > 0 and len(out) > 1:
        merged = [out[0]]
        for s, e in out[1:]:
            ps, pe = merged[-1]
            if s - pe - 1 <= gap_frames:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        out = merged

    return out


def _intensity_max(au_df: pd.DataFrame, cols: List[str], s: int, e: int) -> float:
    cols_exist = [c for c in cols if c in au_df.columns]
    if not cols_exist:
        return 0.0
    return float(np.max(au_df.loc[s:e, cols_exist].to_numpy()))


def _score_simple(intensity: float, thr: float) -> float:
    if intensity <= thr:
        return 0.0
    return float(min(1.0, (intensity - thr) / 3.0))


def detect_au_events_from_video(
    video_path: str,
    detector: Detector,
    cfg: Optional[StateMachineConfig] = None,
    actor: str = "speaker"
) -> Dict:
    """
    Returns:
      {
        "fps": ...,
        "num_frames": ...,
        "events": [ ... ]
      }
    """
    cfg = cfg or StateMachineConfig()
    au_df, fps = extract_au_framewise(video_path, detector=detector)
    n = len(au_df)

    def au(name: str) -> np.ndarray:
        return au_df[name].to_numpy() if name in au_df.columns else np.zeros(n, dtype=float)

    events: List[Dict] = []

    # 1) SMIRK_CONTEMPT: AU14>=2 (optional AU12>=1)
    mask_smirk = (au("AU14") >= 2.0)
    for s, e in _segments_from_mask(mask_smirk, fps, cfg):
        inten = _intensity_max(au_df, ["AU14"], s, e)
        events.append({
            "event_type": "SMIRK_CONTEMPT",
            "actor": actor,
            "t_start": round(s / fps, 3),
            "t_end": round((e + 1) / fps, 3),
            "intensity": round(inten, 3),
            "score": round(_score_simple(inten, 2.0), 3)
        })

    # 2) LIP_PRESS: AU24>=2 (optional AU23>=1)
    mask_lip_press = (au("AU24") >= 2.0)
    for s, e in _segments_from_mask(mask_lip_press, fps, cfg):
        inten = _intensity_max(au_df, ["AU24"], s, e)
        events.append({
            "event_type": "LIP_PRESS",
            "actor": actor,
            "t_start": round(s / fps, 3),
            "t_end": round((e + 1) / fps, 3),
            "intensity": round(inten, 3),
            "score": round(_score_simple(inten, 2.0), 3)
        })

    # 3) LIP_TIGHTEN: AU23>=2 (optional AU17>=1)
    mask_lip_tighten = (au("AU23") >= 2.0)
    for s, e in _segments_from_mask(mask_lip_tighten, fps, cfg):
        inten = _intensity_max(au_df, ["AU23"], s, e)
        events.append({
            "event_type": "LIP_TIGHTEN",
            "actor": actor,
            "t_start": round(s / fps, 3),
            "t_end": round((e + 1) / fps, 3),
            "intensity": round(inten, 3),
            "score": round(_score_simple(inten, 2.0), 3)
        })

    # 4) DEADPAN: key AUs all < 1.0
    deadpan_cols = ["AU12", "AU14", "AU23", "AU24", "AU04", "AU01", "AU02", "AU07"]
    deadpan_stack = np.vstack([au(c) for c in deadpan_cols])
    mask_deadpan = np.all(deadpan_stack < 1.0, axis=0)
    for s, e in _segments_from_mask(mask_deadpan, fps, cfg):
        inten = float(np.max(deadpan_stack[:, s:e+1])) if deadpan_stack.size else 0.0
        events.append({
            "event_type": "DEADPAN",
            "actor": actor,
            "t_start": round(s / fps, 3),
            "t_end": round((e + 1) / fps, 3),
            "intensity": round(1.0 - min(1.0, inten / 1.0), 3),
            "score": 1.0
        })

    # 5) SQUINT: AU07>=2 (optional AU06>=1)
    mask_squint = (au("AU07") >= 2.0)
    for s, e in _segments_from_mask(mask_squint, fps, cfg):
        inten = _intensity_max(au_df, ["AU07"], s, e)
        events.append({
            "event_type": "SQUINT",
            "actor": actor,
            "t_start": round(s / fps, 3),
            "t_end": round((e + 1) / fps, 3),
            "intensity": round(inten, 3),
            "score": round(_score_simple(inten, 2.0), 3)
        })

    # 6) BROW_RAISE: AU01>=1 AND AU02>=1
    mask_brow_raise = (au("AU01") >= 1.0) & (au("AU02") >= 1.0)
    for s, e in _segments_from_mask(mask_brow_raise, fps, cfg):
        inten = _intensity_max(au_df, ["AU01", "AU02"], s, e)
        events.append({
            "event_type": "BROW_RAISE",
            "actor": actor,
            "t_start": round(s / fps, 3),
            "t_end": round((e + 1) / fps, 3),
            "intensity": round(inten, 3),
            "score": round(_score_simple(inten, 1.0), 3)
        })

    # 7) BROW_FURROW: AU04>=2 (optional AU07>=1)
    mask_brow_furrow = (au("AU04") >= 2.0)
    for s, e in _segments_from_mask(mask_brow_furrow, fps, cfg):
        inten = _intensity_max(au_df, ["AU04"], s, e)
        events.append({
            "event_type": "BROW_FURROW",
            "actor": actor,
            "t_start": round(s / fps, 3),
            "t_end": round((e + 1) / fps, 3),
            "intensity": round(inten, 3),
            "score": round(_score_simple(inten, 2.0), 3)
        })

    events.sort(key=lambda x: (x["t_start"], x["t_end"], x["event_type"]))
    return {"fps": fps, "num_frames": int(n), "events": events}