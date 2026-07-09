#!/usr/bin/env python3
"""
USB camera + Hailo-8 real-time **person detection** with Web Dashboard.

Captures frames from /dev/video0 via OpenCV, runs YOLOv8s inference
on the Hailo NPU (person class only), draws detection boxes, and serves a web dashboard.

Usage:
    python3 usb_camera_hailo.py [--output-dir ./output] [--fps-target 10] [--port 5000]
"""

import argparse
import glob
import io
import json
import os
import signal
import shutil
import sys
import threading
import time
import warnings

# Signal handling for graceful shutdown
_shutdown_requested = False

def _handle_sigterm(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_sigterm)

# Suppress warnings (must happen before cv2 import)
os.environ["GST_DEBUG"] = "-1"
os.environ["OPENCV_LOGLEVEL"] = "FATAL"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
warnings.filterwarnings("ignore")

# Redirect stderr during cv2/hailo imports
class _StderrSuppressor:
    def __init__(self):
        self._stderr = os.dup(2)
        self._null = os.open(os.devnull, os.O_WRONLY)

    def suppress(self):
        os.dup2(self._null, 2)

    def restore(self):
        os.dup2(self._stderr, 2)
        os.close(self._null)

_suppressor = _StderrSuppressor()
_suppressor.suppress()

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from hailo_platform import VDevice, HEF, HailoSchedulingAlgorithm, FormatType

_suppressor.restore()

# COCO 80 类物体标签（中文）
COCO_CLASSES = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公交车",
    "火车", "卡车", "船", "交通灯", "消防栓",
    "停车标志", "停车计时器", "长椅", "鸟", "猫", "狗",
    "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿",
    "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘",
    "滑雪板", "单板", "运动球", "风筝", "棒球棒",
    "棒球手套", "滑板", "冲浪板", "网球拍",
    "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗",
    "香蕉", "苹果", "三明治", "橙子", "西兰花", "胡萝卜",
    "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发",
    "盆栽", "床", "餐桌", "马桶", "电视", "笔记本电脑",
    "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
    "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶",
    "剪刀", "泰迪熊", "吹风机", "牙刷",
]

HEF_PATH = "/usr/share/hailo-models/yolov8s_h8l.hef"

# Inference config
INPUT_SIZE = 640
CONF_THRESHOLD = 0.35

# Colors for classes (pre-generated)
CLASS_COLORS = np.random.randint(80, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


# ── Shared state between detection thread and Flask ──────────────────
class DetectionState:
    """Thread-safe shared state."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frame_bytes = None       # Latest annotated frame as JPEG bytes
        self._start_time = time.monotonic()
        self._frame_count = 0
        self._total_detections = 0
        self._current_fps = 0.0
        self._current_latency = 0.0
        self._current_detections = []  # Current frame detection list
        self._detection_history = []   # Last N detection summaries
        self._running = False
        self._error = None

    def update_frame(self, frame_bytes, fps, latency, detections):
        with self._lock:
            self._frame_bytes = frame_bytes
            self._current_fps = fps
            self._current_latency = latency
            self._current_detections = detections
            self._frame_count += 1
            self._total_detections += len(detections)
            if len(self._detection_history) >= 100:
                self._detection_history.pop(0)
            self._detection_history.append({
                "frame": self._frame_count,
                "count": len(detections),
                "labels": detections[:5]
            })

    def mark_running(self):
        with self._lock:
            self._running = True
            self._start_time = time.monotonic()

    def mark_error(self, msg):
        with self._lock:
            self._error = msg

    def get_stats(self):
        with self._lock:
            uptime = time.monotonic() - self._start_time
            avg_fps = self._frame_count / uptime if uptime > 0 else 0.0
            return {
                "running": bool(self._running),
                "error": str(self._error) if self._error else None,
                "frame_count": int(self._frame_count),
                "total_detections": int(self._total_detections),
                "current_fps": float(self._current_fps),
                "avg_fps": float(avg_fps),
                "current_latency": float(self._current_latency),
                "uptime": float(uptime),
                "current_detections": [[str(d[0]), float(d[1])] for d in self._current_detections[:5]],
                "detection_history": [
                    {"frame": int(h["frame"]), "count": int(h["count"])}
                    for h in self._detection_history[-20:]
                ]
            }

    def get_frame(self):
        with self._lock:
            return self._frame_bytes


state = DetectionState()

# Output directory (set by run_detection, used by Flask routes)
_output_dir = None


# ── Flask app ──────────────────────────────────────────────────────────
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>USB 摄像头 + Hailo-8 行人检测</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    height: 100vh; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0;
  }
  .page {
    display: flex; flex-direction: column;
    height: 100vh; padding: 0;
  }
  .header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid #334155;
    padding: 6px 16px;
    display: flex; justify-content: space-between; align-items: center;
    flex-shrink: 0;
  }
  .header h1 { font-size: 1rem; font-weight: 600; white-space: nowrap; }
  .header h1 span { color: #38bdf8; }
  .status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 500;
  }
  .status-running { background: #064e3b; color: #6ee7b7; }
  .status-running::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: #34d399; animation: pulse 2s infinite; }
  .status-stopped { background: #7f1d1d; color: #fca5a5; }
  .status-stopped::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .main {
    display: grid; grid-template-columns: 1fr 240px;
    flex: 1; overflow: hidden;
    gap: 4px; padding: 4px;
    min-height: 0;
  }
  .left-col {
    display: flex; flex-direction: column;
    min-height: 0; overflow: hidden;
  }
  .card {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    padding: 6px 8px;
  }
  .card-title {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
    color: #94a3b8; margin-bottom: 6px; font-weight: 600;
  }
  .video-wrap { flex: 1; min-height: 0; display: flex; }
  .video-container { width: 100%; height: 100%; border-radius: 6px; overflow: hidden; background: #020617; }
  .video-container img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .history-table { width: 100%; font-size: 0.75rem; }
  .history-table th { text-align: left; color: #64748b; padding: 3px 4px; border-bottom: 1px solid #334155; }
  .history-table td { padding: 3px 4px; border-bottom: 1px solid #1e293b; }
  .history-table tr:last-child td { border-bottom: none; }
  .bar { height: 5px; background: #334155; border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  /* Right sidebar */
  .right-col {
    display: flex; flex-direction: column; gap: 8px;
    overflow: hidden; min-height: 0;
  }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .stat-item { background: #0f172a; border-radius: 6px; padding: 6px 8px; text-align: center; }
  .stat-value { font-size: 1.2rem; font-weight: 700; line-height: 1.2; }
  .stat-value.green { color: #4ade80; }
  .stat-value.yellow { color: #facc15; }
  .stat-value.blue { color: #38bdf8; }
  .stat-value.purple { color: #c084fc; }
  .stat-label { font-size: 0.65rem; color: #64748b; margin-top: 2px; }
  .uptime { font-size: 0.7rem; color: #64748b; margin-top: 6px; text-align: center; }
  .detection-list { overflow-y: auto; flex: 1; min-height: 0; }
  .detection-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 8px; border-radius: 4px; margin-bottom: 3px;
    background: #0f172a; font-size: 0.8rem;
  }
  .detection-item .label { font-weight: 500; }
  .detection-item .score {
    padding: 2px 6px; border-radius: 9999px; font-size: 0.7rem; font-weight: 600;
  }
  .score-high { background: #064e3b; color: #6ee7b7; }
  .score-mid { background: #713f12; color: #fde047; }
  .score-low { background: #7f1d1d; color: #fca5a5; }
  .controls { flex-shrink: 0; }
  .btn {
    width: 100%; padding: 6px; border: none; border-radius: 6px; font-size: 0.8rem;
    font-weight: 600; cursor: pointer;
  }
  .btn-restart { background: #1e40af; color: #bfdbfe; }
  .btn-restart:hover { background: #2563eb; }
  .no-frame {
    display: flex; align-items: center; justify-content: center;
    height: 100%; color: #475569; font-size: 0.9rem;
  }
  /* Gallery under video */
  .gallery-card { flex-shrink: 0; }
  .gallery-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
  }
  /* Lightbox */
  .img-browser { display: flex; flex-direction: column; height: 100%; gap: 0; }
  .img-toolbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; flex-shrink: 0;
    background: #1e293b; border-bottom: 1px solid #334155;
  }
  .img-toolbar span { font-size: 0.8rem; color: #94a3b8; }
  .img-toolbar-btns { display: flex; gap: 6px; align-items: center; }
  .btn-sm {
    padding: 4px 10px; border: 1px solid #334155; border-radius: 6px;
    background: #0f172a; color: #e2e8f0; font-size: 0.75rem; cursor: pointer;
  }
  .btn-sm:hover { background: #1e293b; border-color: #38bdf8; }
  .btn-sm:disabled { opacity: 0.4; cursor: default; }
  .img-grid-wrap { overflow-x: auto; overflow-y: hidden; padding: 6px; min-height: 160px; }
  .img-grid {
    display: grid;
    gap: 6px;
    grid-auto-flow: column;
    grid-auto-columns: minmax(300px, auto);
  }
  .img-card {
    position: relative; border-radius: 6px; overflow: hidden;
    background: #1e293b; border: 1px solid #334155;
    cursor: pointer; transition: border-color 0.2s;
  }
  .img-card:hover { border-color: #38bdf8; }
  .img-card img {
    width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block;
  }
  .img-card .img-label {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: rgba(15,23,42,0.85); padding: 3px 6px;
    font-size: 0.7rem; color: #94a3b8; text-align: center;
  }
  /* Lightbox */
  .lightbox-overlay {
    display: none; position: fixed; inset: 0; z-index: 1000;
    background: rgba(0,0,0,0.85); align-items: center; justify-content: center;
  }
  .lightbox-overlay.active { display: flex; }
  .lightbox-close {
    position: absolute; top: 12px; right: 16px;
    background: none; border: none; color: #e2e8f0;
    font-size: 1.8rem; cursor: pointer; z-index: 1001;
  }
  .lightbox-close:hover { color: #38bdf8; }
  .lightbox-nav {
    position: absolute; top: 50%; transform: translateY(-50%);
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
    color: #e2e8f0; font-size: 1.5rem; cursor: pointer;
    width: 44px; height: 44px;
    display: flex; align-items: center; justify-content: center;
    padding: 0; z-index: 1001;
  }
  .lightbox-nav:hover { background: #334155; }
  .lightbox-prev { left: 16px; }
  .lightbox-next { right: 16px; }
  .lightbox-img {
    width: 100%; height: 100%;
    max-width: 95vw; max-height: 95vh;
    object-fit: contain;
  }
  .lightbox-info {
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(30,41,59,0.9); padding: 6px 14px; border-radius: 8px;
    font-size: 0.8rem; color: #94a3b8; z-index: 1001;
  }
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1>📷 USB 摄像头 + <span>Hailo-8</span> 行人检测</h1>
    <div id="statusArea"></div>
  </div>

  <div class="main">
    <div class="left-col">
      <div class="video-wrap">
        <div class="video-container">
          <img id="videoFeed" src="/stream" alt="等待画面..."
               onerror="this.style.display='none'; document.getElementById('noFrame').style.display='flex';">
          <div id="noFrame" class="no-frame" style="display:none">等待检测画面...</div>
        </div>
      </div>
      <div class="card gallery-card">
        <div class="gallery-header">
          <div class="card-title" style="margin:0">🖼️ 图片浏览</div>
          <div class="img-toolbar-btns">
            <span id="imgCount" style="font-size:0.7rem;color:#64748b;margin-right:4px">共 0 张</span>
            <button class="btn-sm" id="prevPage" onclick="changePage(-1)">‹</button>
            <span id="pageInfo" style="font-size:0.7rem;color:#64748b;min-width:32px;text-align:center">1/1</span>
            <button class="btn-sm" id="nextPage" onclick="changePage(1)">›</button>
          </div>
        </div>
        <div class="img-grid-wrap">
          <div class="img-grid" id="imgGrid">
            <div style="color:#475569;text-align:center;padding:20px;grid-column:1/-1">加载中...</div>
          </div>
        </div>
      </div>
    </div>

    <div class="right-col">
      <div class="card" style="flex-shrink:0">
        <div class="card-title">运行状态</div>
        <div class="stats-grid">
          <div class="stat-item">
            <div class="stat-value green" id="fpsVal">--</div>
            <div class="stat-label">FPS</div>
          </div>
          <div class="stat-item">
            <div class="stat-value yellow" id="latencyVal">--</div>
            <div class="stat-label">延迟 ms</div>
          </div>
          <div class="stat-item">
            <div class="stat-value blue" id="frameVal">0</div>
            <div class="stat-label">总帧数</div>
          </div>
          <div class="stat-item">
            <div class="stat-value purple" id="detVal">0</div>
            <div class="stat-label">检测数</div>
          </div>
        </div>
        <div class="uptime" id="uptimeVal">运行 0s</div>
      </div>

      <div class="card" style="flex:1;min-height:0;display:flex;flex-direction:column">
        <div class="card-title">当前检测</div>
        <div class="detection-list" id="detList">
          <div style="color: #475569; text-align: center; padding: 10px;">等待检测...</div>
        </div>
      </div>

      <div class="card" style="flex-shrink:0">
        <div class="card-title">检测历史（最近 10 帧）</div>
        <table class="history-table">
          <thead><tr><th style="width:50px">帧号</th><th style="width:40px">数</th><th>分布</th></tr></thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>

      <div class="controls">
        <button class="btn btn-restart" onclick="restartService()">🔄 重启服务</button>
      </div>
    </div>
  </div>
</div>

<!-- Lightbox -->
<div class="lightbox-overlay" id="lightbox">
  <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
  <button class="lightbox-nav lightbox-prev" onclick="navLightbox(-1)">&#8249;</button>
  <button class="lightbox-nav lightbox-next" onclick="navLightbox(1)">&#8250;</button>
  <img class="lightbox-img" id="lightboxImg" src="">
  <div class="lightbox-info" id="lightboxInfo"></div>
</div>

<script>
// ── Image browser state ──
let currentPage = 1;
let totalPages = 1;
let currentImages = [];
let pageImages = {};          // cache: page number -> images[] (ascending / oldest-first)
let globalCount = 0;          // total images known
let globalIndex = -1;         // current lightbox position (0-based, chronological)

async function loadImages(page) {
  try {
    const res = await fetch(`/api/images?page=${page}&per_page=24`);
    const data = await res.json();
    currentPage = data.page;
    totalPages = data.total_pages;
    currentImages = data.images;
    pageImages[page] = data.images;  // cache ascending order
    globalCount = data.total;
    renderGrid(data.images);
    document.getElementById('imgCount').textContent = `共 ${data.total} 张图片`;
    document.getElementById('pageInfo').textContent = `${currentPage} / ${totalPages}`;
    document.getElementById('prevPage').disabled = currentPage <= 1;
    document.getElementById('nextPage').disabled = currentPage >= totalPages;
  } catch (e) {
    document.getElementById('imgGrid').innerHTML =
      `<div style="color:#ef4444;text-align:center;padding:40px;grid-column:1/-1;">加载失败: ${e.message}</div>`;
  }
}

function renderGrid(images) {
  const grid = document.getElementById('imgGrid');
  if (images.length === 0) {
    grid.innerHTML = '<div style="color:#475569;text-align:center;padding:40px;">暂无图片</div>';
    return;
  }
  grid.innerHTML = images.map((img, idx) => {
    const frameNum = img.name.replace('frame_', '').replace('.jpg', '').replace(/^0+/, '') || '0';
    return `<div class="img-card" onclick="openLightbox(${idx})">
      <img src="${img.url}" alt="${img.name}" loading="lazy">
    </div>`;
  }).join('');
}

function changePage(delta) {
  const newPage = currentPage + delta;
  if (newPage >= 1 && newPage <= totalPages) { loadImages(newPage); }
}

// ── Lightbox (cross-page, chronological) ──
function getGlobalBounds(page) {
  const per = 24;
  return [page * per, Math.min(page * per + per - 1, globalCount - 1)];
}

async function loadPage(p) {
  if (pageImages[p]) return pageImages[p];
  try {
    const res = await fetch(`/api/images?page=${p}&per_page=24`);
    const data = await res.json();
    pageImages[p] = data.images;
    globalCount = data.total;
    totalPages = data.total_pages;
    return data.images;
  } catch { return []; }
}

function updateLightboxDisplay(page, idx) {
  const imgs = pageImages[page];
  if (!imgs || !imgs[idx]) return;
  const img = imgs[idx];
  const frameNum = img.name.replace('frame_', '').replace('.jpg', '').replace(/^0+/, '') || '0';
  const sizeKB = (img.size / 1024).toFixed(1);
  document.getElementById('lightboxImg').src = img.url;
  document.getElementById('lightboxInfo').textContent =
    `帧 #${frameNum}  ·  ${sizeKB} KB  ·  第 ${page}/${totalPages} 页`;
}

async function loadAndNavigate(delta) {
  let newGlobal = globalIndex + delta;
  if (newGlobal < 0 || newGlobal >= globalCount) return;
  globalIndex = newGlobal;

  const per = 24;
  const targetPage = Math.floor(globalIndex / per) + 1;

  if (targetPage !== currentPage && targetPage >= 1) {
    const imgs = await loadPage(targetPage);
    if (imgs.length > 0) {
      currentPage = targetPage;
      pageImages[targetPage] = imgs;
      updateLightboxDisplay(targetPage, globalIndex % per);
      document.getElementById('pageInfo').textContent = `${currentPage} / ${totalPages}`;
      return;
    }
  }

  updateLightboxDisplay(currentPage, globalIndex % per);
}

async function openLightbox(idx) {
  globalIndex = (currentPage - 1) * 24 + idx;
  document.getElementById('lightbox').classList.add('active');
  updateLightboxDisplay(currentPage, idx);
}

function closeLightbox() {
  document.getElementById('lightbox').classList.remove('active');
}

async function navLightbox(delta) {
  await loadAndNavigate(delta);
}

// Keyboard: Esc to close, arrows to navigate
document.addEventListener('keydown', (e) => {
  const lb = document.getElementById('lightbox');
  if (lb.classList.contains('active')) {
    if (e.key === 'Escape') closeLightbox();
    if (e.key === 'ArrowLeft') { e.preventDefault(); navLightbox(-1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); navLightbox(1); }
  }
});

function updateScoreClass(score) {
  if (score >= 0.5) return 'score-high';
  if (score >= 0.3) return 'score-mid';
  return 'score-low';
}

function formatUptime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  if (h > 0) return `运行 ${h}小时${m}分`;
  if (m > 0) return `运行 ${m}分${sec}秒`;
  return `运行 ${sec}秒`;
}

let lastPoll = 0;
async function pollStats() {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    lastPoll = Date.now();

    // Status
    const statusArea = document.getElementById('statusArea');
    if (data.running) {
      statusArea.innerHTML = '<span class="status-badge status-running">运行中</span>';
    } else {
      statusArea.innerHTML = '<span class="status-badge status-stopped">已停止</span>';
    }

    // Stats
    document.getElementById('fpsVal').textContent = Math.floor(data.current_fps);
    document.getElementById('latencyVal').textContent = Math.floor(data.current_latency);
    document.getElementById('frameVal').textContent = data.frame_count.toLocaleString();
    document.getElementById('detVal').textContent = data.total_detections.toLocaleString();
    document.getElementById('uptimeVal').textContent = formatUptime(data.uptime);

    // Current detections
    const detList = document.getElementById('detList');
    if (data.current_detections.length === 0) {
      detList.innerHTML = '<div style="color: #475569; text-align: center; padding: 20px;">未检测到物体</div>';
    } else {
      detList.innerHTML = data.current_detections.map(d =>
        `<div class="detection-item">
          <span class="label">${d[0]}</span>
          <span class="score ${updateScoreClass(d[1])}">${(d[1] * 100).toFixed(0)}%</span>
        </div>`
      ).join('');
    }

    // History
    const historyBody = document.getElementById('historyBody');
    const maxCount = Math.max(1, ...data.detection_history.map(h => h.count));
    const rows = data.detection_history.slice(-10).map(h => {
      const pct = (h.count / maxCount) * 100;
      const colors = h.count >= 5 ? '#ef4444' : h.count >= 1 ? '#facc15' : '#475569';
      return `<tr>
        <td>#${h.frame}</td>
        <td>${h.count}</td>
        <td style="flex:1; margin-left: 8px;"><div class="bar"><div class="bar-fill" style="width:${pct}%; background:${colors};"></div></div></td>
      </tr>`;
    }).join('');
    historyBody.innerHTML = rows;

  } catch (e) {
    // Silently fail — stats may be temporarily unavailable
  }
}

async function restartService() {
  if (!confirm('确定要重启检测服务吗？')) return;
  try {
    await fetch('/api/restart', { method: 'POST' });
    alert('重启请求已发送');
  } catch (e) {
    alert('重启失败: ' + e.message);
  }
}

// Poll stats every 1 second
setInterval(pollStats, 1000);
pollStats();
loadImages(1);

// Auto-refresh image gallery every 5 seconds (only if new images detected)
let lastImageCount = 0;
setInterval(async () => {
  try {
    const res = await fetch('/api/images/total');
    const data = await res.json();
    if (data.total !== lastImageCount) {
      lastImageCount = data.total;
      pageImages = {};             // invalidate cache
      const maxValidPage = Math.max(1, Math.ceil(data.total / 24));
      if (currentPage > maxValidPage) currentPage = 1;
      loadImages(currentPage);
    } else if (lastImageCount === 0) {
      lastImageCount = data.total;
    }
  } catch (e) { /* ignore */ }
}, 5000);
</script>
</body>
</html>
"""

app = Flask(__name__)

@app.route('/')
def index():
    resp = render_template_string(HTML_TEMPLATE)
    from flask import make_response
    r = make_response(resp)
    r.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    return r

@app.route('/stream')
def stream():
    """MJPEG stream endpoint."""
    def generate():
        while True:
            frame = state.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def api_stats():
    return jsonify(state.get_stats())

@app.route('/api/restart', methods=['POST'])
def api_restart():
    """Restart detection by sending SIGTERM to self (systemd will restart)."""
    import signal as sig
    sig.signal(sig.SIGTERM, _handle_sigterm)
    _shutdown_requested = True
    return jsonify({"status": "restarting"})

@app.route('/api/images')
def api_images():
    """List output images with pagination, newest first."""
    global _output_dir
    if _output_dir is None or not os.path.isdir(_output_dir):
        return jsonify({"images": [], "total": 0, "page": 1, "total_pages": 0})

    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 24, type=int)
    per_page = min(per_page, 100)  # cap

    # Collect all files with metadata
    images = []
    for fname in os.listdir(_output_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg')):
            continue
        fpath = os.path.join(_output_dir, fname)
        try:
            stat = os.stat(fpath)
            images.append({
                "name": fname,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": f"/api/image/{fname}",
            })
        except OSError:
            images.append({
                "name": fname,
                "size": 0,
                "mtime": 0,
                "url": f"/api/image/{fname}",
            })

    # Sort by mtime descending — newest first
    images.sort(key=lambda x: x["mtime"], reverse=True)

    total = len(images)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    start = (page - 1) * per_page
    end = start + per_page
    images = images[start:end]

    return jsonify({
        "images": images,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
    })

@app.route('/api/images/total')
def api_images_total():
    """Get total image count only (for lightbox cross-page navigation)."""
    global _output_dir
    if _output_dir is None or not os.path.isdir(_output_dir):
        return jsonify({"total": 0})
    total = sum(1 for f in os.listdir(_output_dir)
                if f.lower().endswith(('.jpg', '.jpeg')))
    return jsonify({"total": total})

@app.route('/api/image/<path:filename>')
def api_image(filename):
    """Serve a single output image."""
    global _output_dir
    if _output_dir is None:
        return jsonify({"error": "output_dir not set"}), 500
    # Sanitize filename to prevent directory traversal
    safe_name = os.path.basename(filename)
    fpath = os.path.join(_output_dir, safe_name)
    if not os.path.isfile(fpath):
        return jsonify({"error": "not found"}), 404
    from flask import send_file
    return send_file(fpath, mimetype='image/jpeg')


# ── Detection functions ────────────────────────────────────────────────

def load_hailo_model(hef_path):
    """Load HEF model onto Hailo NPU."""
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    device = VDevice(params)
    hef = HEF(hef_path)
    infer = device.create_infer_model(hef_path)

    input_format = hef.get_input_vstream_infos()[0].format.type
    infer.input().set_format_type(input_format)
    for output in infer.outputs:
        output.set_format_type(FormatType.FLOAT32)

    cfg = infer.configure()

    inp_name = infer.input_names[0]
    out_name = infer.output_names[0]
    out_shape = infer.output(out_name).shape

    return device, cfg, inp_name, out_name, out_shape


def init_camera(device="/dev/video0", width=640, height=480):
    """Open USB camera via V4L2."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        state.mark_error(f"无法打开摄像头 {device}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.read()  # Warm-up
    return cap, actual_w, actual_h


def preprocess_frame(frame):
    """Resize frame to 640x640 for YOLOv8 input."""
    return cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))


def parse_detections(output_list, orig_w, orig_h, conf_thresh=0.35):
    """Parse NMS output — only detect '人' (person, COCO class 0)."""
    detections = []
    # COCO class 0 = person — skip all other classes
    det_array = output_list[0]
    if not isinstance(det_array, np.ndarray) or det_array.size == 0:
        return detections
    for row in det_array:
        score = row[4]
        if score < conf_thresh:
            continue
        y_min, x_min, y_max, x_max = row[0], row[1], row[2], row[3]
        x1 = int(x_min * orig_w)
        y1 = int(y_min * orig_h)
        x2 = int(x_max * orig_w)
        y2 = int(y_max * orig_h)
        detections.append(["人", score, x1, y1, x2, y2])
    return detections


def draw_timestamp(frame):
    """Draw current time in bottom-left corner."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    (tw, th) = cv2.getTextSize(ts, font, scale, thickness)[0]
    h, w = frame.shape[:2]
    x, y = 10, h - 10
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, ts, (x, y - 4), font, scale, (255, 255, 255), thickness)


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    for label_text, score, x1, y1, x2, y2 in detections:
        cls_id = COCO_CLASSES.index(label_text) if label_text in COCO_CLASSES else 0
        color = CLASS_COLORS[cls_id].tolist()
        label = f"{label_text}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame


def _box_iou(a, b):
    """Intersection over union for two [x1,y1,x2,y2] boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_blank_region(frame, x1, y1, x2, y2, mean_thresh=240, std_thresh=15):
    """Check if the detected region is mostly blank/overexposed (e.g. white paper)."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return True
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    std = gray.std()
    return mean > mean_thresh and std < std_thresh


def run_detection(output_dir=None, target_fps=10, conf_thresh=0.35, port=5000):
    """Main detection loop with web dashboard."""
    global _output_dir
    global CONF_THRESHOLD
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Session management: increment counter but DO NOT delete old session files
        session_file = os.path.join(output_dir, ".session_id")
        try:
            with open(session_file) as f:
                prev_session = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            prev_session = None

        cur_session = (prev_session or 0) + 1
        with open(session_file, "w") as f:
            f.write(str(cur_session))

        _output_dir = output_dir
        run_detection._session_id = cur_session
    else:
        _output_dir = None
        run_detection._session_id = 0

    CONF_THRESHOLD = conf_thresh

    # Start Flask in background thread
    def run_flask():
        app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    print(f"\n🌐 Web dashboard: http://0.0.0.0:{port}")
    print(f"   Stream URL:    http://<YOUR_IP>:{port}/stream")
    print(f"   API stats:     http://<YOUR_IP>:{port}/api/stats")
    print()

    # Initialize detection
    print("⏳ Loading Hailo model...")
    device, cfg, inp_name, out_name, out_shape = load_hailo_model(HEF_PATH)
    print("✓ Model loaded")

    print("⏳ Opening camera...")
    cap, cam_w, cam_h = init_camera()
    print(f"✓ Camera {cam_w}x{cam_h}")

    state.mark_running()

    out_data = np.empty(out_shape, dtype=np.float32)

    frame_count = 0
    fps_times = []

    try:
        while True:
            frame_start = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                continue

            # Preprocess & infer
            resized = preprocess_frame(frame)
            bindings = cfg.create_bindings(
                input_buffers={inp_name: resized},
                output_buffers={out_name: out_data}
            )

            cfg.wait_for_async_ready(timeout_ms=1000)
            t0 = time.monotonic()
            job = cfg.run_async([bindings])
            job.wait(5000)
            t1 = time.monotonic()
            infer_ms = (t1 - t0) * 1000

            # Parse & draw
            out = bindings.output(out_name).get_buffer()
            detections = parse_detections(out, cam_w, cam_h, CONF_THRESHOLD)
            annotated = draw_detections(frame.copy(), detections)

            # FPS calc
            frame_time = time.monotonic() - frame_start
            fps_times.append(frame_time)
            window = min(20, len(fps_times))
            avg_fps = window / sum(fps_times[-window:])

            # Update shared state for web dashboard
            _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state.update_frame(jpeg.tobytes(), avg_fps, infer_ms,
                             [[d[0], d[1]] for d in detections[:5]])

            # Save frame to disk — check every 60 frames (~6s), skip if nearly identical
            if output_dir and frame_count % 60 == 0 and len(detections) > 0:
                sid = run_detection._session_id
                d = detections[0]  # [label, score, x1, y1, x2, y2]
                x1, y1, x2, y2 = int(d[2]), int(d[3]), int(d[4]), int(d[5])
                person_crop = annotated[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else annotated
                # Region-based histogram: 4x4 grid, 16 bins each = 256-dim feature
                gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                ch, cw = gray.shape[:2]
                feat = []
                for sy in range(0, ch, max(1, ch // 4)):
                    for sx in range(0, cw, max(1, cw // 4)):
                        cell = gray[sy:min(sy + ch // 4, ch), sx:min(sx + cw // 4, cw)]
                        h = cv2.calcHist([cell], [0], None, [16], [0, 256]).flatten()
                        feat.append(h)
                feat = np.array(feat).flatten().astype(np.float32)
                feat /= (feat.sum() or 1)

                prev = getattr(run_detection, "_last_saved_state", None)
                should_save = True
                # Skip blank/overexposed regions (e.g. white paper false positives)
                if _is_blank_region(frame, x1, y1, x2, y2):
                    should_save = False
                if prev is not None:
                    try:
                        bh = cv2.compareHist(feat, prev["feat"], cv2.HISTCMP_BHATTACHARYYA)
                        iou = _box_iou([x1, y1, x2, y2], prev["bbox"])
                        # Skip only if person looks very similar (histogram matches closely)
                        # AND bbox heavily overlaps
                        if bh < 0.3 and iou > 0.8:
                            should_save = False
                    except Exception:
                        # If comparison fails (size mismatch, etc.), just save
                        pass
                if should_save:
                    draw_timestamp(annotated)
                    run_detection._last_saved_state = {"feat": feat, "bbox": [x1, y1, x2, y2]}
                    path = os.path.join(output_dir, f"sess_{sid:04d}_f{frame_count:05d}.jpg")
                    cv2.imwrite(path, annotated)

            frame_count += 1

            # Shutdown check
            if _shutdown_requested:
                print("\n🛑 Shutdown signal received")
                break

            # Frame rate control
            elapsed = time.monotonic() - t1
            target_interval = 1.0 / target_fps
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

    except KeyboardInterrupt:
        pass

    print(f"\n📊 Processed {frame_count} frames")
    cap.release()
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="USB Camera + Hailo-8 Real-time Object Detection (Web Dashboard)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default (10 FPS, port 5000)
  %(prog)s --port 8080              # Web UI on port 8080
  %(prog)s --fps-target 20          # Target 20 FPS
  %(prog)s --conf-thresh 0.3        # Slightly lower threshold
        """
    )
    parser.add_argument("--output-dir", default="./output",
                        help="Directory to save annotated frames (default: ./output)")
    parser.add_argument("--fps-target", type=int, default=10,
                        help="Target processing FPS (default: 10)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Web dashboard port (default: 5000)")
    parser.add_argument("--conf-thresh", type=float, default=0.35,
                        help="Confidence threshold 0-1 (default: 0.35)")
    args = parser.parse_args()

    run_detection(
        output_dir=args.output_dir,
        target_fps=args.fps_target,
        conf_thresh=args.conf_thresh,
        port=args.port
    )
