# Workplace Safety Sentinel — Track A (PipeGen)

**Embedded SF: CraftifAI Physical AI Hackathon — Track 1: Workplace Safety Sentinel**

## Problem

Workers enter hazard zones or skip PPE, and nobody notices until an incident. Manual monitoring of site cameras is unreliable and does not scale.

## Solution

A real-time edge AI vision pipeline built with **PipeGen v0.5.1**. The exported DeepStream runtime runs YOLOv8n object detection followed by YOLO11n-pose on each frame, overlays boxes and skeletons, and writes live frame metadata to JSON Lines. The PipeGen design IR shows separate model branches; the exported runtime processes them sequentially so pose shapes and detections can be emitted together. This JSON feed is the actionable output for an operator or downstream service.

## Pipeline Architecture

| Component | Detail |
|-----------|--------|
| **Input** | Video file (`.mp4`) or RTSP stream, 3840×2160 resolution |
| **Task 1** | Object Detection — YOLOv8n (COCO, 80 classes) |
| **Task 2** | Pose Estimation — YOLO11n-pose (17 COCO keypoints, sequential in exported runtime) |
| **Model Precision** | FP32 TensorRT engines |
| **Output** | Live display or saved video, plus JSON Lines metadata at `pipegen-export/out/events/cam0.jsonl` |
| **Framework** | DeepStream 7.1, TensorRT 10.3.0 |
| **Hardware** | NVIDIA RTX 3050 6GB (x86-64 Linux) |

### Pipeline Flow

```
Input (video/RTSP)
  └──► YOLOv8n Detection ──► YOLO11n Pose ──► Decode/draw boxes + skeletons
                                      ├──► JSON Lines metadata (detection boxes + pose shapes)
                                      └──► Live display or saved video
```

### Models

| Model | Purpose | Parameters | Engine Size | TRT Layers | Speed |
|-------|---------|-----------|-------------|------------|-------|
| YOLOv8n | COCO object detection | 3.15M | 21.6 MB | 236 | **167 fps** (FP32) |
| YOLO11n-pose | Pose estimation (17 keypoints) | 2.87M | 19.2 MB | 250 | FP32 |

The 167 fps figure is the YOLOv8n engine benchmark from PipeGen, not an
end-to-end throughput measurement for the two-model pipeline.

## Repository Contents

```
├── pipegen-export/          ← PipeGen export folder (self-contained)
│   ├── run.sh               ← Single-command launcher (Docker-based)
│   ├── PipelineIR.json      ← Pipeline intermediate representation
│   ├── manifest.json        ← Export metadata
│   ├── README.md            ← Export and runtime documentation
│   ├── models/              ← ONNX + TensorRT FP32 engines
│   └── pipeline/            ← DeepStream configs, custom parsers, scripts
├── evidence/                ← Submission evidence
│   ├── pipeline_ir.png      ← Pipeline IR node graph screenshot
│   ├── tensorrt_speed_report.png ← TensorRT speed/power report (167 fps)
│   ├── live_run.png         ← Pipeline run evidence (screenshot)
│   ├── live_run.mp4         ← Pipeline run evidence (headless output video)
│   ├── sample_incident.json ← Example detection record (illustrative)
│   └── sample_incidents.jsonl ← Example incident stream
├── zone_monitor.py          ← Offline restricted-zone policy demo
├── README.md                ← This file
└── .gitignore
```

## How to Run

```bash
cd pipegen-export
./run.sh --input <path-to-video.mp4>               # live display + JSONL metadata
./run.sh --input <path-to-video.mp4> --headless     # saves output video + JSONL metadata
```

**Requirements:** NVIDIA GPU, Docker, NVIDIA Container Toolkit.
No PipeGen installation needed — the export is self-contained.

First run builds parser libraries and resolves TensorRT engines (~1 min). Subsequent runs start immediately. JSONL metadata is written under the selected output directory at `events/cam0.jsonl`; each line contains frame time/index, detection boxes, and pose skeleton shapes. Set a restricted zone in 3840×2160 frame pixels with `VMS_ZONE=x1,y1,x2,y2`; each sampled record then includes `restricted_zone_people`, and positive counts are also reported as safety alerts in the run log. Choose the rectangle to match the camera view.

## Evidence

| Evidence | File | Description |
|----------|------|-------------|
| Pipeline IR node graph | `evidence/pipeline_ir.png` | Visual graph of the 8-node pipeline |
| TensorRT speed report | `evidence/tensorrt_speed_report.png` | 167 fps, FP32, 5.993 ms/frame, 25.6 W |
| Live pipeline run | `evidence/live_run.png` | Pipeline compilation + run screenshot |
| Headless run output | `evidence/live_run.mp4` | Actual pipeline output video (11 MB) |
| Example incident records | `evidence/sample_incidents.jsonl` | Synthetic output from the separate zone-policy demo; not evidence of live alerts |

## Known Limitations & Future Work

- The live export writes detection and pose metadata to JSON Lines and can count detected people whose bottom-center point falls inside a configured restricted-zone rectangle. It reports per-sampled-frame presence counts, not persistent person tracks or one-time entry transitions. The separate `zone_monitor.py` remains an offline policy demo, and its sample incidents are not live-run evidence.
- PPE detection, MQTT alerts, INT8 quantization, and Jetson deployment are optional/bonus and not implemented.
- Built and verified on WSL2 with NVIDIA RTX 3050 (official target is native x86-64 Linux + X11).

## Built With

- [PipeGen v0.5.1](https://developer.craftifai.com) — CraftifAI's edge AI pipeline builder
- DeepStream 7.1 + TensorRT 10.3.0
- NVIDIA RTX 3050 6GB
