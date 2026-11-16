| 실험 | 가중치 | 추적기 | HOTA | IDF1 | MOTA | IDSW | Frag | FPS | 평균 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 한글 자모 test | hangul_jamo.pt | baseline | 59.22 | 66.00 | 77.62 | 67 | 140 | 18.7 | 53.6 |
| 한글 자모 test | hangul_jamo.pt | improved | 61.44 | 69.41 | 76.43 | 100 | 194 | 18.5 | 54.0 |
| 한글 자모 test | hangul_jamo.pt | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 65.82 | 76.86 | 78.53 | 44 | 146 | 18.7 | 53.6 |
| test | yolo11n.pt | baseline | 40.88 | 43.95 | 29.02 | 44 | 78 | 18.8 | 53.1 |
| test | yolo11n.pt | improved | 41.10 | 43.92 | 29.21 | 47 | 82 | 18.8 | 53.1 |
| test | yolo11n.pt | tuned | 42.88 | 47.73 | 31.81 | 65 | 105 | 18.8 | 53.3 |
| ablation | yolo11n.pt | baseline | 40.88 | 43.95 | 29.02 | 44 | 78 | 18.9 | 53.0 |
| ablation | yolo11n.pt | baseline+appearance | 41.16 | 44.34 | 29.16 | 49 | 80 | 18.8 | 53.2 |
| ablation | yolo11n.pt | baseline+recover_lost | 40.95 | 44.28 | 29.16 | 36 | 79 | 18.8 | 53.1 |
| ablation | yolo11n.pt | baseline+reid_buffer | 40.54 | 43.17 | 29.09 | 43 | 79 | 18.8 | 53.2 |
| ablation | yolo11n.pt | baseline+freeze_out_of_frame | 40.61 | 43.64 | 29.04 | 51 | 76 | 18.8 | 53.1 |
| ablation | yolo11n.pt | improved | 41.10 | 43.92 | 29.21 | 47 | 82 | 18.8 | 53.2 |
