| 실험 | 가중치 | 추적기 | HOTA | IDF1 | MOTA | IDSW | Frag | FPS | 평균 ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 한글 자모 test | hangul_jamo.pt | baseline | 59.22 | 66.00 | 77.62 | 67 | 140 | 18.7 | 53.6 |
| 한글 자모 test | hangul_jamo.pt | improved | 61.44 | 69.41 | 76.43 | 100 | 194 | 18.5 | 54.0 |
| 한글 자모 test | hangul_jamo.pt | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 65.82 | 76.86 | 78.53 | 44 | 146 | 18.7 | 53.6 |
