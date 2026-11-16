# 한글 자모 평가 비교

| 평가 | 추적기 | HOTA | IDF1 | MOTA | IDSW | 검출률 | 프레임 단위 | 다수결(같은 프레임) | 화면 표시(전체) | 트랙 단위 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rot0 | baseline | 57.57 | 60.19 | 75.34 | 78 | 79.0 | 90.0 | 92.3 | 90.7 | 94.0 |
| rot0 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 63.16 | 69.35 | 76.34 | 58 | 79.7 | 89.7 | 92.0 | 90.5 | 94.9 |
| rot45 | baseline | 55.99 | 59.47 | 77.13 | 72 | 80.8 | 89.5 | 93.2 | 92.4 | 94.8 |
| rot45 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 64.12 | 71.22 | 77.90 | 55 | 81.4 | 89.3 | 93.4 | 92.7 | 95.9 |
| rot90 | baseline | 54.60 | 58.45 | 74.04 | 62 | 78.0 | 77.0 | 83.7 | 82.1 | 83.0 |
| rot90 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 61.09 | 67.91 | 74.63 | 53 | 78.6 | 76.8 | 83.8 | 82.4 | 81.7 |

분류 정확도 단위는 %. 각 열의 뜻은 `docs/HANGUL.md` 6절 참고.
