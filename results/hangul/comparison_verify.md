# 한글 자모 평가 비교: ID 복원 시 자모 확인

| 평가 | 추적기 | HOTA | IDF1 | MOTA | IDSW | 검출률 | 프레임 단위 | 다수결(같은 프레임) | 화면 표시(전체) | 트랙 단위 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val_verify_off | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 66.74 | 77.31 | 76.23 | 29 | 79.3 | 92.7 | 96.7 | 96.4 | 95.5 |
| val_verify_reentry1 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 64.64 | 73.39 | 76.17 | 37 | 79.3 | 92.7 | 96.0 | 95.7 | 94.6 |
| val_verify_reentry4 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 64.93 | 73.82 | 76.18 | 35 | 79.3 | 92.7 | 96.1 | 95.8 | 95.8 |
| val_verify_reentry8 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 65.45 | 74.63 | 76.20 | 33 | 79.3 | 92.7 | 96.3 | 96.0 | 95.7 |
| val_verify_reentry15 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 66.27 | 76.47 | 76.22 | 30 | 79.3 | 92.7 | 96.6 | 96.3 | 95.5 |
| val_verify_all1 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 60.20 | 65.91 | 75.98 | 61 | 79.3 | 92.7 | 95.2 | 94.8 | 90.1 |
| val_verify_all8 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 64.49 | 72.73 | 76.15 | 39 | 79.3 | 92.7 | 96.0 | 95.7 | 92.1 |
| test_verify_off | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 65.82 | 76.86 | 78.53 | 44 | 81.8 | 95.0 | 97.7 | 97.6 | 98.6 |
| test_verify_reentry8 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 65.12 | 75.36 | 78.49 | 49 | 81.8 | 95.0 | 98.8 | 98.6 | 97.5 |
| test_verify_reentry15 | baseline+recover_lost+reid_buffer+freeze_out_of_frame | 65.85 | 76.74 | 78.50 | 48 | 81.8 | 95.0 | 98.5 | 98.4 | 98.7 |

분류 정확도 단위는 %. 각 열의 뜻은 `docs/HANGUL.md` 6절 참고.
