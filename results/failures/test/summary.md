# 추적 실패 사례 분석

- 시퀀스: MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN
- 검출 가중치: yolo11n.pt

| 추적기 | 유형 | 교차 | 가림 | 화면 경계 | 기타 | 합계 |
|---|---|---:|---:|---:|---:|---:|
| baseline | IDSW | 21 | 9 | 2 | 12 | 44 |
| baseline | FRAG | 18 | 11 | 10 | 8 | 47 |
| improved | IDSW | 19 | 4 | 7 | 17 | 47 |
| improved | FRAG | 15 | 11 | 9 | 12 | 47 |
| tuned | IDSW | 26 | 11 | 9 | 19 | 65 |
| tuned | FRAG | 18 | 15 | 13 | 20 | 66 |

IDSW: 같은 사람의 추적 ID가 바뀐 횟수, FRAG: 추적이 끊겼다가 다시 이어진 횟수.
교차 = 다른 사람과 IoU ≥ 0.3, 가림 = visibility < 0.5, 화면 경계 = 가장자리 3% 이내.
