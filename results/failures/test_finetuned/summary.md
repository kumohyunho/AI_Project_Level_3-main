# 추적 실패 사례 분석

- 시퀀스: MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-10-FRCNN, MOT17-11-FRCNN, MOT17-13-FRCNN
- 검출 가중치: person_mot17.pt

| 추적기 | 유형 | 교차 | 가림 | 화면 경계 | 기타 | 합계 |
|---|---|---:|---:|---:|---:|---:|
| baseline | IDSW | 36 | 14 | 3 | 29 | 82 |
| baseline | FRAG | 31 | 19 | 15 | 23 | 88 |
| improved | IDSW | 58 | 18 | 10 | 29 | 115 |
| improved | FRAG | 45 | 25 | 17 | 35 | 122 |
| tuned_person | IDSW | 53 | 21 | 13 | 26 | 113 |
| tuned_person | FRAG | 36 | 23 | 17 | 34 | 110 |

IDSW: 같은 사람의 추적 ID가 바뀐 횟수, FRAG: 추적이 끊겼다가 다시 이어진 횟수.
교차 = 다른 사람과 IoU ≥ 0.3, 가림 = visibility < 0.5, 화면 경계 = 가장자리 3% 이내.
