# ByteTrack 선택적 진단 기록

```python
trace = {}
tracks = tracker.update(boxes, scores, classes, features, (width, height), diagnostics=trace)
```

기존 호출은 마지막 인자를 생략하면 됩니다. 진단은 할당 비용·임계값·트랙 갱신·출력 필터를 변경하지 않습니다.
trace는 프레임마다 비우고 다시 채우므로 저장하려면 호출 직후 파일에 기록하세요.

| 필드 | 의미 |
|---|---|
| `tracker_frame` | 추적기의 내부 프레임 번호, 1부터 시작 |
| `detections` | update 입력의 원래 인덱스·좌표·신뢰도·클래스·high/low/필터 분류 |
| `associations` | 실제 연결한 detection_index·track_id·단계·현재 출력 여부 |
| `states` | 갱신 후 tracked/lost 및 해당 프레임 removed 목록, 활성 여부·좌표·상태 문구 |
| `unassigned_indices` | 입력 검출 중 연결 기록이 없는 인덱스. 임계값·면적 제한으로 제외된 검출도 포함 |

연결 단계는 high(1차), low(2차), recover(추가 복구), confirm(미확정 트랙 확인), new(새 트랙), reenter(갤러리 재진입)입니다.
new 트랙은 첫 프레임을 제외하면 확인 대기 중이라 associations에 있어도 visible=false일 수 있습니다.
이는 오검출이나 추적 실패의 확정 판정이 아닙니다.

입력 boxes는 이 추적기를 호출한 검출기의 출력입니다. NMS 전 박스나 검출기의 내부 신뢰도 계산을 복원하지 않습니다.
states의 lost 예측 박스는 실제 검출이 아닙니다. 출력에서 사라진 모든 ID가 removed 목록에 포함된다는 뜻도 아닙니다.
정답 ID와의 비교가 없으므로 이 기록만으로 ID Switches·HOTA·MOTA를 계산했다고 해석하면 안 됩니다.

실제 ByteTrack의 기본·개선 프리셋에서 기록 유무에 따른 ID·박스·공분산·lost 목록 일치와
낮은 점수 연결·재등장·미확정 트랙·제거를 테스트합니다. 이는 실제 동물 정확도 측정이 아닙니다.
