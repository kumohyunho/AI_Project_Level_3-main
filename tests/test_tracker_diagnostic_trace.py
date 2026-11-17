"""기록 유무에 따른 실제 ByteTrack 동작 동일성과 실제 연결 인덱스를 검증합니다."""
import json
import unittest

import numpy as np

from tracker import ByteTracker, TrackerConfig


class TrackerTraceTest(unittest.TestCase):
    def update(self, tracker, boxes, scores, trace=None):
        return tracker.update(np.asarray(boxes).reshape(-1, 4), scores, [0]*len(boxes), diagnostics=trace)

    def test_enabled_and_disabled_produce_same_states_boxes_and_ids(self):
        for preset in ("baseline", "improved"):
            before = ByteTracker(TrackerConfig.preset(preset))
            traced = ByteTracker(TrackerConfig.preset(preset))
            for score in (0.9, 0.15, None, None, 0.9, 0.9):
                boxes = [[10., 10, 50, 60]] if score is not None else []
                scores = [score] if score is not None else []
                record = {}
                a = self.update(before, boxes, scores)
                b = self.update(traced, boxes, scores, record)
                self.assertEqual([t.track_id for t in a], [t.track_id for t in b])
                for x, y in zip(a, b):
                    np.testing.assert_array_equal(x.box, y.box)
                    np.testing.assert_array_equal(x.covariance, y.covariance)
                self.assertEqual([t.track_id for t in before.lost], [t.track_id for t in traced.lost])
                json.dumps(record, allow_nan=False)

    def test_new_low_high_lost_and_returned_associations_are_exact(self):
        tracker = ByteTracker(TrackerConfig.preset("baseline"))
        trace = {}
        self.update(tracker, [[10,10,50,60]], [0.9], trace)
        self.assertEqual(trace["associations"], [dict(detection_index=0, track_id=1, stage="new", visible=True)])
        self.update(tracker, [[10,10,50,60]], [0.15], trace)
        self.assertEqual(trace["detections"][0]["group"], "low")
        self.assertEqual(trace["associations"][0]["stage"], "low")
        self.update(tracker, [], [], trace)
        self.assertEqual(trace["states"][0]["state"], "lost")
        self.update(tracker, [[10,10,50,60]], [0.9], trace)
        self.assertEqual(trace["associations"][0]["track_id"], 1)
        self.assertEqual(trace["associations"][0]["stage"], "high")
        self.assertEqual(trace["states"][0]["status"], "재등장")

    def test_original_indices_survive_score_groups_and_unconfirmed_tracks(self):
        tracker = ByteTracker(TrackerConfig.preset("baseline"))
        trace = {}
        self.update(tracker, [], [], trace)
        boxes = [[0,0,1,1], [10,10,50,60], [70,10,100,60], [120,10,150,60]]
        self.update(tracker, boxes, [0.95, 0.9, 0.2, 0.05], trace)
        self.assertEqual(trace["associations"], [dict(detection_index=1, track_id=1, stage="new", visible=False)])
        self.assertEqual(trace["unassigned_indices"], [0, 2, 3])
        self.assertEqual([d["group"] for d in trace["detections"]], ["filtered_area", "high", "low", "filtered_confidence"])
        self.update(tracker, [[10,10,50,60]], [0.9], trace)
        self.assertEqual(trace["associations"][0]["stage"], "confirm")
        self.assertTrue(trace["associations"][0]["visible"])

    def test_removed_state_and_empty_frame_are_not_invented_detections(self):
        config = TrackerConfig.preset("baseline")
        config.track_buffer = 1
        tracker = ByteTracker(config)
        trace = {}
        self.update(tracker, [[10,10,50,60]], [0.9], trace)
        self.update(tracker, [], [], trace)
        self.update(tracker, [], [], trace)
        self.assertEqual(trace["detections"], [])
        self.assertEqual(trace["associations"], [])
        self.assertEqual(trace["states"][0]["state"], "removed")
