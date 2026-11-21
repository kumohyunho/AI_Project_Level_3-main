"""compare_camouflage_coco.py의 묶음 검출 통계 테스트 (모델 없이)."""
import tempfile
import unittest
from pathlib import Path

from compare_camouflage_coco import grouping_stats, load_mot


class GroupingStatsTest(unittest.TestCase):
    def test_one_box_holding_two_animals_counts_as_grouped(self):
        # 정답: 프레임마다 동물 두 마리 (x 0~10, 20~30). 예측: 두 마리를 감싼 큰 박스 하나
        truth = {1: [(0, 0, 10, 10), (20, 0, 30, 10)], 2: [(0, 0, 10, 10), (20, 0, 30, 10)]}
        predicted = {1: [(0, 0, 30, 10)], 2: [(0, 0, 10, 10)]}
        stats = grouping_stats(truth, predicted, [1, 2])
        self.assertEqual(stats["truth_per_frame"], 2.0)
        self.assertEqual(stats["predicted_per_frame"], 1.0)
        self.assertEqual((stats["grouped"], stats["predicted"]), (1, 2))
        self.assertAlmostEqual(stats["area_ratio"], (300 + 100) / 2 / 100)

    def test_no_predictions(self):
        stats = grouping_stats({1: [(0, 0, 10, 10)]}, {}, [1])
        self.assertEqual((stats["predicted"], stats["grouped_ratio"], stats["area_ratio"]), (0, 0.0, 0.0))

    def test_load_mot_converts_left_top_width_height(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.txt"
            path.write_text("3,1,10.0,20.0,5.0,8.0,0.9,-1,-1,-1\n", encoding="utf-8")
            self.assertEqual(load_mot(path)[3], [(10.0, 20.0, 15.0, 28.0)])


if __name__ == "__main__":
    unittest.main()
