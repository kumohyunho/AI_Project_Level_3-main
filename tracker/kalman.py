"""등속 운동 모델 칼만 필터 (요구 사항 2: 이동 예측).

상태 벡터는 8차원 ``(cx, cy, a, h, vcx, vcy, va, vh)``입니다.
``cx, cy``는 박스 중심, ``a``는 가로/세로 비율, ``h``는 높이이며 뒤의 네 값은 각각의 속도입니다.
관측값은 앞의 네 값 ``(cx, cy, a, h)``입니다. 잡음의 크기는 박스 높이에 비례하도록 두어
가까이 있는 큰 사람과 멀리 있는 작은 사람을 같은 비율로 다룹니다.
"""
import numpy as np

_STD_POSITION = 1.0 / 20
_STD_VELOCITY = 1.0 / 160


class KalmanFilter:
    def __init__(self):
        self.motion = np.eye(8)
        for index in range(4):
            self.motion[index, 4 + index] = 1.0  # 위치 += 속도 × 1프레임
        self.observe = np.eye(4, 8)

    @staticmethod
    def initiate(measurement):
        """처음 검출된 박스로 상태와 불확실성(공분산)을 만듭니다. 속도는 0에서 시작합니다."""
        mean = np.r_[measurement, np.zeros(4)]
        height = measurement[3]
        std = [2 * _STD_POSITION * height, 2 * _STD_POSITION * height, 1e-2,
               2 * _STD_POSITION * height, 10 * _STD_VELOCITY * height,
               10 * _STD_VELOCITY * height, 1e-5, 10 * _STD_VELOCITY * height]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        """다음 프레임의 위치를 예측하고, 예측한 만큼 불확실성을 키웁니다."""
        height = mean[3]
        std = [_STD_POSITION * height, _STD_POSITION * height, 1e-2, _STD_POSITION * height,
               _STD_VELOCITY * height, _STD_VELOCITY * height, 1e-5, _STD_VELOCITY * height]
        mean = self.motion @ mean
        covariance = self.motion @ covariance @ self.motion.T + np.diag(np.square(std))
        return mean, covariance

    def project(self, mean, covariance):
        height = mean[3]
        std = [_STD_POSITION * height, _STD_POSITION * height, 1e-1, _STD_POSITION * height]
        projected = self.observe @ covariance @ self.observe.T + np.diag(np.square(std))
        return self.observe @ mean, projected

    def update(self, mean, covariance, measurement):
        """연결된 검출 박스로 예측을 보정합니다. 칼만 이득이 예측과 관측의 신뢰도를 나눕니다."""
        projected_mean, projected_cov = self.project(mean, covariance)
        gain = np.linalg.solve(projected_cov, (covariance @ self.observe.T).T).T
        innovation = measurement - projected_mean
        mean = mean + innovation @ gain.T
        covariance = covariance - gain @ projected_cov @ gain.T
        return mean, covariance
