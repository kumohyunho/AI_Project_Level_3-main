"""검증 결과의 NMS만 GUI·영상 공통 설정에 설치합니다. 추적 신뢰도는 변경하지 않습니다."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from extensions.camouflage import CLASS_NAME, validate_nms


def configure(args):
    weights = args.weights.resolve()
    target = weights.with_suffix(".inference.json")
    if target.exists():
        raise FileExistsError(f"기존 실행 설정을 먼저 다른 이름으로 보관하세요: {target}")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("extension") != "camouflage" or metadata.get("classes") != [CLASS_NAME]:
        raise ValueError("보호색 모델과 학습 설정 JSON이 필요합니다.")
    if metadata.get("weights_sha256", digest) != digest:
        raise ValueError("학습 설정과 모델 가중치가 다릅니다.")
    provenance = dict(source="수동 지정", tracking_evaluation="미실행")
    nms = args.nms_iou
    if args.report:
        payload = args.report.read_bytes()
        report = json.loads(payload)
        if report.get("split") != "val" or report.get("weights_sha256") != digest:
            raise ValueError("같은 모델로 실행한 val 비교 보고서가 필요합니다.")
        if report.get("imgsz") != 640 or report.get("max_det") != 300:
            raise ValueError("현재 GUI와 같은 imgsz=640, max_det=300 보고서가 필요합니다.")
        best = report["best"]
        if best not in report["candidates"]:
            raise ValueError("보고서의 선택 후보가 후보 목록에 없습니다.")
        nms = best["nms_iou"]
        provenance.update(source="val 검출 F1 선택 후보의 NMS만 적용",
                          report_sha256=hashlib.sha256(payload).hexdigest(),
                          detection_best=best, selected_conf_applied=False)
    data = dict(extension="camouflage", weights_sha256=digest, nms_iou=validate_nms(nms),
                max_det=300, **provenance)
    with TemporaryDirectory(dir=weights.parent) as folder:
        staged = Path(folder) / target.name
        staged.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        staged.rename(target)
    print(f"NMS 설정 저장: {target}\nGUI를 완전히 종료한 뒤 다시 실행하세요.")
    print("추적 low/high/new 임계값은 유지합니다. 실제 추적 정확도 개선은 미검증입니다.")
    return data


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=Path("models/camouflage.pt"))
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path, help="#13 summary.json, 같은 모델의 val 결과")
    source.add_argument("--nms-iou", type=float, help="보고서가 없을 때 사용자가 명시할 NMS IoU")
    return p


if __name__ == "__main__":
    configure(parser().parse_args())
