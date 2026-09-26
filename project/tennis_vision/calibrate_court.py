"""Four-click court calibration for a single, fixed tennis-camera video."""
import argparse
from pathlib import Path

import cv2
import yaml

NAMES = ("near-left doubles corner", "near-right doubles corner", "far-right doubles corner", "far-left doubles corner")


def frame_at(video: Path, frame_number: int):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_number} from {video}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("court.yaml"))
    parser.add_argument("--frame", type=int, default=0, help="Choose a clear frame with all four court corners visible")
    args = parser.parse_args()
    source = frame_at(args.input, args.frame)
    points: list[list[int]] = []
    window = "Tennis Vision — court calibration"

    def redraw():
        image = source.copy()
        for index, (x, y) in enumerate(points):
            cv2.circle(image, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(image, str(index + 1), (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        instruction = "Done: press Enter" if len(points) == 4 else f"Click {len(points) + 1}/4: {NAMES[len(points)]}"
        cv2.putText(image, instruction, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(image, "u=undo  Esc=cancel", (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, image)

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, click)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled.")
        if key == ord("u") and points:
            points.pop(); redraw()
        if key in (10, 13) and len(points) == 4:
            break
    cv2.destroyAllWindows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump({"image_corners": points, "calibration_frame": args.frame}, sort_keys=False), encoding="utf-8")
    print(f"Saved {args.output}. Re-run analysis with --court {args.output}")


if __name__ == "__main__":
    main()
