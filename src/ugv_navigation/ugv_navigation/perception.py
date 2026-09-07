"""
Resolution-independent, classical image perception helpers.

The functions in this module deliberately do not depend on ROS.  They return
image-space proposals, which lets a learned model later replace this baseline
while preserving the node's ROS-facing interfaces.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


BoundingBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class PerceptionResult:
    """Image-space output from the deterministic traversability baseline."""

    traversability_mask: np.ndarray
    obstacle_mask: np.ndarray
    obstacles: List[BoundingBox]
    debug_image: np.ndarray


def _odd_kernel_size(height: int, width: int) -> int:
    """Return a small odd morphology kernel scaled to the input image."""
    return max(3, int(min(height, width) * 0.02) | 1)


def process_bgr_image(frame: np.ndarray) -> PerceptionResult:
    """
    Estimate traversable pixels and obstacle proposals from a BGR image.

    The lower 55 percent of the frame is the candidate ground region.  Within
    that ROI, strong brightness departures, saturated colour, and local edges
    are treated as obstacle evidence.  Morphology joins nearby evidence before
    contours are converted into bounding boxes.  This is intentionally a
    simple image-space heuristic, not semantic terrain understanding.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected a non-empty BGR image with three channels')
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ValueError('Expected an image at least 2 by 2 pixels')

    height, width = frame.shape[:2]
    roi_top = int(height * 0.45)
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[roi_top:, :] = 255

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[roi_top:, :]
    median_brightness = float(np.median(roi_gray))
    brightness_delta = max(25.0, median_brightness * 0.18)

    brightness_outlier = cv2.inRange(
        gray,
        max(0, int(median_brightness - brightness_delta)),
        min(255, int(median_brightness + brightness_delta)),
    )
    brightness_outlier = cv2.bitwise_not(brightness_outlier)
    saturation_outlier = cv2.inRange(hsv[:, :, 1], 105, 255)
    low = max(20, int(median_brightness * 0.66))
    high = max(low + 20, int(median_brightness * 1.33))
    edges = cv2.Canny(gray, low, min(255, high))

    kernel_size = _odd_kernel_size(height, width)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    edge_evidence = cv2.dilate(edges, kernel, iterations=1)
    evidence = cv2.bitwise_or(brightness_outlier, saturation_outlier)
    evidence = cv2.bitwise_or(evidence, edge_evidence)
    evidence = cv2.bitwise_and(evidence, roi)
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, kernel, iterations=2)
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_OPEN, kernel, iterations=1)

    min_area = max(40, int((height - roi_top) * width * 0.003))
    contours, _ = cv2.findContours(
        evidence, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_mask = np.zeros_like(roi)
    obstacles: List[BoundingBox] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        cv2.drawContours(obstacle_mask, [contour], -1, 255, thickness=cv2.FILLED)
        obstacles.append((x, y, box_width, box_height))

    obstacles.sort(key=lambda box: box[0])
    traversability_mask = cv2.bitwise_and(roi, cv2.bitwise_not(obstacle_mask))
    debug_image = _make_debug_image(frame, traversability_mask, obstacles, roi_top)
    return PerceptionResult(
        traversability_mask=traversability_mask,
        obstacle_mask=obstacle_mask,
        obstacles=obstacles,
        debug_image=debug_image,
    )


def _make_debug_image(
    frame: np.ndarray,
    traversability_mask: np.ndarray,
    obstacles: List[BoundingBox],
    roi_top: int,
) -> np.ndarray:
    """Overlay green traversable and red obstacle proposals on the source."""
    debug = frame.copy()
    traversable = traversability_mask > 0
    green = np.zeros_like(debug)
    green[:, :] = (0, 180, 0)
    debug[traversable] = cv2.addWeighted(
        debug[traversable], 0.55, green[traversable], 0.45, 0)
    cv2.line(debug, (0, roi_top), (debug.shape[1] - 1, roi_top), (255, 255, 0), 1)
    for x, y, width, height in obstacles:
        cv2.rectangle(debug, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(debug, 'obstacle', (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return debug
