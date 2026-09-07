"""Unit tests for ROS-independent perception processing."""

import cv2
import numpy as np
import pytest

from ugv_navigation.perception import process_bgr_image


def test_uniform_ground_is_mostly_traversable():
    """A uniform lower ROI should produce no obstacle proposals."""
    frame = np.full((120, 200, 3), 120, dtype=np.uint8)
    result = process_bgr_image(frame)

    assert result.obstacles == []
    assert result.traversability_mask.shape == frame.shape[:2]
    assert np.count_nonzero(result.traversability_mask[54:, :]) > 0.95 * 66 * 200
    assert np.count_nonzero(result.traversability_mask[:54, :]) == 0


def test_coloured_roi_object_becomes_obstacle_proposal():
    """A large saturated region in the ground ROI is detected."""
    frame = np.full((150, 240, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame, (80, 85), (145, 135), (0, 0, 255), thickness=cv2.FILLED)
    result = process_bgr_image(frame)

    assert result.obstacles
    assert any(x <= 80 and y <= 85 and x + w >= 145 and y + h >= 135
               for x, y, w, h in result.obstacles)
    assert result.obstacle_mask[110, 110] == 255
    assert result.traversability_mask[110, 110] == 0
    assert result.debug_image.shape == frame.shape


@pytest.mark.parametrize('shape', [(64, 96), (240, 320), (480, 640)])
def test_processing_is_resolution_independent(shape):
    """Different valid image resolutions yield correctly sized outputs."""
    height, width = shape
    frame = np.full((height, width, 3), 100, dtype=np.uint8)
    result = process_bgr_image(frame)

    assert result.traversability_mask.shape == shape
    assert result.obstacle_mask.shape == shape


def test_invalid_input_is_rejected():
    """The processor reports malformed inputs clearly."""
    with pytest.raises(ValueError):
        process_bgr_image(np.zeros((20, 20), dtype=np.uint8))
