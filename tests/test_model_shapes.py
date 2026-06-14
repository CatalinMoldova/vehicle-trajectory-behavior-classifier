import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from vehicle_behavior.model import build_cnn_lstm_model, build_lstm_model


@pytest.mark.parametrize("builder", [build_cnn_lstm_model, build_lstm_model])
def test_model_output_shape(builder):
    model = builder(sequence_length=60, num_classes=3)
    x_speed = np.zeros((4, 60, 1), dtype=np.float32)
    x_pos = np.zeros((4, 60, 2), dtype=np.float32)
    out = model.predict([x_speed, x_pos], verbose=0)
    assert out.shape == (4, 3)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-5)
