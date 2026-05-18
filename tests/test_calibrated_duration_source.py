from __future__ import annotations

from waferagent.calibrated_cost_model import CalibratedCostModel


def test_calibrated_cost_model_uses_coefficients(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text(
        """
        {
          "schema_version": "round3.forward.1",
          "prefill_coef": [1.0, 0.5, 0.0, 2.0, 0.0],
          "decode_tpot_coef": [0.1, 0.001, 0.0, 0.01, 0.0]
        }
        """,
        encoding="utf-8",
    )
    model = CalibratedCostModel.from_json(path)
    assert model.prefill_ms(10, batch_size=4) == 14.0
    assert model.decode_ms(100, 5, batch_size=2) == 5 * (0.1 + 0.1 + 0.02)
