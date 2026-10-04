import unittest
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import partialmethod
from itertools import product
from types import SimpleNamespace
import numpy as np
from src.datasets import non_eeg
from src.portable import portable_features


class PortableTests(unittest.TestCase):

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(TemporaryDirectory()))

    def test_portable_features_keys_and_finite(self):
        rng = np.random.RandomState(0)
        feats = portable_features(
            eda=rng.rand(240) + 5,
            temp=rng.rand(240) + 33,
            acc_mag=rng.rand(1920) + 1,
            hr=rng.rand(70) * 10 + 70,
            eda_fs=4,
            temp_fs=4,
        )
        self.assertEqual(len(feats), 18)
        self.assertTrue(any((k.startswith("EDA_") for k in feats)))
        self.assertTrue(any((k.startswith("HR_") for k in feats)))
        self.assertTrue(np.isfinite(feats["EDA_mean"]) and np.isfinite(feats["HR_std"]))

    def test_portable_features_handle_empty(self):
        feats = portable_features(
            eda=np.array([]),
            temp=np.array([1.0]),
            acc_mag=np.array([]),
            hr=np.array([]),
            eda_fs=4,
            temp_fs=4,
        )
        self.assertEqual(len(feats), 18)
        self.assertTrue(np.isnan(feats["EDA_mean"]))

    def test_same_columns_both_datasets(self):
        rng = np.random.RandomState(1)
        a = portable_features(
            rng.rand(240),
            rng.rand(240),
            rng.rand(1920),
            rng.rand(70),
            eda_fs=4,
            temp_fs=4,
        )
        b = portable_features(
            rng.rand(480),
            rng.rand(480),
            rng.rand(480),
            rng.rand(60),
            eda_fs=8,
            temp_fs=8,
        )
        self.assertEqual(set(a), set(b))

    def _case_slopes_match_physical_units_across_sampling_rates(self, missing):
        for fs in (4, 8):
            t = np.arange(60 * fs) / fs
            eda = 3.0 + 0.25 * t
            temp = 35.0 - 0.01 * t
            if missing:
                eda[[1, 7, 20, 90]] = np.nan
                temp[[4, 8, 35]] = np.inf
            features = portable_features(eda, temp, [], [], eda_fs=fs, temp_fs=fs)
            np.testing.assert_allclose(
                features["EDA_slope"], 0.25, rtol=1e-06, atol=1e-12
            )
            np.testing.assert_allclose(
                features["TEMP_slope"], -0.01, rtol=1e-06, atol=1e-12
            )

    def _case_invalid_sampling_rates_are_rejected(self, rate, channel):
        rates = {"eda_fs": 4, "temp_fs": 4, channel: rate}
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            portable_features([1, 2], [35, 36], [], [], **rates)

    def test_sampling_rates_must_be_explicit(self):
        with self.assertRaisesRegex(TypeError, "eda_fs"):
            portable_features([1, 2], [35, 36], [], [])

    def _case_noneeg_windows_align_annotations_sensor_channels_and_hr(
        self, sensor_offset
    ):
        tmp_path = self.tmp_path
        self.stack.enter_context(patch.object(non_eeg, "DATA_DIR", tmp_path))
        (tmp_path / "Subject1_AccTempEDA.hea").touch()
        sensor_time = np.arange(180 * 8 + sensor_offset) / 8
        sensor = np.column_stack(
            [
                np.zeros((len(sensor_time), 2)),
                np.ones(len(sensor_time)),
                35 - 0.01 * sensor_time,
                3 + 0.25 * sensor_time,
            ]
        )
        hr = np.column_stack([np.zeros(181), 70 + np.arange(181)])
        annotations = SimpleNamespace(
            sample=np.array([0, 480, 960]) + sensor_offset,
            aux_note=["Relax", "CognitiveStress", "PhysicalStress"],
        )

        def read(record):
            return (
                SimpleNamespace(fs=1, p_signal=hr)
                if record.endswith("SpO2HR")
                else SimpleNamespace(fs=8, p_signal=sensor)
            )

        self.stack.enter_context(patch.object(non_eeg, "read_record", read))
        self.stack.enter_context(
            patch.object(non_eeg, "read_annotations", lambda *args: annotations)
        )
        frame = non_eeg.build(["Subject1"])
        self.assertEqual(frame["label"].tolist(), [0, 1])
        np.testing.assert_allclose(frame["EDA_slope"], 0.25)
        np.testing.assert_allclose(frame["TEMP_slope"], -0.01)
        first_hr = int(sensor_offset > 0)
        np.testing.assert_allclose(
            frame["HR_mean"], [99.5 + first_hr, 159.5 + first_hr]
        )
        np.testing.assert_allclose(frame["HR_min"], [70 + first_hr, 130 + first_hr])
        np.testing.assert_allclose(frame["HR_max"], [129 + first_hr, 189 + first_hr])

    def test_noneeg_rejects_annotations_beyond_available_samples(self):
        tmp_path = self.tmp_path
        self.stack.enter_context(patch.object(non_eeg, "DATA_DIR", tmp_path))
        (tmp_path / "Subject1_AccTempEDA.hea").touch()
        annotations = SimpleNamespace(
            sample=np.array([0, 900]), aux_note=["Relax", "CognitiveStress"]
        )
        self.stack.enter_context(
            patch.object(non_eeg, "read_annotations", lambda *args: annotations)
        )
        self.stack.enter_context(
            patch.object(
                non_eeg,
                "read_record",
                lambda record: SimpleNamespace(
                    fs=1 if record.endswith("SpO2HR") else 8,
                    p_signal=np.ones(
                        (
                            60 if record.endswith("SpO2HR") else 480,
                            2 if record.endswith("SpO2HR") else 5,
                        )
                    ),
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "outside the sensor record"):
            non_eeg.build(["Subject1"])


for index, missing in enumerate([False, True]):
    setattr(
        PortableTests,
        f"test_slopes_match_physical_units_across_sampling_rates_{index}",
        partialmethod(
            PortableTests._case_slopes_match_physical_units_across_sampling_rates,
            missing=missing,
        ),
    )
for index, (rate, channel) in enumerate(
    product(
        [0, -1, np.nan, np.inf, True, np.bool_(True), None, "invalid"],
        ["eda_fs", "temp_fs"],
    )
):
    setattr(
        PortableTests,
        f"test_invalid_sampling_rates_are_rejected_{index}",
        partialmethod(
            PortableTests._case_invalid_sampling_rates_are_rejected,
            rate=rate,
            channel=channel,
        ),
    )
for index, sensor_offset in enumerate([0, 1, 7]):
    setattr(
        PortableTests,
        f"test_noneeg_windows_align_annotations_sensor_channels_and_hr_{index}",
        partialmethod(
            PortableTests._case_noneeg_windows_align_annotations_sensor_channels_and_hr,
            sensor_offset=sensor_offset,
        ),
    )
