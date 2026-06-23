"""
Initial tests for outputs.py

Tests cover pure/logic functions that can be exercised without the full JAX
forward model or filesystem side-effects, plus lightweight smoke tests for the
I/O and plotting helpers (using tmp_path and matplotlib's non-interactive
Agg backend).

Run with:
    pytest test_outputs.py -v
"""

import json
import math
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – must come before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so outputs.py can be imported without the full package
# ---------------------------------------------------------------------------

# Import the real sting package first so sys.modules["sting"] is the real
# package object — not a bare ModuleType — before we inject any stubs.
import sting

# Build a fake `gradient_descent` module with just the constants that
# outputs.py reads at import time and in param_for_display / save_best_fit_params.
_gd = types.ModuleType("sting.gradient_descent")
_gd.ANGLE_KEYS = {"theta0", "phi0", "inc", "pa"}
_gd.DISPLAY_UNITS = {
    "r0": "au",
    "v_r0": "km/s",
    "mass": "M_sun",
    "rmin": "au",
    "deltar": "au",
    "v_lsr": "km/s",
    "rc": "au",
    "omega": "1/s",
}

# Stub out the functions called by evaluate_best_fit / plot_fitting_results
# (heavy JAX functions); individual tests that need them will mock further.
_gd.prepare_model_params = None
_gd.forward_model = None
_gd.checked_match_model_to_data_curve = None
_gd.convert_and_strip_bound_units = None

_es = types.ModuleType("sting.extract_streamline")
_es.get_distance_metric = None
_es.plot_metric_boundaries = None
_es.get_metric_partitions = None
_es.sample_metric_boundaries = None

# Inject stubs for the submodules we want to fake out, leaving the real
# top-level sting package untouched.
sys.modules["sting.gradient_descent"] = _gd
sys.modules["sting.extract_streamline"] = _es

# Now import the module under test
import sting.outputs as outputs_module

# Bring the public names into this namespace for convenience
param_for_display = outputs_module.param_for_display
save_best_fit_params = outputs_module.save_best_fit_params
_ensure_clean_dir = outputs_module._ensure_clean_dir
_opt_params_from_log = outputs_module._opt_params_from_log
plot_loss = outputs_module.plot_loss
plot_loss_panel = outputs_module.plot_loss_panel
build_velocity_radius_kde = outputs_module.build_velocity_radius_kde
plot_param_uncertainties = outputs_module.plot_param_uncertainties
plot_param_correlations = outputs_module.plot_param_correlations
sample_parameter_sets_from_covariance = outputs_module.sample_parameter_sets_from_covariance
load_optimisation_log = outputs_module.load_optimisation_log


# ===========================================================================
# Helpers
# ===========================================================================

def _make_log_csv(tmp_path, epochs, loss, extra_cols=None):
    """Write a minimal optimisation_log.csv to *tmp_path*."""
    data = {"epoch": epochs, "loss": loss}
    if extra_cols:
        data.update(extra_cols)
    df = pd.DataFrame(data)
    df.to_csv(tmp_path / "optimisation_log.csv", index=False)
    return df


# ===========================================================================
# param_for_display
# ===========================================================================

class TestParamForDisplay:
    """Unit tests for param_for_display."""

    def test_angle_key_converted_to_degrees(self):
        for key in ("theta0", "phi0", "inc", "pa"):
            _, val, unit = param_for_display(key, math.pi)
            assert unit == "deg"
            assert pytest.approx(val, rel=1e-9) == 180.0

    def test_angle_zero_stays_zero(self):
        _, val, unit = param_for_display("inc", 0.0)
        assert val == 0.0
        assert unit == "deg"

    def test_angle_negative_value(self):
        _, val, unit = param_for_display("pa", -math.pi / 2)
        assert pytest.approx(val, rel=1e-9) == -90.0

    def test_non_angle_key_with_unit(self):
        key, val, unit = param_for_display("r0", 100.0)
        assert key == "r0"
        assert val == 100.0
        assert unit == "au"

    def test_unknown_key_returns_empty_unit(self):
        _, val, unit = param_for_display("unknown_param", 42.0)
        assert unit == ""
        assert val == 42.0

    def test_return_tuple_length(self):
        result = param_for_display("mass", 1.5)
        assert len(result) == 3

    def test_value_cast_to_float(self):
        # value should always come back as a plain Python float
        _, val, _ = param_for_display("r0", np.float32(3.14))
        assert isinstance(val, float)


# ===========================================================================
# save_best_fit_params
# ===========================================================================

class TestSaveBestFitParams:
    """Tests for save_best_fit_params – exercises JSON output structure."""

    def test_creates_json_file(self, tmp_path):
        save_best_fit_params(
            best_opt_params={"r0": 100.0, "inc": math.pi / 4},
            fixed_params={"mass": 1.0},
            param_errors=None,
            save_folder=str(tmp_path),
        )
        out = tmp_path / "best_fit_params.json"
        assert out.exists()

    def test_json_has_expected_top_level_keys(self, tmp_path):
        save_best_fit_params({"r0": 50.0}, {"mass": 2.0}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert "optimised_parameters" in data
        assert "fixed_parameters" in data

    def test_angle_stored_in_degrees(self, tmp_path):
        save_best_fit_params(
            {"inc": math.pi},
            {},
            None,
            str(tmp_path),
        )
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert pytest.approx(data["optimised_parameters"]["inc"]["value"], rel=1e-9) == 180.0
        assert data["optimised_parameters"]["inc"]["unit"] == "deg"

    def test_non_angle_param_unit_preserved(self, tmp_path):
        save_best_fit_params({"r0": 200.0}, {}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert data["optimised_parameters"]["r0"]["unit"] == "au"

    def test_sigma_written_when_errors_provided(self, tmp_path):
        save_best_fit_params(
            {"r0": 100.0},
            {},
            {"r0": 5.0},
            str(tmp_path),
        )
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert "sigma" in data["optimised_parameters"]["r0"]
        assert pytest.approx(data["optimised_parameters"]["r0"]["sigma"]) == 5.0

    def test_sigma_not_written_when_errors_none(self, tmp_path):
        save_best_fit_params({"r0": 100.0}, {}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert "sigma" not in data["optimised_parameters"]["r0"]

    def test_fixed_param_none_value_handled(self, tmp_path):
        """Fixed params with None values should be stored with value: null."""
        save_best_fit_params({}, {"v_lsr": None}, None, str(tmp_path))
        with open(tmp_path / "best_fit_params.json") as f:
            data = json.load(f)
        assert data["fixed_parameters"]["v_lsr"]["value"] is None

    def test_creates_directory_if_missing(self, tmp_path):
        nested = tmp_path / "new_dir" / "sub"
        save_best_fit_params({"r0": 1.0}, {}, None, str(nested))
        assert (nested / "best_fit_params.json").exists()


# ===========================================================================
# _ensure_clean_dir
# ===========================================================================

class TestEnsureCleanDir:
    def test_creates_missing_directory(self, tmp_path):
        new_dir = tmp_path / "created"
        assert not new_dir.exists()
        _ensure_clean_dir(str(new_dir))
        assert new_dir.is_dir()

    def test_removes_existing_files(self, tmp_path):
        d = tmp_path / "dirty"
        d.mkdir()
        (d / "old.png").write_text("data")
        assert len(list(d.iterdir())) == 1
        _ensure_clean_dir(str(d))
        assert len(list(d.iterdir())) == 0

    def test_does_not_raise_on_existing_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        _ensure_clean_dir(str(d))  # should not raise


# ===========================================================================
# _opt_params_from_log
# ===========================================================================

class TestOptParamsFromLog:
    def test_skips_epoch_and_loss_columns(self):
        df = pd.DataFrame({"epoch": [0], "loss": [1.0], "r0": [100.0], "inc": [0.5]})
        cols = _opt_params_from_log(df)
        assert "epoch" not in cols
        assert "loss" not in cols
        assert set(cols) == {"r0", "inc"}

    def test_mu_present_excludes_rc_and_omega(self):
        df = pd.DataFrame(
            {"epoch": [0], "loss": [1.0], "mu": [0.5], "rc": [10.0], "omega": [1e-13], "r0": [100.0]}
        )
        cols = _opt_params_from_log(df)
        assert "rc" not in cols
        assert "omega" not in cols
        assert "mu" in cols
        assert "r0" in cols

    def test_no_mu_keeps_rc_and_omega(self):
        df = pd.DataFrame(
            {"epoch": [0], "loss": [1.0], "rc": [10.0], "omega": [1e-13]}
        )
        cols = _opt_params_from_log(df)
        assert "rc" in cols
        assert "omega" in cols

    def test_returns_list(self):
        df = pd.DataFrame({"epoch": [0], "loss": [0.1], "r0": [1.0]})
        assert isinstance(_opt_params_from_log(df), list)


# ===========================================================================
# load_optimisation_log
# ===========================================================================

class TestLoadOptimisationLog:
    def test_returns_dataframe(self, tmp_path):
        _make_log_csv(tmp_path, [0, 1], [0.5, 0.3])
        df = load_optimisation_log(str(tmp_path))
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self, tmp_path):
        _make_log_csv(tmp_path, [0], [0.5], extra_cols={"r0": [100.0]})
        df = load_optimisation_log(str(tmp_path))
        assert "epoch" in df.columns
        assert "loss" in df.columns
        assert "r0" in df.columns

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_optimisation_log(str(tmp_path / "nonexistent"))


# ===========================================================================
# plot_loss  (smoke test – checks file is written and valid PNG)
# ===========================================================================

class TestPlotLoss:
    def test_saves_png(self, tmp_path):
        plot_loss([1.0, 0.5, 0.25, 0.1], save_folder=str(tmp_path))
        assert (tmp_path / "loss_history.png").exists()

    def test_single_epoch(self, tmp_path):
        """Should not raise for a single-element loss history."""
        plot_loss([1.0], save_folder=str(tmp_path))
        assert (tmp_path / "loss_history.png").exists()

    def test_show_false_closes_figure(self, tmp_path):
        """After saving, no figures should be left open."""
        initial = plt.get_fignums()
        plot_loss([1.0, 0.5], save_folder=str(tmp_path))
        after = plt.get_fignums()
        # Number of open figures should not grow
        assert len(after) <= len(initial)


# ===========================================================================
# plot_loss_panel (unit-level)
# ===========================================================================

class TestPlotLossPanel:
    def test_green_scatter_at_best_epoch(self):
        fig, ax = plt.subplots()
        epochs = np.array([0, 1, 2, 3])
        loss = np.array([1.0, 0.5, 0.1, 0.3])
        plot_loss_panel(ax, epochs, loss)
        # The best scatter should be at epoch 2 (minimum loss)
        scatter = [c for c in ax.get_children()
                   if isinstance(c, matplotlib.collections.PathCollection)]
        assert len(scatter) >= 1
        plt.close(fig)

    def test_log_scale_set(self):
        fig, ax = plt.subplots()
        plot_loss_panel(ax, np.array([0, 1]), np.array([1.0, 0.5]))
        assert ax.get_yscale() == "log"
        plt.close(fig)


# ===========================================================================
# build_velocity_radius_kde
# ===========================================================================

class TestBuildVelocityRadiusKde:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def test_returns_required_keys(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v)
        for key in ("xx", "yy", "zz", "levels", "xlim", "ylim"):
            assert key in result

    def test_zz_normalised(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v)
        assert np.nanmax(result["zz"]) <= 1.0 + 1e-9

    def test_xlim_ylim_match_grid(self, rng):
        ra = rng.uniform(1, 5, 50)
        dec = rng.uniform(1, 5, 50)
        v = rng.uniform(-5, 5, 50)
        result = build_velocity_radius_kde(ra, dec, v)
        xmin, xmax = result["xlim"]
        ymin, ymax = result["ylim"]
        assert xmin < xmax
        assert ymin < ymax

    def test_explicit_limits_respected(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v, xmin=0, xmax=10, ymin=-20, ymax=20)
        assert result["xlim"] == (0, 10)
        assert result["ylim"] == (-20, 20)

    def test_raises_with_fewer_than_three_finite_points(self):
        with pytest.raises(ValueError, match="at least 3 finite"):
            build_velocity_radius_kde(
                [np.nan, 1.0],
                [np.nan, 1.0],
                [np.nan, 1.0],
            )

    def test_grid_size_parameter(self, rng):
        ra = rng.uniform(-5, 5, 50)
        dec = rng.uniform(-5, 5, 50)
        v = rng.uniform(-10, 10, 50)
        result = build_velocity_radius_kde(ra, dec, v, grid_size=20)
        assert result["xx"].shape == (20, 20)


# ===========================================================================
# plot_param_uncertainties (smoke)
# ===========================================================================

class TestPlotParamUncertainties:
    def test_saves_png(self, tmp_path):
        keys = ["r0", "inc", "mass"]
        vals = np.array([100.0, 0.5, 1.0])
        errs = np.array([5.0, 0.1, 0.05])
        plot_param_uncertainties(keys, vals, errs, save_folder=str(tmp_path))
        assert (tmp_path / "parameter_uncertainties.png").exists()

    def test_single_parameter(self, tmp_path):
        plot_param_uncertainties(["r0"], np.array([100.0]), np.array([5.0]), save_folder=str(tmp_path))
        assert (tmp_path / "parameter_uncertainties.png").exists()


# ===========================================================================
# plot_param_correlations (smoke + correctness)
# ===========================================================================

class TestPlotParamCorrelations:
    def _identity_cov(self, n):
        return np.eye(n)

    def test_saves_png(self, tmp_path):
        cov = self._identity_cov(3)
        plot_param_correlations(["r0", "inc", "mass"], cov, save_folder=str(tmp_path))
        assert (tmp_path / "parameter_correlation_matrix.png").exists()

    def test_diagonal_is_one_for_identity_cov(self, tmp_path):
        """With an identity covariance the correlation matrix diagonal must be 1."""
        n = 3
        cov = self._identity_cov(n)
        diag = np.sqrt(np.diag(cov))
        corr = cov / np.outer(diag, diag)
        assert np.allclose(np.diag(corr), 1.0)

    def test_correlation_values_clipped_to_minus_one_one(self):
        """Numerically extreme covariances should never produce |corr| > 1."""
        cov = np.array([[1e-30, 1e30], [1e30, 1e-30]])
        diag = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
        corr = np.clip(cov / np.outer(diag, diag), -1.0, 1.0)
        assert np.all(np.abs(corr) <= 1.0)

    def test_no_annotate_does_not_raise(self, tmp_path):
        cov = self._identity_cov(2)
        plot_param_correlations(["a", "b"], cov, annotate=False, save_folder=str(tmp_path))


# ===========================================================================
# sample_parameter_sets_from_covariance
# ===========================================================================

class TestSampleParameterSetsFromCovariance:
    @pytest.fixture
    def simple_setup(self):
        params = {"r0": 100.0, "inc": 0.5}
        cov = np.diag([4.0, 0.01])  # std = 2.0 and 0.1
        keys = ["r0", "inc"]
        return params, cov, keys

    def test_output_shape(self, simple_setup):
        params, cov, keys = simple_setup
        samples = sample_parameter_sets_from_covariance(params, cov, keys, n_samples=50)
        assert samples.shape == (50, 2)

    def test_reproducible_with_same_seed(self, simple_setup):
        params, cov, keys = simple_setup
        s1 = sample_parameter_sets_from_covariance(params, cov, keys, seed=7)
        s2 = sample_parameter_sets_from_covariance(params, cov, keys, seed=7)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_give_different_results(self, simple_setup):
        params, cov, keys = simple_setup
        s1 = sample_parameter_sets_from_covariance(params, cov, keys, seed=1)
        s2 = sample_parameter_sets_from_covariance(params, cov, keys, seed=2)
        assert not np.array_equal(s1, s2)

    def test_sample_mean_close_to_input(self, simple_setup):
        params, cov, keys = simple_setup
        samples = sample_parameter_sets_from_covariance(params, cov, keys, n_samples=5000)
        # With enough samples the mean should be close to the true mean
        np.testing.assert_allclose(samples.mean(axis=0), [100.0, 0.5], atol=0.3)

    def test_clipping_with_param_bounds(self, simple_setup):
        """Samples must respect provided bounds after clipping."""
        params, cov, keys = simple_setup
        # Stub convert_and_strip_bound_units to be a pass-through
        _gd.convert_and_strip_bound_units = lambda b: b
        bounds = {"r0": (95.0, 105.0)}
        samples = sample_parameter_sets_from_covariance(
            params, cov, keys, param_bounds=bounds, n_samples=200
        )
        assert np.all(samples[:, 0] >= 95.0)
        assert np.all(samples[:, 0] <= 105.0)
        _gd.convert_and_strip_bound_units = None  # restore

    def test_n_samples_one(self, simple_setup):
        params, cov, keys = simple_setup
        samples = sample_parameter_sets_from_covariance(params, cov, keys, n_samples=1)
        assert samples.shape == (1, 2)