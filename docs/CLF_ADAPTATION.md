# CLF adaptation of the authentic mCEL implementation

Source baseline: `GitMunish1/Clf-main`, master commit
`ea17e54fdfad2cefe5c861f2b81e91a7a33c5688`.
Development branch: `feature/clf-mcel-clf-adaptation-v1`.

Retain the actual single-input Wi-Fi mCEL network, overlapping cell labels,
classification head and cell-offset regression head. The CLF profile uses 24
approved AP features and the original 512-unit backbone/head arrangement. Motion
is optional external tracking evidence, not an invented second neural input.
There is no BLE input. No model was trained in this change.

The complete architecture review, mathematical encoding/decoding specification,
backend migration plan, limitations, measured CPU smoke-test latency and test
report are maintained in the paired backend branch:

[CLF two-repository architecture audit](https://github.com/GitMunish1/clf-backend/blob/feature/clf-mcel-wifi24-integration-v1/docs/audits/mcel-two-repository-architecture-2026-10-25.md)

## Source layout

- `model/model_definition.py`: authentic builder; corrected scalar Input shape.
- `model/mcel_model.py`: original equal-weight cross entropy + masked regression
  training objective; no training executed here.
- `data/mcel_data_provider.py`: original overlapping label and coordinate encoding;
  corrected singular inverse-distance weights.
- `config/clf/wifi24.yml`: explicit untrained CLF network profile.
- `data/clf_data_provider.py`: CLF adapter for raw ordered RSSI, measured XY/floor
  labels and explicit disjoint session partitions. It bypasses research dataset
  connectors and uses fixed `(clip(RSSI,-110,-20)+110)/90` scaling.
- `model/clf_artifact.py`: build network; create feature/grid metadata; write a new
  complete package explicitly marked untrained. Never overwrites an existing path.
- `scripts/verify_backend_contract.py`: isolated actual Keras/backend loader check.
- `tests/test_clf_mcel.py`: forward/backward, loss, serialization, encoding,
  leakage, session split and checkpoint-path checks.

The historical GIA connector remains a research reproduction path with known
normalization/split limitations documented in the audit. Do not use it to import
CLF fingerprints. Existing research `.hdf5` checkpoint paths are still accepted
when loading; new saves use Keras 3's required `.weights.h5` suffix.

## Inputs that must be supplied before later training

1. The exact 24 approved BSSIDs in canonical lowercase colon form and immutable
   order. `make_feature_contract(ordered_bssids)` validates them; no real CLF
   inventory is hard-coded or published by this change.
2. Raw RSSI matrix `(N,24)` in that order, missing=-110 dBm. Do not pass already
   normalized RSSI into the raw-data provider.
3. Measured local-metre positions `(N,2)` and floors `(N,)`, Ground=0, First=1.
   Node IDs require a surveyed coordinate lookup, not numeric-ID arithmetic.
4. Session IDs `(N,)` and explicit nonempty, disjoint, exhaustive train/val/test
   session lists. Design held-out-device and time-based evaluation separately.
5. Measured width/height extents for both floors in the same coordinate frame as
   the labels. Choose grid spacing and overlap deliberately; the tests' 3 m/.3
   values are synthetic fixtures, not an approved CLF field configuration.
6. Measured map-to-local-metre affine calibration per floor or an explicitly
   justified shared transform.

Build the provider with `CLFdataProvider(params, rss_dbm, xy_m, floors,
session_ids, partitions, floor_dimensions, contract)`. Its parameters include
`grid_size`, `padding_ratio`, and optional `weighted_grid_labels` (false default).
`floor_dimensions` maps integer floors 0 and 1 to `(width_m,height_m)`.
`partitions` maps `train`, `val`, `test` to session ID lists. Both classification
and regression padded labels are enabled for this CLF adapter.

`build_clf_model(C)` constructs the untrained network. Alternatively the original
`model_factory.get_model(profile, base_dir, provider)` compiles its authentic
training wrapper. Compiling and calculating gradients are not training; invoking
`fit_model()` later is a separate data-dependent task and was not done here.

Use `make_grid_metadata(...)` with the exact extents/grid parameters used by the
provider. The ordered cells and their origins must match provider output indexes.
`write_untrained_package(model, new_directory, contract, grid, calibration)` saves
an untrained package for inspection. Its status cannot be used to localize users:
the backend rejects it. A genuine future training/export step must supply trained
weights, real training metadata, matching artifacts/hashes and calibration. Never
set a random model's metadata to trained for live use.

## Version and output contract

- Profile `clf-mcel-wifi24-v1`; family `mcel-mlp-v1`.
- Features `clf-features-v4-wifi24`; one tensor `input` with shape `(batch,24)`.
- `output_class`: `(batch,C)` softmax cell probabilities.
- `output_reg`: `(batch,2*C)` tanh XY offsets; scale `grid_size*(.5+padding)`.
- A package binds source provenance, adapted builder digest, AP order, encoding,
  calibration, preprocessing, model and hashes. Changed AP order/dimensions or
  changed cell registry require an explicit compatible rebuild/retraining plan.

The backend decodes Top-3 grid cells, each to its own continuous-coordinate
hypothesis. These are not kNN neighbors or routing-node classes. It returns
`candidates_only` and no accepted location until a temporal graph coordinator is
implemented and evaluated. Routing vertices/entrances are separate from grid cells.

## Verification commands

```bash
python -m pip install -r requirements-verification.txt
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 python -m pytest -q tests
TF_CPP_MIN_LOG_LEVEL=3 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 python scripts/verify_backend_contract.py /path/to/clf-backend
```

Use equivalent environment-variable settings on Windows. These commands were
executed on Linux/Python 3.12.14 with TensorFlow CPU 2.20.0 and Keras 3.15.1.
11 model tests passed. The paired backend had 46 passing tests. Actual `.keras`
serialization and loader interoperability passed.

The cross-repository check first verifies that the untrained export is rejected,
then uses an explicitly synthetic trained-status declaration inside a temporary
software fixture to exercise actual loading. The weights remain untrained, the
script does not start FastAPI, and no user location is accepted. This is not a
trained model, field test or accuracy result. No generated model weights are
committed to Git.
