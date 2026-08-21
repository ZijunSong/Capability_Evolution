# LOSS_NUMERICS_TEST

Command:
`PYTHONPATH=/mnt/songzijun/Capability_Evolution/SCAPE-EasyOPD:/mnt/songzijun/Capability_Evolution/SCAPE /opt/scape-easyopd-smoke7/bin/python -m pytest -q tests/methods/test_scape_component_opd_losses.py`

Result: pass as part of the 19-test SCAPE suite.

Coverage: exact forward KL, exact reverse KL, reverse-KL finite-difference gradient, alpha-JSD BF16 finite check, action CE, projected-action CE metadata (`target_source=harness_effect_projection`, `on_policy_state=true`).
