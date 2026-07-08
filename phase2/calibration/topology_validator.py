"""
Validates the network topology:
  - all required connections present
  - sign constraints respected
  - sparsity within biological range
  - weight magnitudes reasonable
"""

REQUIRED_CONNECTIONS = [
    "ctx_d1", "ctx_d2", "ctx_stn",
    "d1_gpi", "d2_gpe",
    "gpe_stn", "gpe_gpi",
    "stn_gpi", "gpi_thal", "thal_ctx",
    "snc_d1", "snc_d2",
    "sero_gpi", "ne_stn",
    "bayes_striatum", "bayes_gpi",
]

SIGN_RULES = {
    "ctx_d1": +1, "ctx_d2": +1, "ctx_stn": +1,
    "d1_gpi": -1, "d2_gpe": -1,
    "gpe_stn": -1, "gpe_gpi": -1,
    "stn_gpi": +1, "gpi_thal": -1, "thal_ctx": +1,
    "snc_d1": +1, "snc_d2": -1,
    "sero_gpi": +1, "ne_stn": +1,
    "bayes_striatum": +1, "bayes_gpi": +1,
}

def validate_topology(connections: dict) -> bool:
    print("\n--- Topology Validation ---")
    passed = True

    # Check all required connections exist
    for req in REQUIRED_CONNECTIONS:
        if req in connections:
            print(f"  [OK ] {req:<22s} present")
        else:
            print(f"  [ERR] {req:<22s} MISSING")
            passed = False

    # Check sign constraints
    for cname, con in connections.items():
        expected = SIGN_RULES.get(cname)
        if expected is None:
            continue
        import numpy as np
        nz = con.W[con.W != 0]
        if len(nz) == 0:
            continue
        actual_sign = +1 if nz.mean() > 0 else -1
        ok = actual_sign == expected
        status = "OK " if ok else "ERR"
        print(f"  [{status}] {cname:<22s} "
              f"sign={actual_sign:+d} (expected {expected:+d})")
        if not ok:
            passed = False

    print(f"\n  Topology {'PASSED' if passed else 'FAILED'}")
    return passed


def validate_sparsity(connections: dict,
                       min_s: float = 0.3,
                       max_s: float = 0.9) -> bool:
    print("\n--- Sparsity Validation ---")
    passed = True
    for cname, con in connections.items():
        s = con.sparsity()
        ok = min_s <= s <= max_s
        status = "OK " if ok else "WARN"
        print(f"  [{status}] {cname:<22s} sparsity={s:.2f}")
        if not ok:
            passed = False
    return passed


def validate_weight_magnitudes(connections: dict,
                                min_w: float = 1e-11,
                                max_w: float = 1e-8) -> bool:
    print("\n--- Weight Magnitude Validation ---")
    passed = True
    for cname, con in connections.items():
        stats = con.weight_stats()
        mean_abs = abs(stats["mean"])
        ok = min_w <= mean_abs <= max_w
        status = "OK " if ok else "WARN"
        print(f"  [{status}] {cname:<22s} "
              f"|mean|={mean_abs:.2e}  max={stats['max']:.2e}")
        if not ok:
            passed = False
    return passed