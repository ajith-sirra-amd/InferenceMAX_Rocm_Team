"""Run independent chain numerics or replay a captured real Kimi failure."""
import os

if __name__ == "__main__":
    fixture_path = os.environ.get("FAILURE_FIXTURE_PATH", "")
    if fixture_path:
        from rope_balanced_diagnostics import run
        run(fixture_path)
    else:
        from fp8_weight_preflight import run as run_fp8_weights
        from projection_numerics import run_case
        from direct_kv_preflight import run
        run_fp8_weights()
        run_case(0.125)
        run_case(0.137)
        print("KV_CHAIN_NUMERICS_PASS", flush=True)
        run()
