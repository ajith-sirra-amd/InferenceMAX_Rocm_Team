"""Run independent chain numerics or replay a captured real Kimi failure."""
import os

if __name__ == "__main__":
    from fp8_weight_preflight import run as run_fp8_weights
    run_fp8_weights()
    fixture_path = os.environ.get("FAILURE_FIXTURE_PATH", "")
    if fixture_path:
        from replay_failure import run
        run(fixture_path)
    else:
        from projection_numerics import run_case
        from direct_kv_preflight import run
        run_case(0.125)
        run_case(0.137)
        print("KV_CHAIN_NUMERICS_PASS", flush=True)
        run()
