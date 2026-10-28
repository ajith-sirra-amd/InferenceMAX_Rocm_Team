"""Run independent chain numerics before exercising integrated graph replay."""
from projection_numerics import run_case
from direct_kv_preflight import run

if __name__ == "__main__":
    run_case(0.125)
    run_case(0.137)
    print("KV_CHAIN_NUMERICS_PASS", flush=True)
    run()
