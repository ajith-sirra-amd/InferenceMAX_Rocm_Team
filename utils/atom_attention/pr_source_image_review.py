"""Read-only source applicability and ROCm image metadata for PR 64288."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path.cwd()
OUT = ROOT / "pr-review-evidence"
OUT.mkdir(exist_ok=True)
REPORT = {"checked_at": datetime.now(timezone.utc).isoformat(), "errors": []}
BASE = "a1cfd2c1b584c4ac197db9cc9ac84a6d973d39df"
NATIVE = "492080779e9feea19308092b0aba0840e87c6115"
REFERENCES = {
    59124: "132f6daa6b17cb4000ef58355943bb39339104f2",
    62447: "cefacabced64eda821bcb49493cc970056d15429",
}
CURRENT_IMAGE = (
    "vllm/vllm-openai-rocm:nightly-rocm100-"
    "edad8b03c5285a1c12e7e4f83c1af709b319a01b@sha256:"
    "009169c114af5b15e25eddc808afb3c1fc3f12c94e746472b16f39fcf39cd716"
)


def run(args, *, cwd=ROOT, data=None, check=True, timeout=180):
    result = subprocess.run(args, cwd=cwd, input=data, text=True,
                            capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"{args}: {result.stderr[-4000:]}")
    return {"exit_code": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}


def gh_json(endpoint):
    return json.loads(run(["gh", "api", endpoint])["stdout"])


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "native-fp8-pr-review"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def inspect_sources():
    source = ROOT / "review-vllm"
    source.mkdir()
    run(["git", "init", str(source)])
    for label, sha in (("base", BASE), ("native", NATIVE)):
        run(["git", "fetch", "--depth=1",
             "https://github.com/ajith-sirra-amd/vllm.git", sha],
            cwd=source, timeout=300)
    observations = {}
    for number, expected in REFERENCES.items():
        endpoint = f"repos/vllm-project/vllm/pulls/{number}"
        before = gh_json(endpoint)
        if before["head"]["sha"] != expected:
            raise RuntimeError(f"Reference #{number} moved; review required")
        patch = run(["gh", "api", endpoint, "-H",
                     "Accept: application/vnd.github.diff"])["stdout"]
        if gh_json(endpoint)["head"]["sha"] != expected:
            raise RuntimeError(f"Reference #{number} changed while reading")
        (OUT / f"pr-{number}.diff").write_text(patch)
        item = {"head": expected, "base": before["base"]["sha"],
                "state": before["state"], "checks": {}}
        for label, sha in (("common_base", BASE), ("native_head", NATIVE)):
            run(["git", "checkout", "--detach", sha], cwd=source)
            item["checks"][label] = run(
                ["git", "apply", "--check", "-"], cwd=source, data=patch,
                check=False)
        observations[str(number)] = item
    REPORT["sources"] = {"common_base": BASE, "native_head": NATIVE,
                         "references": observations,
                         "scope": "Patch applicability only; not runtime validation"}


def inspect_images():
    tags = {}
    for name in ("nightly-rocm100", "nightly"):
        query = urlencode({"page_size": 100, "name": name,
                           "ordering": "last_updated"})
        payload = fetch_json(
            "https://hub.docker.com/v2/repositories/vllm/"
            f"vllm-openai-rocm/tags?{query}")
        for item in payload.get("results", []):
            if item["name"].startswith("nightly"):
                tags[item["name"]] = {
                    key: item.get(key)
                    for key in ("name", "digest", "last_updated",
                                "full_size", "images")
                }
    ordered = sorted(tags.values(),
                     key=lambda x: x["last_updated"] or "", reverse=True)
    REPORT["images"] = {"current_image": CURRENT_IMAGE,
                        "tags": ordered,
                        "scope": "Registry metadata only; no image was deployed"}
    if not ordered:
        raise RuntimeError("Registry returned no nightly image candidates")
    candidates = [item for item in ordered
                  if item["name"].startswith("nightly-rocm100")][:5]
    selected = [("current", CURRENT_IMAGE)]
    selected += [(item["name"], "vllm/vllm-openai-rocm@"+item["digest"])
                 for item in candidates if item.get("digest")]
    manifests = {}
    for name, image in selected:
        manifests[name] = run(
            ["docker", "buildx", "imagetools", "inspect", image,
             "--format", "{{json .Image}}"], check=False)
    REPORT["images"]["configs"] = manifests
    REPORT["images"]["candidate_names"] = [item["name"] for item in candidates]


def inspect_wheels():
    wheels = {}
    for package in ("vllm", "torch", "triton"):
        url = f"https://wheels.vllm.ai/rocm/{BASE}/rocm100/{package}/metadata.json"
        try:
            wheels[package] = {"url": url, "metadata": fetch_json(url)}
        except Exception as exc:
            wheels[package] = {"url": url, "error": str(exc)}
    REPORT["base_wheels"] = wheels


for name, function in (("sources", inspect_sources),
                       ("images", inspect_images),
                       ("wheels", inspect_wheels)):
    try:
        function()
    except Exception as exc:
        REPORT["errors"].append({"stage": name, "error": str(exc)})
    (OUT / "review-report.json").write_text(json.dumps(REPORT, indent=2)+"\n")
print(json.dumps({
    "errors": REPORT["errors"],
    "sources": REPORT.get("sources"),
    "image_candidates": REPORT.get("images", {}).get("candidate_names", []),
}, indent=2))
raise SystemExit(bool(REPORT["errors"]))
