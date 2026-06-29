import sys, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

h = {
    "Authorization": f"Bearer {cfg.GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# 1. identita'
r = requests.get("https://api.github.com/user", headers=h, timeout=10)
print("=== /user ===", r.status_code)
if r.ok:
    u = r.json()
    print(f"  login: {u['login']}")

# 2. scopes concessi (nell'header della risposta)
print(f"  X-OAuth-Scopes: {r.headers.get('X-OAuth-Scopes', 'n/a')}")

# 3. accesso diretto al repo (senza actions)
r2 = requests.get(f"https://api.github.com/repos/{cfg.GITHUB_REPO}", headers=h, timeout=10)
print(f"\n=== /repos/{cfg.GITHUB_REPO} ===", r2.status_code)
if r2.ok:
    print(f"  private: {r2.json().get('private')}")
    print(f"  permissions: {r2.json().get('permissions')}")
else:
    print(f"  error: {r2.text[:300]}")

# 4. actions runs (endpoint che falliva)
r3 = requests.get(
    f"https://api.github.com/repos/{cfg.GITHUB_REPO}/actions/workflows/e2e-tests.yml/runs",
    headers=h, params={"per_page": 3}, timeout=10
)
print(f"\n=== actions/runs === {r3.status_code}")
if not r3.ok:
    print(f"  error: {r3.text[:300]}")
else:
    print(f"  total: {r3.json().get('total_count')}")
