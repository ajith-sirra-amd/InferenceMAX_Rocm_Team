from pathlib import Path
import os,json,subprocess,base64,zipfile
from core import OUT,ROOT,REPO,BRANCH
def api(path,payload=None):
    args=["gh","api",path]
    if payload is not None:args+=["--method","POST","--input","-"]
    p=subprocess.run(args,input=json.dumps(payload) if payload is not None else None,
        capture_output=True,text=True,check=True)
    return json.loads(p.stdout)
assert os.environ["GITHUB_REPOSITORY"]==REPO
current=api(f"repos/{REPO}/git/ref/heads/{BRANCH}")["object"]["sha"]
assert current==os.environ["GITHUB_SHA"],"Source branch advanced; refuse to overwrite"
parent=api(f"repos/{REPO}/git/commits/{current}")
files=[OUT/"Kimi-K3_MI355X_Animated.pptx",OUT/"Kimi-K3_MI355X_Reference.pptx",
OUT/"Kimi-K3_MI355X_Reference.pdf",OUT/"HANDBOOK.md",OUT/"SOURCES.md",
OUT/"build-validation.json",OUT/"render-validation.json"]
files+=list((OUT/"guide").glob("*.md"))+list(OUT.glob("contact-sheet-*.png"))+list(OUT.glob("preview-slide-*.png"))
tree=[]
for f in files:
    data=base64.b64encode(f.read_bytes()).decode()
    sha=api(f"repos/{REPO}/git/blobs",{"content":data,"encoding":"base64"})["sha"]
    rel=f.relative_to(OUT).as_posix()
    tree.append({"path":"presentations/kimi-k3-mi355x/"+rel,"mode":"100644","type":"blob","sha":sha})
readme="""# Kimi-K3 on AMD MI355X

A 75-slide conceptual teaching deck with click-by-click builds, speaker notes, an illustrated reference PDF and detailed Markdown chapters.

- [Download animated PowerPoint](Kimi-K3_MI355X_Animated.pptx?raw=true)
- [Download reference PDF](Kimi-K3_MI355X_Reference.pdf?raw=true)
- [Read the handbook](HANDBOOK.md)
- [Worked examples](guide/11-worked-examples.md)
- [Source references](SOURCES.md)
- [Slide overview](contact-sheet-01.png)

The narrated MP4 course, chapter videos and captions are retained in the owner's local Downloads folder. Automated video publication is disabled. See VIDEO.md for the chapter guide.

Use desktop PowerPoint Slide Show for animation. The PDF and reference PPTX intentionally show complete diagrams.
"""
blob=api(f"repos/{REPO}/git/blobs",{"content":readme,"encoding":"utf-8"})["sha"]
tree.append({"path":"presentations/kimi-k3-mi355x/README.md","mode":"100644","type":"blob","sha":blob})
t=api(f"repos/{REPO}/git/trees",{"base_tree":parent["tree"]["sha"],"tree":tree})
commit=api(f"repos/{REPO}/git/commits",{"message":"docs: publish Kimi-K3 MI355X teaching deck and handbook",
"tree":t["sha"],"parents":[current]})
subprocess.run(["gh","api",f"repos/{REPO}/git/refs/heads/{BRANCH}","--method","PATCH","--input","-"],
input=json.dumps({"sha":commit["sha"],"force":False}),text=True,check=True)
with zipfile.ZipFile(OUT/"Kimi-K3_PowerPoint_and_Handbook.zip","w",zipfile.ZIP_DEFLATED) as z:
    for f in files:z.write(f,f.relative_to(OUT).as_posix())
print("PUBLISHED",commit["sha"])
