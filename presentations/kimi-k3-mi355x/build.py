from core import *
from layouts import render
import shutil,zipfile
from collections import defaultdict

p=make_prs();stage=make_prs();meta=[];canvases=[]
guide=OUT/"guide";guide.mkdir(exist_ok=True)
chap=defaultdict(list)
for i,data in enumerate(SLIDES,1):
    c=Canvas(p,data,i);render(c);c.finish();canvases.append(c)
    pages=[]
    for count in range(1,len(c.groups)+1):
        hide={sp for group in c.groups[count:] for sp in group}
        duplicate_visible(c.slide,stage,hide)
        pages.append(len(stage.slides)-1)
    if not pages:
        duplicate_visible(c.slide,stage,set());pages=[len(stage.slides)-1]
    meta.append({"index":i,"chapter":data["chapter"],"title":data["title"],"note":data["note"],
                 "pages":pages,"builds":len(c.groups),"source_keys":data.get("refs",[])})
    lines=[f'<a id="s-{i:02d}"></a>',f'## Slide {i:02d}: {data["title"]}',"",data["subtitle"],""]
    for item in data["items"]:lines.append(f'- **{item[0]}:** {item[1]}')
    lines.extend(["",data.get("formula",""),"",data["note"],"","### Sources",""])
    for key in data.get("refs",[]):
        src=SOURCES[key];lines.append(f'- [{src["label"]}]({src["url"]})')
    lines.extend(["","---",""]);chap[data["chapter"]].append("\n".join(lines))
    for sid,x,y,w,h in c.geometry:
        if min(x,y)<-.01 or x+w>W+.02 or y+h>H+.02:
            raise ValueError(f"Out-of-slide shape {i}:{sid}: {(x,y,w,h)}")
static=OUT/"Kimi-K3_MI355X_Reference.pptx";p.save(static)
for c in canvases:add_animation(c.slide,c.groups)
p.save(OUT/"Kimi-K3_MI355X_Animated.pptx")
stage.save(OUT/"Video_Build_States.pptx")
(OUT/"video_manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
for key,parts in chap.items():
    title=CHAPTERS[key];name=key+"-"+slug(title)+".md"
    (guide/name).write_text(f'# {title}\n\n[Handbook contents](../HANDBOOK.md) · [Worked examples](11-worked-examples.md)\n\n'+"\n".join(parts),encoding="utf-8")
(guide/"11-worked-examples.md").write_text((ROOT/"worked-examples.md").read_text(),encoding="utf-8")
sources="# Sources and evidence\n\nPrepared 2026-10-29. All diagrams are original, editable teaching schematics.\n\n"
for key,src in SOURCES.items():sources+=f'- **{key}:** [{src["label"]}]({src["url"]})\n'
sources+="\n## Evidence boundary\n\nThe 17,406.08912 result is the completed cached FP8 projection + cached FP8 FMHA prototype. New-token FP8 full performance is not established. Run 40804432321 failed on runner disk quota. Prototype GSM8K and native-PR validation are separate.\n"
(OUT/"SOURCES.md").write_text(sources,encoding="utf-8")
intro="""# Kimi-K3 on AMD MI355X — handbook

This is a conceptual course with an editable animated PowerPoint, a static reference PowerPoint/PDF, a narrated chaptered video and detailed Markdown notes.

## How to use it

- Open **Kimi-K3_MI355X_Animated.pptx** in desktop PowerPoint and start Slide Show (F5). Click to reveal the next operation. The Animation Pane exposes editable fade entrance groups.
- The **Reference** PowerPoint and PDF show complete diagrams immediately.
- Each slide's **READ CHAPTER** link opens the relevant online Markdown section. The same files are included here for offline reading.
- The video uses the same progressive diagram states, English synthetic narration and captions. Video and PowerPoint timings are designed separately: the deck is presenter-controlled, the video is narrated.
- Green = FP8; blue = BF16; purple = FP32; amber = conversion/decision. Color is always accompanied by text.
- Model dimensions are pinned to the linked checkpoint. Diagrams of execution and hardware topology are schematic unless explicitly labeled otherwise.

## Learning routes

**Start from basics:** chapters 01–03, then the attention and KDA worked examples.

**Understand the GPU:** chapters 04–05, then matrix layout, state sizing and graph-replay notes.

**Understand the FP8 experiment:** chapters 06–08 plus quantization units, LSE merging and accuracy examples.

**Complete workshop:** all slides in order; pause at the six understanding questions. Allow multiple sessions for the full material.

## Chapters

"""
for key in sorted(chap):intro+=f'- [{key}. {CHAPTERS[key]}](guide/{key}-{slug(CHAPTERS[key])}.md)\n'
intro+="\n- [11. Worked examples, math, sampling and common misunderstandings](guide/11-worked-examples.md)\n- [Sources and evidence](SOURCES.md)\n\n## Important implementation distinction\n\nThis checkpoint sets mla_use_nope=true. The 64-column field named qk_rope_head_dim is retained in the generic MLA layout, but the inspected K3 MLA wrapper passes rotary_emb=None. The diagrams call it the shared K component rather than claiming rotary position encoding is active.\n"
(OUT/"HANDBOOK.md").write_text(intro,encoding="utf-8")
with zipfile.ZipFile(OUT/"Kimi-K3_MI355X_Animated.pptx") as z:
    count=0;effects=0
    for name in z.namelist():
        if re.fullmatch(r"ppt/slides/slide\d+\.xml",name):
            doc=etree.fromstring(z.read(name));ns={"p":"http://schemas.openxmlformats.org/presentationml/2006/main"}
            ids=set(doc.xpath("//p:cNvPr/@id",namespaces=ns))
            targets=doc.xpath("//p:timing//p:spTgt/@spid",namespaces=ns)
            assert set(targets)<=ids,(name,set(targets)-ids)
            tids=doc.xpath("//p:timing//p:cTn/@id",namespaces=ns);assert len(tids)==len(set(tids))
            effects+=len(doc.xpath("//p:animEffect",namespaces=ns));count+=1
    assert count==len(SLIDES)
report={"slides":len(SLIDES),"video_build_pages":len(stage.slides),"entrance_effects":effects,
        "notes_word_count":sum(len(d["note"].split()) for d in SLIDES),"shape_bounds":"passed",
        "animation_targets":"passed","timing_ids":"unique"}
(OUT/"build-validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
