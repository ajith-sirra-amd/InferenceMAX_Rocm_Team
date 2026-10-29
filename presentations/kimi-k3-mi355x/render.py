from pathlib import Path
import subprocess,json,math,re
import fitz
from PIL import Image,ImageDraw,ImageFont
from core import ROOT,OUT,SLIDES
for name in ["Kimi-K3_MI355X_Reference.pptx","Video_Build_States.pptx","Kimi-K3_MI355X_Animated.pptx"]:
    subprocess.run(["libreoffice","-env:UserInstallation=file:///tmp/kimi-lo-"+name.split(".")[0],
        "--headless","--convert-to","pdf","--outdir",str(OUT),str(OUT/name)],check=True,timeout=600)
ref=fitz.open(OUT/"Kimi-K3_MI355X_Reference.pdf")
assert len(ref)==len(SLIDES)
frames=OUT/"frames";frames.mkdir(exist_ok=True)
thumbs=[]
for i,page in enumerate(ref):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.0,1.0),alpha=False)
    im=Image.frombytes("RGB",[pix.width,pix.height],pix.samples)
    im.thumbnail((480,270))
    thumbs.append(im)
    if i in [0,5,8,19,20,28,35,36,53,55,57,64]:
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(OUT/f"preview-slide-{i+1:02d}.png")
for start in range(0,len(thumbs),20):
    cols=4;rows=math.ceil(min(20,len(thumbs)-start)/cols)
    sheet=Image.new("RGB",(cols*500,rows*306),(229,234,241));draw=ImageDraw.Draw(sheet)
    for off,im in enumerate(thumbs[start:start+20]):
        x=(off%cols)*500+10;y=(off//cols)*306+10
        sheet.paste(im,(x,y));draw.text((x,y+276),f"{start+off+1:02d}   {SLIDES[start+off]['title'][:57]}",fill=(20,32,50))
    sheet.save(OUT/f"contact-sheet-{start//20+1:02d}.png")
stage=fitz.open(OUT/"Video_Build_States.pdf")
for i,page in enumerate(stage):
    page.get_pixmap(matrix=fitz.Matrix(1280/page.rect.width,1280/page.rect.width),alpha=False).save(frames/f"frame-{i:04d}.png")
# Check that the final reference contains all intended text in native PDF text.
warnings=[]
norm=lambda s:re.sub(r"\s+","",s).replace("−","-").replace("–","-").replace("→","")
for i,(page,data) in enumerate(zip(ref,SLIDES),1):
    text=norm(page.get_text())
    if norm(data["title"]) not in text:warnings.append({"slide":i,"missing_title":data["title"]})
    for a in data["items"]:
        # Custom diagrams can intentionally replace the text with graphical labels.
        if data["viz"] in ["flow","pipeline","equation","table","compare","questions","stats","glossary","sources","dtype","quant","journey","architecture"]:
            if norm(a[0]) not in text:warnings.append({"slide":i,"missing_heading":a[0]})
    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["bbox"][2]>page.rect.width+1 or span["bbox"][3]>page.rect.height+1:
                    warnings.append({"slide":i,"off_page":span["text"]})
report={"reference_pdf_pages":len(ref),"video_pdf_pages":len(stage),"pdf_text_warnings":warnings}
(OUT/"render-validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))
if warnings:raise SystemExit("Rendered PDF validation needs review")
