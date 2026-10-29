from pathlib import Path
import subprocess,json,urllib.request,wave,re,shutil,math
from PIL import Image
from core import ROOT,OUT,CHAPTERS,slug
VOICE=ROOT/".voice";VOICE.mkdir(exist_ok=True)
voicebase="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
for suffix in ["",".json"]:
    target=VOICE/("voice.onnx"+suffix)
    if not target.exists():urllib.request.urlretrieve(voicebase+suffix,target)
WORK=OUT/"video-work";WORK.mkdir(exist_ok=True)
VID=OUT/"videos";VID.mkdir(exist_ok=True)
meta=json.loads((OUT/"video_manifest.json").read_text())
fontstyle="FontName=DejaVu Sans,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101A2A,BackColour=&H00101A2A,BorderStyle=1,Outline=1,Shadow=0,MarginV=10,Alignment=2"
def run(args,**kwargs):subprocess.run(args,check=True,**kwargs)
def stamp(s):
    ms=round(s*1000);h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);ss,ms=divmod(ms,1000)
    return f"{h:02d}:{m:02d}:{ss:02d},{ms:03d}"
def speech_text(s):
    mapping={"Kimi-K3":"Kee mee K three","Kimi Delta":"Kee mee Delta","MI355X":"M I three fifty five X",
    "BF16":"B F sixteen","FP32":"F P thirty two","FP8":"F P eight","MXFP4":"M X F P four",
    "MXFP8":"M X F P eight","FP4":"F P four","FMHA":"F M H A","MLA":"M L A","KDA":"K D A",
    "MoE":"M O E","HBM":"H B M","LSE":"L S E","DCP8":"D C P eight","TP8":"T P eight",
    "DCP":"D C P","vLLM":"V L L M","GSM8K":"G S M eight K","SiTU":"see too","RoPE":"rope",
    "NoPE":"no positional encoding","Q/K/V":"Q, K, and V","K/V":"K and V","kv_b_proj":"K V B projection"}
    for a,b in mapping.items():s=s.replace(a,b)
    return s
def chunks(text):
    words=text.split();out=[];part=[]
    for word in words:
        if len(" ".join(part+[word]))>94 and part:out.append(" ".join(part));part=[]
        part.append(word)
    if part:out.append(" ".join(part))
    return out
fullsubs=[];offset=0;chapters={};segments=[]
for item in meta:
    i=item["index"];work=WORK/f"s{i:02d}";work.mkdir(exist_ok=True)
    wav=work/"narration.wav"
    if not wav.exists():
        run(["piper","--model",str(VOICE/"voice.onnx"),"--output_file",str(wav),"--length_scale","1.02"],
            input=speech_text(item["note"]).encode(),stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    with wave.open(str(wav)) as w:duration=w.getnframes()/w.getframerate()+.65
    phrases=chunks(item["note"]);words=sum(len(x.split()) for x in phrases)
    srt=[];t=.1
    for num,phrase in enumerate(phrases,1):
        dt=(duration-.75)*len(phrase.split())/words
        # Two balanced subtitle lines, with exact narration text.
        ws=phrase.split();cut=max(1,len(ws)//2);caption=" ".join(ws[:cut])+"\n"+" ".join(ws[cut:])
        srt.append(f"{num}\n{stamp(t)} --> {stamp(t+dt)}\n{caption}\n")
        fullsubs.append((offset+t,offset+t+dt,caption))
        t+=dt
    (work/"captions.srt").write_text("\n".join(srt),encoding="utf-8")
    pages=item["pages"];n=len(pages);hold=duration/n;entries=[]
    for pos,page in enumerate(pages):
        img=Image.open(OUT/"frames"/f"frame-{page:04d}.png").convert("RGB")
        current=work/f"state-{pos}.png";img.save(current)
        entries.extend([f"file '{current.name}'",f"duration {hold-(.333333 if pos<n-1 else 0):.6f}"])
        if pos<n-1:
            nxt=Image.open(OUT/"frames"/f"frame-{pages[pos+1]:04d}.png").convert("RGB")
            for k in range(1,5):
                blend=work/f"fade-{pos}-{k}.png";Image.blend(img,nxt,k/5).save(blend)
                entries.extend([f"file '{blend.name}'","duration 0.083333"])
    entries.append(f"file 'state-{n-1}.png'")
    (work/"frames.ffconcat").write_text("ffconcat version 1.0\n"+"\n".join(entries))
    clip=work/"clip.mp4"
    filt="fps=12,scale=1216:684,pad=1280:720:32:0:color=0x0B1220,drawbox=x=0:y=654:w=1280:h=66:color=0x0B1220:t=fill,subtitles=captions.srt:force_style='"+fontstyle+"'"
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","concat","-safe","0","-i","frames.ffconcat",
         "-i","narration.wav","-vf",filt,"-af","apad=pad_dur=0.65","-t",str(duration),
         "-c:v","libx264","-preset","veryfast","-crf","25","-pix_fmt","yuv420p","-threads","2",
         "-c:a","aac","-b:a","96k","-ar","22050","-movflags","+faststart","clip.mp4"],cwd=work)
    # Use actual muxed duration when joining captions and chapter boundaries.
    probe=json.loads(subprocess.check_output(["ffprobe","-v","quiet","-show_format","-of","json",str(clip)]))
    actual=float(probe["format"]["duration"])
    segments.append({"slide":i,"path":str(clip),"start":offset,"duration":actual,"chapter":item["chapter"]})
    ch=chapters.setdefault(item["chapter"],{"start":offset,"end":offset,"clips":[]})
    ch["clips"].append(str(clip));ch["end"]=offset+actual
    offset+=actual
    print(f'VIDEO slide {i}/{len(meta)} complete; {actual:.1f} sec',flush=True)
def join(paths,target,metadata=None):
    listing=WORK/(target.stem+".concat");listing.write_text("".join("file '"+p+"'\n" for p in paths))
    args=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","concat","-safe","0","-i",str(listing)]
    if metadata:args+=["-i",str(metadata),"-map_metadata","1"]
    args+=["-c","copy","-movflags","+faststart",str(target)];run(args)
fm=WORK/"chapters.ffmetadata";m=";FFMETADATA1\ntitle=Kimi-K3 on AMD MI355X\n"
for key,ch in chapters.items():
    m+=f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(ch["start"]*1000)}\nEND={round(ch["end"]*1000)}\ntitle={CHAPTERS[key]}\n'
fm.write_text(m)
join([x["path"] for x in segments],VID/"Kimi-K3_MI355X_Full_Course.mp4",fm)
for key,ch in chapters.items():join(ch["clips"],VID/(key+"-"+slug(CHAPTERS[key])+".mp4"))
srt="\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n" for i,(a,b,text) in enumerate(fullsubs,1))
(VID/"Kimi-K3_MI355X_Full_Course.srt").write_text(srt,encoding="utf-8")
(VID/"Kimi-K3_MI355X_Full_Course.vtt").write_text("WEBVTT\n\n"+re.sub(r"(\d\d:\d\d:\d\d),(\d\d\d)",r"\1.\2",srt),encoding="utf-8")
readme="# Narrated video course\n\nEnglish synthetic narration (Piper, en_US-lessac-medium), progressive builds and burned-in captions. No background music.\n\n"
for key,ch in chapters.items():
    readme+=f'- {stamp(ch["start"]).split(",")[0]} — {CHAPTERS[key]}: {key}-{slug(CHAPTERS[key])}.mp4\n'
readme+=f"\nFull duration: {offset/60:.1f} minutes. Resolution: 1280×720; 12 fps. The diagram changes are intentional gradual builds.\n\nVoice source: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium . Piper: https://github.com/OHF-Voice/piper1-gpl . Voice weights are not bundled.\n"
(VID/"README.md").write_text(readme,encoding="utf-8")
(OUT/"video-validation.json").write_text(json.dumps({"slides":len(segments),"duration_seconds":offset,
"chapters":len(chapters),"narration":"English synthetic","caption_cues":len(fullsubs),"segments":segments},indent=2))
print(f"VIDEO COMPLETE {offset/60:.1f} minutes",flush=True)
