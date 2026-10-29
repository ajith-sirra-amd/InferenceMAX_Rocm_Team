from core import *
def flow(c,items=None):
    items=items or c.d["items"];n=len(items);gap=.22;w=(12.1-gap*(n-1))/n
    for j,it in enumerate(items):
        color=it[2] if len(it)>2 else ["teal","bf16","fp32","amber","fp8","teal"][j%6]
        shapes=c.card(.62+j*(w+gap),2.72,w,2.5,it[0],it[1],color)
        if j:shapes.append(c.arrow(.62+j*(w+gap)-.17,3.94,.62+j*(w+gap)-.03,3.94))
        c.reveal(*shapes)
def cards(c,cols=3):
    it=c.d["items"];rows=math.ceil(len(it)/cols);gap=.25;w=(12.1-gap*(cols-1))/cols;h=(3.95-gap*(rows-1))/rows
    for i,a in enumerate(it):
        color=a[2] if len(a)>2 else ["teal","bf16","amber","fp32","fp8","teal"][i%6]
        x=.62+(i%cols)*(w+gap);y=2.13+(i//cols)*(h+gap)
        if h<2:
            shapes=[c.rect(x,y,w,h,"card",color,True),c.text(x+.18,y+.15,w-.36,.38,a[0],17,color,True),
                    c.text(x+.18,y+.65,w-.36,h-.76,a[1],14,"white")]
        else:shapes=c.card(x,y,w,h,a[0],a[1],color)
        c.reveal(*shapes)
def bars(c,values,labels,colors,unit="",y=2.45,maxval=None):
    maximum=maxval or max(values)
    for i,(value,label,col) in enumerate(zip(values,labels,colors)):
        yy=y+i*.96
        a=c.text(.7,yy,4,.3,label,16,"white")
        b=c.rect(4.55,yy,7.45,.4,"card")
        d=c.rect(4.55,yy,7.45*value/maximum,.4,col)
        e=c.text(4.55,yy+.47,7.45,.27,f"{value:,.5f}".rstrip("0").rstrip(".")+unit,13,col,True)
        c.reveal(a,b,d,e)
def render(c):
    v=c.d["viz"];it=c.d["items"]
    if v=="cover":
        for i,a in enumerate(it):
            x=.65+i*4.17
            c.reveal(*c.card(x,3.15,3.83,2.35,a[0],a[1],["teal","bf16","fp8"][i],True))
        c.text(.65,2.15,10,.55,"A CONCEPTUAL FIELD GUIDE",23,"white",True)
        return
    if v in ("journey","architecture"):cards(c,3 if len(it)==6 else 2);return
    if v in ("compare","questions"):cards(c,3);return
    if v=="stats":
        cols=3 if len(it)==6 else 2;w=(12.1-.25*(cols-1))/cols
        for i,a in enumerate(it):
            x=.62+(i%cols)*(w+.25);y=2.13+(i//cols)*2.1;col=["teal","bf16","fp8","fp32","amber","teal"][i%6]
            c.reveal(c.rect(x,y,w,1.85,"card",col,True),c.text(x+.22,y+.21,w-.44,.62,a[0],30,col,True),
                     c.text(x+.22,y+1.04,w-.44,.58,a[1],15,"white"))
        return
    if v=="glossary":cards(c,3);return
    if v in ("flow","pipeline"):flow(c);return
    if v=="equation":
        for i,a in enumerate(it):
            yy=2.14+i*(3.9/len(it))
            color=["teal","bf16","fp32","fp8"][i%4]
            circle=c.circle(.68,yy+.04,.47,color)
            num=c.text(.71,yy+.12,.4,.22,str(i+1),12,"bg",True,PP_ALIGN.CENTER)
            title=c.text(1.35,yy+.02,3.05,.53,a[0],19,color,True)
            body=c.text(4.53,yy+.02,7.85,.84,a[1],22,"white")
            c.reveal(circle,num,title,body)
        return
    if v=="qkv":
        x=c.rect(.8,3.03,2.55,1.22,"card","neutral",True)
        tx=c.text(.99,3.34,2.16,.55,"Hidden states X",21,"white",True,PP_ALIGN.CENTER)
        c.reveal(x,tx)
        for i,a in enumerate(it):
            y=2.16+i*1.3;col=["bf16","teal","fp32"][i]
            a1=c.arrow(3.45,3.64,5.1,y+.47,col)
            a2=c.rect(5.2,y,6.95,1.01,"card",col,True)
            a3=c.text(5.42,y+.18,3.8,.32,a[0],20,col,True)
            a4=c.text(9.28,y+.19,2.58,.55,a[1],12,"white")
            c.reveal(a1,a2,a3,a4)
        return
    if v=="attention":
        words=["The","robot","moved","the","box","because","it","was","heavy"]
        for j,w in enumerate(words):
            x=.67+j*1.34;col="amber" if w=="it" else "teal" if w=="box" else "muted"
            shapes=[c.rect(x,2.35,1.14,.64,"card",col,True),c.text(x+.04,2.53,1.06,.27,w,14,col,True,PP_ALIGN.CENTER)]
            if j<6:c.reveal(*shapes)
        q=c.card(4.65,4.55,4.05,1.53,"Contextual meaning","A learned weighted read","bf16")
        for j in [1,4,5]:
            q.append(c.arrow(.67+j*1.34+.57,3.12,6.65,4.4,"teal"))
        c.reveal(*q)
        c.text(.8,3.53,3.25,.75,"Illustration of relevance,\nnot a measured attention map.",14,"muted")
        return
    if v=="softmax":
        bars(c,[.665,.245,.090],["score 2 → weight","score 1 → weight","score 0 → weight"],["teal","bf16","fp32"],maxval=1)
        return
    if v=="weighted":
        flow(c,it)
        c.reveal(c.text(3.72,5.55,6,.47,"Sum the weighted vectors",23,"teal",True,PP_ALIGN.CENTER))
        return
    if v in ("mask","splitmask"):
        n=6;cw=.48;x0=1.13;y0=2.68
        c.text(x0,y0-.47,4,.3,"KEY POSITIONS →",12,"muted",True)
        c.text(.67,5.88,3,.25,"QUERY ROWS ↓",10,"muted",True)
        for i in range(n):
            row=[]
            for j in range(n):
                allow=j<=i
                row.append(c.rect(x0+j*cw,y0+i*cw,cw-.055,cw-.055,"teal" if allow else "card2"))
                if not allow:row.append(c.text(x0+j*cw+.13,y0+i*cw+.12,.18,.18,"×",9,"muted"))
            c.reveal(*row)
        if v=="splitmask":
            pref=[]
            for i in range(n):
                for j in range(3):pref.append(c.rect(4.6+j*.42,2.68+i*.48,.365,.425,"fp8"))
            c.reveal(*pref)
            c.text(4.52,2.18,2,.38,"OLD PREFIX",11,"fp8",True)
        for i,a in enumerate(it):
            y=2.38+i*.88
            c.reveal(c.text(7.0,y,5.2,.35,a[0],18,["teal","bf16","fp32","amber"][i],True),
                     c.text(7.0,y+.39,5.2,.42,a[1],13,"white"))
        return
    if v=="heads":
        for i in range(4):
            yy=2.22+i*.88
            col=["teal","bf16","fp32","amber"][i]
            c.reveal(*c.label(.85,yy,2.1,"HEAD "+str(i+1),col),
                c.rect(3.6,yy-.1,4,.53,"card",col),
                c.text(3.8,yy+.025,3.6,.22,"Independent weighted read",12,col),
                c.arrow(3.0,yy+.16,3.49,yy+.16,col))
        c.reveal(*c.card(9.05,2.5,3.3,2.7,"W_O","Mix head outputs into hidden features","fp8"))
        return
    if v=="cache":
        for r,(lab,col) in enumerate([("Persistent history","teal"),("New token rows","amber")]):
            y=2.66+r*1.23
            c.text(.75,y-.47,4,.33,lab,17,col,True)
            cells=[]
            for i in range(10 if r==0 else 3):
                cells+=[c.rect(.78+i*.61,y,.51,.57,col),c.text(.84+i*.61,y+.17,.39,.2,str(i+1),11,"bg",True,PP_ALIGN.CENTER)]
            c.reveal(*cells)
        c.reveal(*c.card(8.05,2.42,4.2,2.99,"Reuse + append","Keep the history consistent with the current request.","bf16"))
        return
    if v=="tiles":
        for t in range(3):
            shapes=[]
            for i in range(4):
                for j in range(4):
                    shapes.append(c.rect(.85+t*4.12+j*.74,2.7+i*.56,.65,.46,"teal" if j==t else "card2","line"))
            shapes.append(c.text(.88+t*4.12,5.22,3.5,.45,["Load tile","Update m, l, a","Continue streaming"][t],17,["teal","bf16","fp32"][t],True))
            c.reveal(*shapes)
        return
    if v=="layers":
        full=set(range(4,93,4))|{93}
        for row in range(4):
            shapes=[]
            for col in range(24):
                idx=row*24+col+1
                if idx>93:break
                x=.77+col*.493;y=2.6+row*.65;color="bf16" if idx in full else "teal"
                shapes.extend([c.rect(x,y,.424,.44,color),c.text(x+.015,y+.135,.39,.16,str(idx),8,"bg",True,PP_ALIGN.CENTER)])
            c.reveal(*shapes)
        c.text(.85,5.57,5,.4,"TEAL  69 KDA layers",19,"teal",True)
        c.text(6.6,5.57,5.7,.4,"BLUE  24 gated MLA layers",19,"bf16",True)
        return
    if v in ("state","mla","projection","newtoken","merge"):
        flow(c,it)
        if v=="state":c.reveal(c.text(3.6,5.57,6.5,.36,"Persistent state passes to the next token →",16,"teal",True))
        return
    if v=="experts":
        c.reveal(*c.card(.73,2.6,2.5,2.56,"Router","One token\n896 scores","amber"))
        for row in range(4):
            shapes=[]
            for col in range(8):
                active=(row*8+col)%5==0
                shapes.append(c.rect(4.0+col*.5,2.75+row*.53,.39,.39,"teal" if active else "card2","line"))
            c.reveal(*shapes)
        c.text(3.95,5.24,4.8,.52,"Illustrated subset of the expert pool",12,"muted")
        c.reveal(*c.card(9,2.6,3.3,2.56,"Combine","16 selected\n+ shared branch","fp8"))
        return
    if v=="residual":
        for i in range(4):
            x=.85+i*2.05;y=2.6
            c.reveal(*c.card(x,y,1.82,1.8,f"Block {i+1}","Depth state",["teal","bf16","fp32","amber"][i]))
            c.arrow(x+.91,4.52,6.73,5.45,"muted")
        c.reveal(c.rect(4.9,5.48,4.5,.55,"card2","fp8"),c.text(5.1,5.6,4.1,.26,"Learned weighted mixture",17,"fp8",True,PP_ALIGN.CENTER))
        return
    if v=="gpu":
        c.rect(.75,2.34,7.25,3.64,"card2","line",True)
        for r in range(2):
            row=[]
            for j in range(4):
                x=1.03+j*1.71;y=2.69+r*1.32
                row.extend([c.rect(x,y,1.45,1.08,"card","teal",True),c.text(x+.12,y+.17,1.2,.25,f"XCD {r*4+j}",14,"teal",True)])
                row.append(c.text(x+.12,y+.67,1.2,.2,"32 CUs",10,"white"))
            c.reveal(*row)
        c.reveal(*c.card(8.52,2.5,3.75,3.3,"HBM3E","288 GB\nadvertised capacity\n\nOne GPU package","bf16"))
        return
    if v=="memory":
        for i,a in enumerate(it):
            w=2.2+i*1.6;x=.7+(7.4-w)/2;y=2.22+i*.91;color=["fp32","teal","bf16","amber"][i]
            c.reveal(c.rect(x,y,w,.7,"card",color,True),c.text(x+.1,y+.17,w-.2,.33,a[0],18,color,True,PP_ALIGN.CENTER),
                     c.text(8.6,y+.14,3.95,.48,a[1],13,"white"))
        return
    if v=="gemm":
        for idx,(x,label,col,rows,cols) in enumerate([(1.0,"A: tokens × features","teal",5,4),(5.0,"B: features × outputs","bf16",4,5),(9.0,"C: token outputs","fp32",5,5)]):
            group=[]
            for i in range(rows):
                for j in range(cols):group.append(c.rect(x+j*.48,2.68+i*.48,.405,.405,col if (i<2 and j<2) else "card2","line"))
            group.append(c.text(x-.15,5.48,3.4,.5,label,14,col,True))
            c.reveal(*group)
        c.text(3.64,3.6,.6,.6,"×",32,"muted")
        c.text(7.7,3.6,.6,.6,"=",30,"muted")
        return
    if v=="roofline":
        c.arrow(1.08,5.65,11.7,5.65);c.arrow(1.08,5.65,1.08,2.17)
        c.text(4.7,5.94,7,.3,"Arithmetic intensity →",14,"muted")
        c.text(1.34,2.1,5,.3,"Attainable FLOP/s",14,"muted")
        c.reveal(c.arrow(1.42,5.4,6.35,2.95,"teal"),c.text(2.55,4.5,3.8,.4,"Bandwidth ceiling",18,"teal",True))
        c.reveal(c.arrow(6.35,2.95,11.48,2.95,"bf16"),c.text(7.15,2.39,4.3,.4,"Compute ceiling",18,"bf16",True))
        c.text(7.1,4.7,4.5,.65,"Conceptual axes;\nnot measured MI355X data.",13,"muted")
        return
    if v=="capacity":
        bars(c,[1400,2304],["Ideal 4-bit weight bytes","8-GPU advertised HBM"],["amber","bf16"],unit=" GB (decimal illustration)",y=2.8,maxval=2500)
        c.text(.74,5.25,11.8,.62,"Scales, BF16 modules, caches and workspaces consume the remaining budget.",17,"white")
        return
    if v in ("tp","dcp"):
        for row in range(2):
            group=[]
            for j in range(4):
                idx=row*4+j;x=.84+j*3.13;y=2.45+row*1.57;col="teal" if v=="tp" else "bf16"
                group.extend(c.card(x,y,2.72,1.35,f"GPU {idx}", "",col))
                group.append(c.text(x+.18,y+.82,2.38,.28,"Feature shard" if v=="tp" else "Context partition",13,"white"))
            c.reveal(*group)
        return
    if v in ("scheduler","timeline","spec"):
        names=["Request A","Request B","Request C"] if v=="scheduler" else ["Draft","Verify","Commit"] if v=="spec" else ["CPU / scheduler","GPU compute","Communication"]
        cols=["teal","bf16","fp32"]
        for i,name in enumerate(names):
            y=2.58+i*1.05
            c.text(.73,y+.09,2.8,.46,name,16,cols[i],True)
            shapes=[]
            for j in range(6):
                x=3.85+j*1.39
                if (i+j)%4==0:continue
                shapes.append(c.rect(x,y,1.16,.57,cols[i]))
            c.reveal(*shapes)
        c.text(4,5.89,7,.28,"TIME →   (schematic, not a measured trace)",12,"muted")
        return
    if v in ("dtype","quant"):
        cards(c,2)
        return
    if v=="performance":
        bars(c,[15476.46881,16915.72154,17406.08912],[a[0] for a in it],["bf16","teal","fp8"],y=2.36,maxval=18500)
        c.text(.75,5.53,11.5,.56,"+9.30% → +2.90% additional  |  +12.47% vs baseline",21,"white",True)
        return
    if v=="table":
        for i,a in enumerate(it):
            y=2.16+i*.63
            c.reveal(c.rect(.72,y,11.85,.53,"card" if i%2==0 else "card2"),
                     c.text(.9,y+.12,4.32,.31,a[0],14,"teal",True),
                     c.text(5.35,y+.12,7,.31,a[1],14,"white"))
        return
    if v=="sources":cards(c,2);return
    cards(c,2)
