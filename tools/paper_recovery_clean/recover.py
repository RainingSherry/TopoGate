#!/usr/bin/env python3
from __future__ import annotations
import os, re, sys, json, time, textwrap
from pathlib import Path
from urllib.parse import quote
import requests

PID = int(os.environ["RECOVER_ID"])
ROOT = Path("recovery_output") / str(PID)
ROOT.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
})
FENCE = chr(96) * 3

TARGETS = {
  1:  {"title":"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second","category":"01_TFM_PFN","mode":"translate","artifact_dir":"source/01","prefix":"01_"},
  18: {"title":"Table Foundation Models: on Knowledge Pre-training for Tabular Learning","category":"02_CrossTable_Multimodal_Semantic","mode":"recover",
       "urls":["https://arxiv.org/pdf/2505.14415.pdf"]},
  24: {"title":"ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data","category":"03_Unsupervised_Clustering_FM","mode":"translate","artifact_dir":"source/03","prefix":"24_"},
  46: {"title":"Dual Mutual Information Constraints for Discriminative Clustering","category":"04_Deep_Clustering_Local_Graph_Topology","mode":"recover",
       "doi":"10.1609/aaai.v37i7.26032",
       "urls":["https://ojs.aaai.org/index.php/AAAI/article/download/26032/25804"],
       "jina":["https://r.jina.ai/https://ojs.aaai.org/index.php/AAAI/article/download/26032/25804"]},
  57: {"title":"CTSyn: A Foundation Model for Cross Tabular Data Generation","category":"06_Tabular_Generation_Schema","mode":"translate","artifact_dir":"source/06","prefix":"57_"},
  60: {"title":"Efficient Table Generation for Zero-Shot Column Type Annotation","category":"06_Tabular_Generation_Schema","mode":"recover",
       "urls":["https://openreview.net/pdf?id=vgHqweeBMb",
               "https://www.researchgate.net/publication/profile/Ehsan-Hoseinzade/publication/400771892_Efficient_Table_Generation_for_Zero-Shot_Column_Type_Annotation/links/698f72bd5d60ab48356db524/Efficient-Table-Generation-for-Zero-Shot-Column-Type-Annotation.pdf"],
       "jina":["https://r.jina.ai/https://openreview.net/pdf?id=vgHqweeBMb",
               "https://r.jina.ai/https://www.researchgate.net/publication/400771892_Efficient_Table_Generation_for_Zero-Shot_Column_Type_Annotation"]},
  62: {"title":"Generating Realistic Synthetic Tabular Data with Integrated LLM and Diffusion Models","category":"06_Tabular_Generation_Schema","mode":"recover",
       "doi":"10.1016/j.neucom.2025.131371",
       "urls":["https://www.sciencedirect.com/science/article/pii/S0925231225020430/pdfft?isDTMRedir=true&download=true",
               "https://api.elsevier.com/content/article/pii/S0925231225020430?httpAccept=application/pdf"],
       "jina":["https://r.jina.ai/https://www.sciencedirect.com/science/article/pii/S0925231225020430"]},
  70: {"title":"MedTransTab: Advancing Medical Cross-Table Tabular Data Generation","category":"06_Tabular_Generation_Schema","mode":"recover",
       "doi":"10.1145/3701551.3703501",
       "urls":["https://dl.acm.org/doi/pdf/10.1145/3701551.3703501"],
       "jina":["https://r.jina.ai/https://dl.acm.org/doi/10.1145/3701551.3703501"]},
  71: {"title":"Representation Learning for Tabular Data: A Comprehensive Survey","category":"08_Surveys","mode":"translate","artifact_dir":"source/08","prefix":"71_"},
  72: {"title":"The Current Generation of Tabular Foundation Models: A Critical Review","category":"08_Surveys","mode":"recover",
       "doi":"10.3390/make8080244",
       "urls":["https://www.mdpi.com/2504-4990/8/8/244/pdf",
               "https://www.mdpi.com/2504-4990/8/8/244/pdf-vor",
               "https://mdpi-res.com/d_attachment/make/make-08-00244/article_deploy/make-08-00244-v2.pdf",
               "https://mdpi-res.com/d_attachment/make/make-08-00244/article_deploy/make-08-00244.pdf"],
       "jina":["https://r.jina.ai/https://www.mdpi.com/2504-4990/8/8/244"]},
}

def safe_stem(pid, title):
    return f"{pid:02d}_" + re.sub(r"[^A-Za-z0-9._-]+","_",title).strip("_")[:150]

def req(url, timeout=60, **kw):
    last=None
    for i in range(5):
        try:
            return S.get(url, timeout=timeout, allow_redirects=True, **kw)
        except Exception as e:
            last=e
            time.sleep(2+i*2)
    raise last

def valid_pdf(b):
    return len(b)>5000 and b[:5]==b"%PDF-"

def discover_oa(t):
    out=[]
    doi=t.get("doi")
    if doi:
        try:
            r=req("https://api.openalex.org/works/https://doi.org/"+doi,30)
            if r.ok:
                w=r.json()
                for loc in [w.get("best_oa_location"),w.get("primary_location")]+list(w.get("locations") or []):
                    if not loc: continue
                    for k in ("pdf_url","landing_page_url"):
                        u=loc.get(k)
                        if u and u not in out: out.append(u)
        except Exception as e:
            print("openalex",repr(e))
        try:
            r=req("https://api.unpaywall.org/v2/"+doi+"?email=paper-recovery@example.com",30)
            if r.ok:
                d=r.json()
                locs=[d.get("best_oa_location")]+list(d.get("oa_locations") or [])
                for loc in locs:
                    if not loc: continue
                    for k in ("url_for_pdf","url"):
                        u=loc.get(k)
                        if u and u not in out: out.append(u)
        except Exception as e:
            print("unpaywall",repr(e))
    return out

def discover_openreview(title):
    outs=[]
    for base in ("https://api2.openreview.net/notes?content.title=","https://api.openreview.net/notes?content.title="):
        try:
            r=req(base+quote(title),30)
            if not r.ok: continue
            for n in r.json().get("notes",[]):
                nid=n.get("id")
                if nid:
                    outs += [f"https://openreview.net/pdf?id={nid}",f"https://openreview.net/attachment?id={nid}&name=pdf"]
        except Exception as e:
            print("openreview api",repr(e))
    return list(dict.fromkeys(outs))

def try_pdf_urls(urls):
    tried=[]
    for u in list(dict.fromkeys(urls)):
        tried.append(u)
        try:
            r=req(u,75,headers={"Accept":"application/pdf,*/*"})
            print("PDFTRY",r.status_code,r.url,r.headers.get("content-type"),len(r.content),flush=True)
            if r.ok and valid_pdf(r.content):
                return r.content,r.url,tried
        except Exception as e:
            print("PDFERR",u,repr(e),flush=True)
    return None,None,tried

def find_source_md(t):
    base=Path(t["artifact_dir"])
    hits=list(base.rglob(t["prefix"]+"*.md"))
    hits=[p for p in hits if not p.name.endswith("_zh.md")]
    if len(hits)!=1:
        raise RuntimeError(f"expected one source md for {PID}, found {hits}")
    return hits[0]

def jina_markdown(urls):
    for u in urls:
        try:
            r=req(u,90,headers={"Accept":"text/plain"})
            txt=r.text
            print("JINA",r.status_code,u,len(txt),flush=True)
            if r.ok and len(txt)>5000:
                return txt,u
        except Exception as e:
            print("JINAERR",u,repr(e),flush=True)
    return None,None

def reconstruct_pdf_from_text(text, out_pdf, title, source):
    import fitz
    doc=fitz.open()
    margin=48
    fontsize=9.5
    lineheight=13
    width=595
    height=842
    notice=("RECONSTRUCTED PUBLIC-FULL-TEXT COPY\n"
            "Original publisher PDF endpoint was inaccessible to the automated downloader.\n"
            f"Source: {source}\n\n")
    plain=re.sub(r"!\[[^\]]*\]\([^)]+\)","[Image omitted]",text)
    plain=re.sub(r"\[([^\]]+)\]\([^)]+\)",r"\1",plain)
    plain=re.sub(r"[#*_\x60>|]","",plain)
    plain=notice+plain
    lines=[]
    for para in plain.splitlines():
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=95, replace_whitespace=False, drop_whitespace=False) or [""])
    i=0
    while i<len(lines):
        page=doc.new_page(width=width,height=height)
        y=margin
        if page.number==0:
            page.insert_text((margin,y),title,fontsize=13,fontname="helv")
            y+=24
        while i<len(lines) and y<height-margin:
            page.insert_text((margin,y),lines[i],fontsize=fontsize,fontname="helv")
            y+=lineheight
            i+=1
    doc.save(out_pdf)
    doc.close()

PROTECTED_RE=re.compile(r'(https?://\S+|doi:\s*\S+|\x60[^\x60\n]+\x60|\$\$.*?\$\$|\$[^\n$]+\$|\\\([^\n]*?\\\)|\\\[[\s\S]*?\\\])',re.I)
def protect(s):
    saved=[]
    def f(m):
        k=f"__PX{len(saved):05d}XP__"; saved.append(m.group(0)); return k
    return PROTECTED_RE.sub(f,s),saved
def restore(s,saved):
    for i,v in enumerate(saved): s=s.replace(f"__PX{i:05d}XP__",v)
    return s

def gtranslate(s):
    if not s.strip(): return s
    p,saved=protect(s)
    params={"client":"gtx","sl":"en","tl":"zh-CN","dt":"t","q":p}
    last=None
    for i in range(12):
        try:
            r=req("https://translate.googleapis.com/translate_a/single",90,params=params)
            if r.ok:
                d=r.json()
                z="".join(x[0] for x in d[0] if x and x[0] is not None)
                return restore(z,saved)
            last=f"{r.status_code} {r.text[:100]}"
        except Exception as e: last=repr(e)
        time.sleep(min(30,2+i*2))
    raise RuntimeError(last)

def translate_md(md):
    blocks=re.split(r"(\n\s*\n)",md)
    out=list(blocks)
    refs=False
    items=[]
    for i,b in enumerate(blocks):
        s=b.strip()
        if not s: continue
        if re.match(r"^#{1,6}\s+(references|bibliography)\s*$",s,re.I):
            refs=True; continue
        if refs or (s.startswith(FENCE) and s.endswith(FENCE)): continue
        alpha=sum(c.isalpha() for c in s)
        if alpha/max(1,len(s))<0.16: continue
        m=re.match(r"^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)?)([\s\S]*)$",b)
        items.append((i,m.group(1),m.group(2)))
    fails=0
    pos=0
    while pos<len(items):
        batch=[]; n=0
        while pos<len(items):
            i,prefix,core=items[pos]
            marker=f"ZXQBLOCK{i:05d}QXZ"
            add=marker+"\n"+core+"\n"
            if batch and n+len(add)>2800: break
            batch.append((i,prefix,core,marker)); n+=len(add); pos+=1
        packed="\n".join(m+"\n"+c for i,p,c,m in batch)
        try:
            z=gtranslate(packed)
            ps=[]
            for i,p,c,m in batch:
                q=z.find(m)
                if q<0: raise ValueError("marker missing")
                ps.append((q,i,p,m))
            ps.sort()
            for k,(q,i,p,m) in enumerate(ps):
                st=q+len(m); en=ps[k+1][0] if k+1<len(ps) else len(z)
                out[i]=p+z[st:en].strip()
        except Exception as e:
            print("batch fallback",repr(e),flush=True)
            for i,p,c,m in batch:
                try: out[i]=p+gtranslate(c)
                except Exception as e2:
                    fails+=1; print("block failed",i,repr(e2),flush=True)
        time.sleep(0.8)
    return "".join(out),fails

def zh_ratio(s):
    z=len(re.findall(r"[\u4e00-\u9fff]",s)); e=len(re.findall(r"[A-Za-z]",s))
    return round(z/max(1,z+e),4)

def pdf_to_md(pdf,t,source):
    header=f"# {t['title']}\n\n- 合并类别: {t['category']}\n- PDF 来源: {source}\n\n---\n\n"
    try:
        import pymupdf4llm
        body=pymupdf4llm.to_markdown(str(pdf),show_progress=False)
        if len(body)>500: return header+body
    except Exception as e: print("pymupdf4llm",repr(e))
    import fitz
    d=fitz.open(pdf); parts=[]
    for n,p in enumerate(d,1): parts.append(f"\n\n<!-- page {n} -->\n\n"+p.get_text())
    d.close()
    return header+"".join(parts)

def main():
    t=TARGETS[PID]
    stem=safe_stem(PID,t["title"])
    pdf=ROOT/(stem+".pdf")
    en=ROOT/(stem+".md")
    zh=ROOT/(stem+"_zh.md")
    report={"id":PID,"title":t["title"],"category":t["category"],"mode":t["mode"],
            "pdf":"","md_en":"","md_zh":"","source_used":"","source_type":"","zh_ratio":0,"translation_failures":0}
    if t["mode"]=="translate":
        src=find_source_md(t)
        md=src.read_text(encoding="utf-8")
        en.write_text(md,encoding="utf-8")
        report["md_en"]="ok_reused"
        z,fails=translate_md(md)
        zh.write_text(z,encoding="utf-8")
        report["translation_failures"]=fails
        report["zh_ratio"]=zh_ratio(z)
        report["md_zh"]="ok" if fails==0 and report["zh_ratio"]>=0.20 else "partial"
        report["pdf"]="existing_original"
        report["source_type"]="existing_artifact"
    else:
        urls=list(t.get("urls",[]))+discover_oa(t)
        if PID in (60,70): urls += discover_openreview(t["title"])
        b,u,tried=try_pdf_urls(urls)
        if b:
            pdf.write_bytes(b)
            report["pdf"]="ok"
            report["source_used"]=u
            report["source_type"]="original_pdf"
            md=pdf_to_md(pdf,t,u)
        else:
            jt,ju=jina_markdown(t.get("jina",[])+["https://r.jina.ai/"+x for x in urls if x.startswith("https://")])
            if not jt:
                report["pdf"]="failed"; report["md_en"]="failed"; report["md_zh"]="failed"
                (ROOT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
                sys.exit(2)
            reconstruct_pdf_from_text(jt,pdf,t["title"],ju)
            report["pdf"]="reconstructed_public_fulltext"
            report["source_used"]=ju
            report["source_type"]="public_fulltext_reconstruction"
            md=f"# {t['title']}\n\n- 合并类别: {t['category']}\n- 来源: {ju}\n- 说明: publisher PDF endpoint inaccessible; content recovered from public full text.\n\n---\n\n"+jt
        en.write_text(md,encoding="utf-8"); report["md_en"]="ok"
        z,fails=translate_md(md)
        zh.write_text(z,encoding="utf-8")
        report["translation_failures"]=fails
        report["zh_ratio"]=zh_ratio(z)
        report["md_zh"]="ok" if fails==0 and report["zh_ratio"]>=0.20 else "partial"
    (ROOT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report["md_zh"]!="ok" or report["md_en"] not in ("ok","ok_reused") or report["pdf"]=="failed":
        sys.exit(3)

if __name__=="__main__": main()
