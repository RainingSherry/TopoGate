#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from urllib.parse import quote, urljoin
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36 academic-paper-archiver/1.0"
S = requests.Session()
S.headers.update({"User-Agent": UA, "Accept": "*/*"})
FENCE = chr(96) * 3

PROTECTED_RE = re.compile(
    r'(https?://\S+|doi:\s*\S+|\x60[^\x60\n]+\x60|\$\$.*?\$\$|\$[^\n$]+\$|\\\([^\n]*?\\\)|\\\[[\s\S]*?\\\])',
    re.I,
)

def read_rows(category: str):
    path = ROOT / f"{category}.tsv"
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) < 3:
            raise ValueError(f"bad manifest line: {raw!r}")
        pid, title, urls = parts[:3]
        rows.append({
            "id": int(pid),
            "title": title.strip(),
            "urls": [u.strip() for u in urls.split(";;") if u.strip()],
            "category": category,
            "stem": f"{int(pid):02d}_" + re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:150],
        })
    return rows

def normalize_title(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def title_score(a: str, b: str) -> float:
    aa, bb = set(normalize_title(a).split()), set(normalize_title(b).split())
    return len(aa & bb) / max(1, len(aa | bb))

def get(url, timeout=60):
    return S.get(url, timeout=timeout, allow_redirects=True)

def is_pdf_response(r):
    ct = (r.headers.get("content-type") or "").lower()
    return r.content[:5] == b"%PDF-" or "application/pdf" in ct

def html_candidates(base, html):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").lower()
        c = m.get("content")
        if c and ("pdf" in name or name in ("citation_pdf_url", "wkhealth_pdf_url")):
            out.append(urljoin(base, c))
    for a in soup.find_all("a", href=True):
        h = urljoin(base, a["href"])
        txt = a.get_text(" ", strip=True).lower()
        if ".pdf" in h.lower() or "download pdf" in txt or txt.strip() == "pdf":
            out.append(h)
    if "nature.com/articles/" in base and not base.rstrip("/").endswith(".pdf"):
        out.append(base.split("?")[0].rstrip("/") + ".pdf")
    if "sciencedirect.com/science/article/pii/" in base and "/pdfft" not in base:
        out.append(base.split("?")[0].rstrip("/") + "/pdfft?isDTMRedir=true&download=true")
    return list(dict.fromkeys(out))

def openalex_candidates(title):
    out = []
    try:
        r = get("https://api.openalex.org/works?search=" + quote(title) + "&per-page=5", 30)
        if r.ok:
            for w in r.json().get("results", []):
                if title_score(title, w.get("title", "")) < 0.55:
                    continue
                locs = [w.get("best_oa_location"), w.get("primary_location")] + list(w.get("locations") or [])
                for loc in locs:
                    if not loc:
                        continue
                    for k in ("pdf_url", "landing_page_url"):
                        if loc.get(k):
                            out.append(loc[k])
    except Exception:
        pass
    return list(dict.fromkeys(out))

def semanticscholar_candidates(title):
    out = []
    try:
        u = (
            "https://api.semanticscholar.org/graph/v1/paper/search?query="
            + quote(title)
            + "&limit=5&fields=title,openAccessPdf,url,externalIds"
        )
        r = get(u, 30)
        if r.ok:
            for w in r.json().get("data", []):
                if title_score(title, w.get("title", "")) < 0.55:
                    continue
                oa = w.get("openAccessPdf") or {}
                if oa.get("url"):
                    out.append(oa["url"])
                arx = (w.get("externalIds") or {}).get("ArXiv")
                if arx:
                    out.append(f"https://arxiv.org/pdf/{arx}.pdf")
    except Exception:
        pass
    return list(dict.fromkeys(out))

def crossref_candidates(title):
    out = []
    try:
        r = get("https://api.crossref.org/works?query.title=" + quote(title) + "&rows=5", 30)
        if r.ok:
            for w in r.json().get("message", {}).get("items", []):
                wt = (w.get("title") or [""])[0]
                if title_score(title, wt) < 0.55:
                    continue
                for link in w.get("link") or []:
                    u = link.get("URL")
                    ct = (link.get("content-type") or "").lower()
                    if u and ("pdf" in ct or ".pdf" in u.lower()):
                        out.append(u)
    except Exception:
        pass
    return list(dict.fromkeys(out))

def validate_pdf_bytes(data: bytes) -> bool:
    if not data.startswith(b"%PDF-") or len(data) < 5000:
        return False
    try:
        import fitz
        doc = fitz.open(stream=data, filetype="pdf")
        ok = doc.page_count > 0
        doc.close()
        return ok
    except Exception:
        return False

def download_pdf(row, dst):
    tried = []
    queue = list(row["urls"])
    queue += openalex_candidates(row["title"])
    queue += semanticscholar_candidates(row["title"])
    queue += crossref_candidates(row["title"])
    seen = set()
    while queue:
        u = queue.pop(0)
        if not u or u in seen:
            continue
        seen.add(u)
        tried.append(u)
        try:
            r = get(u, 75)
            if r.ok and is_pdf_response(r) and validate_pdf_bytes(r.content):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(r.content)
                return True, r.url, len(r.content), tried
            ct = (r.headers.get("content-type") or "").lower()
            if r.ok and ("html" in ct or r.text[:200].lstrip().startswith("<")):
                queue += [x for x in html_candidates(r.url, r.text) if x not in seen]
        except Exception as e:
            print("download candidate failed", u, repr(e), file=sys.stderr)
    return False, None, 0, tried

def pdf_to_markdown(pdf: Path, row):
    header = (
        f"# {row['title']}\n\n"
        f"- 合并类别: {row['category']}\n"
        f"- PDF 来源: {row['urls'][0]}\n\n---\n\n"
    )
    try:
        import pymupdf4llm
        body = pymupdf4llm.to_markdown(str(pdf), show_progress=False)
        if body and len(body) > 500:
            return header + body
    except Exception as e:
        print("pymupdf4llm fallback", repr(e), file=sys.stderr)
    import fitz
    doc = fitz.open(str(pdf))
    parts = []
    for i, p in enumerate(doc, 1):
        parts.append(f"\n\n<!-- page {i} -->\n\n" + p.get_text("text"))
    doc.close()
    return header + "".join(parts)

def protect(text):
    saved = []
    def repl(m):
        key = f"__PROTECTED_{len(saved):04d}__"
        saved.append(m.group(0))
        return key
    return PROTECTED_RE.sub(repl, text), saved

def restore(text, saved):
    for i, v in enumerate(saved):
        text = text.replace(f"__PROTECTED_{i:04d}__", v)
    return text

def google_translate(text, retries=7):
    if not text.strip():
        return text
    protected, saved = protect(text)
    params = {"client": "gtx", "sl": "en", "tl": "zh-CN", "dt": "t", "q": protected}
    last = None
    for k in range(retries):
        try:
            r = S.get("https://translate.googleapis.com/translate_a/single", params=params, timeout=60)
            if r.ok:
                data = r.json()
                result = "".join(x[0] for x in data[0] if x and x[0] is not None)
                return restore(result, saved)
            last = f"{r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(20, 1.7 * (2 ** k)))
    raise RuntimeError(last or "translation failed")

def split_text(s, max_chars=2600):
    if len(s) <= max_chars:
        return [s]
    pieces, cur = [], ""
    parts = re.split(r"(?<=\n\n)|(?<=[.!?。！？])\s+", s)
    for part in parts:
        if len(cur) + len(part) > max_chars and cur:
            pieces.append(cur)
            cur = ""
        if len(part) > max_chars:
            for i in range(0, len(part), max_chars):
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.append(part[i:i + max_chars])
        else:
            cur += part
    if cur:
        pieces.append(cur)
    return pieces

def should_skip_block(block):
    s = block.strip()
    if not s:
        return True
    if s.startswith(FENCE) and s.endswith(FENCE):
        return True
    alpha = sum(c.isalpha() for c in s)
    if alpha / max(1, len(s)) < 0.18:
        return True
    return False

def translate_markdown(md):
    blocks = re.split(r"(\n\s*\n)", md)
    out = list(blocks)
    translatable = []
    in_code = False
    seen_reference_heading = False
    for idx, block in enumerate(blocks):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith(FENCE):
            if stripped.count(FENCE) % 2 == 1:
                in_code = not in_code
            continue
        if re.match(r"^#{1,6}\s+(references|bibliography)\s*$", stripped, re.I):
            seen_reference_heading = True
            continue
        if seen_reference_heading or in_code or should_skip_block(block):
            continue
        m = re.match(r"^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)?)([\s\S]*)$", block)
        translatable.append((idx, m.group(1), m.group(2)))

    failures = 0
    pos = 0
    while pos < len(translatable):
        batch = []
        size = 0
        while pos < len(translatable):
            idx, prefix, core = translatable[pos]
            marker = f"ZXQBLOCK{idx:05d}QXZ"
            addition = marker + "\n" + core + "\n"
            if batch and size + len(addition) > 4200:
                break
            batch.append((idx, prefix, core, marker))
            size += len(addition)
            pos += 1
        packed = "\n".join(marker + "\n" + core for idx, prefix, core, marker in batch)
        try:
            translated = google_translate(packed)
            positions = []
            for idx, prefix, core, marker in batch:
                p = translated.find(marker)
                if p < 0:
                    raise ValueError("batch marker missing")
                positions.append((p, idx, prefix, marker))
            positions.sort()
            for k, (p, idx, prefix, marker) in enumerate(positions):
                s = p + len(marker)
                e = positions[k + 1][0] if k + 1 < len(positions) else len(translated)
                core_zh = translated[s:e].strip("\n ")
                out[idx] = prefix + core_zh
        except Exception as e:
            print("batch translate failed; falling back", repr(e), file=sys.stderr)
            for idx, prefix, core, marker in batch:
                try:
                    out[idx] = prefix + "".join(google_translate(x) for x in split_text(core))
                except Exception as e2:
                    failures += 1
                    print("translate block failed", repr(e2), file=sys.stderr)
        time.sleep(0.08)
    return "".join(out), failures

def chinese_ratio(text):
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    return round(zh / max(1, zh + en), 4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    args = ap.parse_args()
    rows = read_rows(args.category)
    out = Path(os.environ.get("OUTPUT_ROOT", "output")) / args.category
    pdfdir, endir, zhdir = out / "pdf", out / "md_en", out / "md_zh"
    for p in (pdfdir, endir, zhdir):
        p.mkdir(parents=True, exist_ok=True)
    report = []
    for n, row in enumerate(rows, 1):
        print(f"[{args.category}] {n}/{len(rows)} #{row['id']} {row['title']}", flush=True)
        pdf = pdfdir / (row["stem"] + ".pdf")
        en = endir / (row["stem"] + ".md")
        zh = zhdir / (row["stem"] + "_zh.md")
        rec = {
            "id": row["id"], "title": row["title"], "category": args.category,
            "pdf": "", "md_en": "", "md_zh": "", "source_used": "",
            "bytes": 0, "zh_ratio": 0, "tried": []
        }
        ok, src, size, tried = download_pdf(row, pdf)
        rec["tried"] = tried
        if ok:
            rec["pdf"], rec["source_used"], rec["bytes"] = "ok", src, size
            try:
                md = pdf_to_markdown(pdf, row)
                en.write_text(md, encoding="utf-8")
                rec["md_en"] = "ok"
            except Exception as e:
                rec["md_en"] = "ERROR " + repr(e)
                md = ""
            if md:
                try:
                    translated, failures = translate_markdown(md)
                    zh.write_text(translated, encoding="utf-8")
                    ratio = chinese_ratio(translated)
                    rec["zh_ratio"] = ratio
                    rec["md_zh"] = "ok" if failures == 0 and ratio >= 0.20 else f"PARTIAL failures={failures} ratio={ratio}"
                except Exception as e:
                    rec["md_zh"] = "ERROR " + repr(e)
        else:
            rec["pdf"] = "FAILED"
        report.append(rec)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
