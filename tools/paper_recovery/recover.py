#!/usr/bin/env python3
from __future__ import annotations
import asyncio, json, os, re, subprocess, sys, time
from pathlib import Path
from urllib.parse import urljoin
import requests

FENCE = chr(96) * 3
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
S = requests.Session()
S.headers.update({
    "User-Agent": UA,
    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})

TARGETS = {
    1: {
        "title": "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second",
        "category": "01_TFM_PFN",
        "urls": ["https://arxiv.org/pdf/2207.01848.pdf"],
    },
    18: {
        "title": "Table Foundation Models: on Knowledge Pre-training for Tabular Learning",
        "category": "02_CrossTable_Multimodal_Semantic",
        "urls": [
            "https://arxiv.org/pdf/2505.14415.pdf",
            "https://openreview.net/pdf?id=QV4P8Csw17",
        ],
    },
    24: {
        "title": "ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data",
        "category": "03_Unsupervised_Clustering_FM",
        "urls": ["https://arxiv.org/pdf/2505.10704.pdf"],
    },
    46: {
        "title": "Dual Mutual Information Constraints for Discriminative Clustering",
        "category": "04_Deep_Clustering_Local_Graph_Topology",
        "landing": "https://ojs.aaai.org/index.php/AAAI/article/view/26032",
        "urls": [
            "https://ojs.aaai.org/index.php/AAAI/article/download/26032/25804",
            "https://ojs.aaai.org/index.php/AAAI/article/download/26032/25804?download=1",
        ],
    },
    57: {
        "title": "CTSyn: A Foundation Model for Cross Tabular Data Generation",
        "category": "06_Tabular_Generation_Schema",
        "urls": ["https://arxiv.org/pdf/2406.04619.pdf"],
    },
    60: {
        "title": "Efficient Table Generation for Zero-Shot Column Type Annotation",
        "category": "06_Tabular_Generation_Schema",
        "landing": "https://www.researchgate.net/publication/400771892_Efficient_Table_Generation_for_Zero-Shot_Column_Type_Annotation",
        "urls": [
            "https://www.researchgate.net/publication/profile/Ehsan-Hoseinzade/publication/400771892_Efficient_Table_Generation_for_Zero-Shot_Column_Type_Annotation/links/698f72bd5d60ab48356db524/Efficient-Table-Generation-for-Zero-Shot-Column-Type-Annotation.pdf",
            "https://openreview.net/attachment?id=vgHqweeBMb&name=pdf",
            "https://openreview.net/pdf?id=vgHqweeBMb",
        ],
    },
    62: {
        "title": "Generating Realistic Synthetic Tabular Data with Integrated LLM and Diffusion Models",
        "category": "06_Tabular_Generation_Schema",
        "landing": "https://www.sciencedirect.com/science/article/pii/S0925231225020430",
        "urls": [
            "https://www.sciencedirect.com/science/article/pii/S0925231225020430/pdfft?isDTMRedir=true&download=true",
            "https://www.sciencedirect.com/science/article/pii/S0925231225020430/pdfft?download=true",
            "https://www.sciencedirect.com/science/article/pii/S0925231225020430/pdf",
        ],
    },
    70: {
        "title": "MedTransTab: Advancing Medical Cross-Table Tabular Data Generation",
        "category": "06_Tabular_Generation_Schema",
        "urls": [
            "https://arxiv.org/pdf/2503.01691.pdf",
            "https://dl.acm.org/doi/pdf/10.1145/3701551.3703501",
        ],
    },
    71: {
        "title": "Representation Learning for Tabular Data: A Comprehensive Survey",
        "category": "08_Surveys",
        "urls": ["https://arxiv.org/pdf/2504.16109.pdf"],
    },
    72: {
        "title": "The Current Generation of Tabular Foundation Models: A Critical Review",
        "category": "08_Surveys",
        "landing": "https://www.mdpi.com/2504-4990/8/8/244",
        "urls": [
            "https://www.mdpi.com/2504-4990/8/8/244/pdf-vor",
            "https://www.mdpi.com/2504-4990/8/8/244/pdf",
        ],
    },
}

def safe_stem(pid, title):
    return f"{pid:02d}_" + re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:150]

def validate_pdf(data):
    if not data or not data.startswith(b"%PDF-") or len(data) < 5000:
        return False
    try:
        import fitz
        d = fitz.open(stream=data, filetype="pdf")
        ok = d.page_count > 0
        d.close()
        return ok
    except Exception:
        return False

def save_if_pdf(data, dst):
    if validate_pdf(data):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return True
    return False

def requests_try(url, landing=None):
    headers = {}
    if landing:
        headers["Referer"] = landing
    try:
        r = S.get(url, headers=headers, timeout=45, allow_redirects=True)
        print("requests", r.status_code, r.url, r.headers.get("content-type"), len(r.content), flush=True)
        if validate_pdf(r.content):
            return r.content, r.url
    except Exception as e:
        print("requests error", url, repr(e), flush=True)
    return None, None

def curl_try(url, landing=None):
    tmp = Path("/tmp/curl_candidate.pdf")
    try:
        tmp.unlink(missing_ok=True)
        cmd = [
            "curl", "-L", "--compressed", "--fail-with-body", "--retry", "2",
            "--connect-timeout", "20", "--max-time", "90",
            "-A", UA,
            "-H", "Accept: application/pdf,text/html;q=0.9,*/*;q=0.8",
            "-H", "Accept-Language: en-US,en;q=0.9",
        ]
        if landing:
            cmd += ["-e", landing]
        cmd += ["-o", str(tmp), url]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if tmp.exists():
            data = tmp.read_bytes()
            print("curl", p.returncode, url, len(data), p.stderr[-500:], flush=True)
            if validate_pdf(data):
                return data, url
    except Exception as e:
        print("curl error", url, repr(e), flush=True)
    return None, None

async def browser_try(target):
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        print("playwright import failed", repr(e), flush=True)
        return None, None

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            user_agent=UA,
            locale="en-US",
            accept_downloads=True,
            viewport={"width": 1365, "height": 900},
        )
        page = await context.new_page()
        landing = target.get("landing")
        if landing:
            try:
                print("browser landing", landing, flush=True)
                await page.goto(landing, wait_until="domcontentloaded", timeout=90000)
                await page.wait_for_timeout(5000)
            except Exception as e:
                print("landing error", repr(e), flush=True)

        candidates = list(target["urls"])
        if landing:
            try:
                hrefs = await page.locator("a").evaluate_all(
                    "(els) => els.map(e => e.href).filter(Boolean)"
                )
                for h in hrefs:
                    hl = h.lower()
                    if ("pdf" in hl or "pdfft" in hl or "download" in hl) and h not in candidates:
                        candidates.append(h)
            except Exception as e:
                print("link scan error", repr(e), flush=True)

        for url in candidates:
            # First reuse browser cookies through Playwright's request context.
            try:
                resp = await context.request.get(
                    url,
                    headers={"Referer": landing or url, "Accept": "application/pdf,*/*"},
                    timeout=90000,
                    fail_on_status_code=False,
                )
                body = await resp.body()
                print("browser request", resp.status, url, resp.headers.get("content-type"), len(body), flush=True)
                if validate_pdf(body):
                    await browser.close()
                    return body, str(resp.url)
            except Exception as e:
                print("browser request error", url, repr(e), flush=True)

            # Then attempt browser navigation and capture the response body.
            try:
                resp = await page.goto(url, wait_until="commit", timeout=90000)
                if resp:
                    await page.wait_for_timeout(3000)
                    body = await resp.body()
                    print("browser goto", resp.status, url, resp.headers.get("content-type"), len(body), flush=True)
                    if validate_pdf(body):
                        await browser.close()
                        return body, str(resp.url)
            except Exception as e:
                print("browser goto error", url, repr(e), flush=True)

        await browser.close()
    return None, None

def download_target(pid, target, dst):
    tried = []
    landing = target.get("landing")
    for url in target["urls"]:
        tried.append(("requests", url))
        data, used = requests_try(url, landing)
        if data and save_if_pdf(data, dst):
            return True, used, tried
        tried.append(("curl", url))
        data, used = curl_try(url, landing)
        if data and save_if_pdf(data, dst):
            return True, used, tried

    print("falling back to playwright for", pid, flush=True)
    data, used = asyncio.run(browser_try(target))
    tried.append(("playwright", used or "failed"))
    if data and save_if_pdf(data, dst):
        return True, used, tried
    return False, None, tried

def pdf_to_markdown(pdf, target, source_used):
    header = (
        f"# {target['title']}\n\n"
        f"- 合并类别: {target['category']}\n"
        f"- PDF 来源: {source_used or target['urls'][0]}\n\n---\n\n"
    )
    try:
        import pymupdf4llm
        body = pymupdf4llm.to_markdown(str(pdf), show_progress=False)
        if body and len(body) > 500:
            return header + body
    except Exception as e:
        print("pymupdf4llm fallback", repr(e), flush=True)
    import fitz
    doc = fitz.open(str(pdf))
    parts = []
    for i, p in enumerate(doc, 1):
        parts.append(f"\n\n<!-- page {i} -->\n\n" + p.get_text("text"))
    doc.close()
    return header + "".join(parts)

PROTECTED_RE = re.compile(
    r'(https?://\S+|doi:\s*\S+|\x60[^\x60\n]+\x60|\$\$.*?\$\$|\$[^\n$]+\$|\\\([^\n]*?\\\)|\\\[[\s\S]*?\\\])',
    re.I,
)

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

def google_translate(text, retries=10):
    if not text.strip():
        return text
    protected, saved = protect(text)
    params = {"client": "gtx", "sl": "en", "tl": "zh-CN", "dt": "t", "q": protected}
    last = None
    for k in range(retries):
        try:
            r = S.get("https://translate.googleapis.com/translate_a/single", params=params, timeout=90)
            if r.ok:
                data = r.json()
                out = "".join(x[0] for x in data[0] if x and x[0] is not None)
                return restore(out, saved)
            last = f"{r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(45, 2 + 2.0 * k))
    raise RuntimeError(last or "translation failed")

def should_skip(block):
    s = block.strip()
    if not s:
        return True
    if s.startswith(FENCE) and s.endswith(FENCE):
        return True
    alpha = sum(c.isalpha() for c in s)
    return alpha / max(1, len(s)) < 0.18

def translate_markdown(md):
    blocks = re.split(r"(\n\s*\n)", md)
    out = list(blocks)
    items = []
    in_code = False
    references = False
    for idx, block in enumerate(blocks):
        s = block.strip()
        if not s:
            continue
        if s.startswith(FENCE):
            if s.count(FENCE) % 2:
                in_code = not in_code
            continue
        if re.match(r"^#{1,6}\s+(references|bibliography)\s*$", s, re.I):
            references = True
            continue
        if references or in_code or should_skip(block):
            continue
        m = re.match(r"^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)?)([\s\S]*)$", block)
        items.append((idx, m.group(1), m.group(2)))

    failures = 0
    pos = 0
    while pos < len(items):
        batch = []
        chars = 0
        while pos < len(items):
            idx, prefix, core = items[pos]
            marker = f"ZXQBLOCK{idx:05d}QXZ"
            add = marker + "\n" + core + "\n"
            if batch and chars + len(add) > 3200:
                break
            batch.append((idx, prefix, core, marker))
            chars += len(add)
            pos += 1

        packed = "\n".join(marker + "\n" + core for idx, prefix, core, marker in batch)
        try:
            translated = google_translate(packed)
            positions = []
            for idx, prefix, core, marker in batch:
                p = translated.find(marker)
                if p < 0:
                    raise ValueError("marker missing")
                positions.append((p, idx, prefix, marker))
            positions.sort()
            for k, (p, idx, prefix, marker) in enumerate(positions):
                st = p + len(marker)
                en = positions[k + 1][0] if k + 1 < len(positions) else len(translated)
                out[idx] = prefix + translated[st:en].strip("\n ")
        except Exception as e:
            print("batch translation fallback", repr(e), flush=True)
            for idx, prefix, core, marker in batch:
                try:
                    out[idx] = prefix + google_translate(core)
                except Exception as e2:
                    failures += 1
                    print("block translation failed", idx, repr(e2), flush=True)
        time.sleep(0.6)
    return "".join(out), failures

def chinese_ratio(text):
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    return round(zh / max(1, zh + en), 4)

def main():
    out = Path("recovery_output")
    out.mkdir(exist_ok=True)
    report = []
    for pid, target in TARGETS.items():
        stem = safe_stem(pid, target["title"])
        base = out / target["category"]
        pdf = base / "pdf" / f"{stem}.pdf"
        en = base / "md_en" / f"{stem}.md"
        zh = base / "md_zh" / f"{stem}_zh.md"
        for p in (pdf.parent, en.parent, zh.parent):
            p.mkdir(parents=True, exist_ok=True)

        rec = {
            "id": pid, "title": target["title"], "category": target["category"],
            "pdf": "", "md_en": "", "md_zh": "", "source_used": "",
            "bytes": 0, "zh_ratio": 0, "tried": []
        }
        print("\n### RECOVER", pid, target["title"], flush=True)
        ok, src, tried = download_target(pid, target, pdf)
        rec["tried"] = tried
        if not ok:
            rec["pdf"] = "FAILED"
            report.append(rec)
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        rec["pdf"] = "ok"
        rec["source_used"] = src or target["urls"][0]
        rec["bytes"] = pdf.stat().st_size
        try:
            md = pdf_to_markdown(pdf, target, rec["source_used"])
            en.write_text(md, encoding="utf-8")
            rec["md_en"] = "ok"
        except Exception as e:
            rec["md_en"] = "ERROR " + repr(e)
            md = ""

        if md:
            try:
                zhtext, failures = translate_markdown(md)
                zh.write_text(zhtext, encoding="utf-8")
                ratio = chinese_ratio(zhtext)
                rec["zh_ratio"] = ratio
                rec["md_zh"] = "ok" if failures == 0 and ratio >= 0.20 else f"PARTIAL failures={failures} ratio={ratio}"
            except Exception as e:
                rec["md_zh"] = "ERROR " + repr(e)

        report.append(rec)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = [r for r in report if r["pdf"] != "ok" or r["md_en"] != "ok" or r["md_zh"] != "ok"]
    print(json.dumps({"total": len(report), "complete": len(report)-len(failed), "failed": failed}, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
