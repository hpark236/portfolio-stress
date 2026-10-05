"""Nancy Pelosi's periodic transaction reports (PTRs) from the House Clerk.

The Clerk publishes a yearly index of every financial disclosure filing
(https://disclosures-clerk.house.gov/public_disc/financial-pdfs/<year>FD.zip) and each PTR as a
typed PDF. This module downloads the index, finds the member's PTRs, converts each PDF to text
with pdftotext and parses the transaction table.

Output rows: {doc, filed, owner, asset, ticker, kind, type, partial, traded, notified,
amount_lo, amount_hi, shares, detail}
"""

import io
import json
import re
import ssl
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "ptr"
BASE = "https://disclosures-clerk.house.gov/public_disc"
UA = {"User-Agent": "portfolio-stress research (github.com/hpark236/portfolio-stress)"}


try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
        return r.read()


def filings(last="Pelosi", first="Nancy", years=range(2019, date.today().year + 1)):
    """Every PTR the member filed, oldest first: [(doc_id, year, filed_date)]."""
    out = []
    for y in years:
        z = zipfile.ZipFile(io.BytesIO(_get(f"{BASE}/financial-pdfs/{y}FD.zip")))
        root = ET.fromstring(z.read(f"{y}FD.xml"))
        for m in root:
            row = {c.tag: (c.text or "").strip() for c in m}
            if row.get("Last", "").lower() == last.lower() and row.get("First", "").lower().startswith(first.lower()[:5]) and row.get("FilingType") == "P":
                out.append((row["DocID"], y, datetime.strptime(row["FilingDate"], "%m/%d/%Y").date()))
    return sorted(out, key=lambda r: r[2])


def pdf_text(doc_id, year):
    CACHE.mkdir(parents=True, exist_ok=True)
    pdf = CACHE / f"{doc_id}.pdf"
    if not pdf.exists():
        pdf.write_bytes(_get(f"{BASE}/ptr-pdfs/{year}/{doc_id}.pdf"))
    return subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout


# A transaction row: optional owner code, asset text, type (P, S, S (partial), E), two dates, amount.
# Older filings use a small-caps font that pdftotext returns in mixed case, so text is uppercased first.
ROW = re.compile(
    r"^(?P<lead>[ \t]*)(?P<owner>SP|JT|DC)?[ \t]+(?P<asset>\S[^\n]*?)[ \t]{2,}(?P<type>P|S \(PARTIAL\)|S|E)[ \t]+"
    r"(?P<traded>\d{2}/\d{2}/\d{4})[ \t]+(?P<notified>\d{2}/\d{2}/\d{4})[ \t]+(?P<amount>\$[\d,]+(?:[ \t]*-[ \t]*\$?[\d,]*)?|OVER \$[\d,]+)",
    re.M,
)
TICKER = re.compile(r"\(([A-Z][A-Z.]{0,5})\)")
KIND = re.compile(r"\[([A-Z]{2})\]")
DESC = re.compile(r"^\s*D(?:ESCRIPTION)?\s*:\s*(.+)", re.M)
NUM = lambda s: int(s.replace(",", "").replace("$", "")) if s else None


def parse(text, doc_id, filed):
    text = text.upper()
    rows = []
    matches = list(ROW.finditer(text))
    for i, m in enumerate(matches):
        block = text[m.start(): matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        lines = block.splitlines()
        a0, t0 = m.start("asset") - m.start(), m.start("type") - m.start()
        # the asset name and ticker can wrap onto the next lines, inside the asset column
        cont = []
        for line in lines[1:6]:
            if re.match(r"^\s*(F(ILING)?\s+S|D(ESCRIPTION)?\s*:)", line):
                break
            seg = line[a0:t0].strip()
            if seg and not re.fullmatch(r"[A-Z]", seg):
                cont.append(seg)
        asset = " ".join([m["asset"].strip(), *cont])
        nums = re.findall(r"[\d,]{3,}", m["amount"])
        if not nums:
            continue
        lo = NUM(nums[0])
        hi = NUM(nums[1]) if len(nums) > 1 else None
        if hi is None:  # the upper bound wraps onto a later line, at the end of the amount column
            for line in lines[1:9]:
                w = re.search(r"\$([\d,]{3,})\s*(?:[A-Z]\s*)?$", line)
                if w:
                    hi = NUM(w.group(1)); break
        desc = DESC.search(block)
        detail = re.sub(r"\s+", " ", desc.group(1)).strip() if desc else ""
        shares = re.search(r"(?:PURCHASED|SOLD|EXERCISED|RECEIVED|GIFTED|DONATED)\s+([\d,]+)\s+(?:SHARES|CALL OPTIONS|PUT OPTIONS)", detail)
        t = TICKER.search(asset)
        k = KIND.search(asset)
        rows.append({
            "doc": doc_id, "filed": filed.isoformat(),
            "owner": m["owner"] or "", "asset": re.sub(r"\s*\[[A-Z]{2}\]", "", asset).strip(),
            "ticker": t.group(1) if t else None, "kind": k.group(1) if k else None,
            "type": "S" if m["type"].startswith("S") else m["type"], "partial": "PARTIAL" in m["type"],
            "traded": datetime.strptime(m["traded"], "%m/%d/%Y").date().isoformat(),
            "notified": datetime.strptime(m["notified"], "%m/%d/%Y").date().isoformat(),
            "amount_lo": lo, "amount_hi": hi,
            "shares": NUM(shares.group(1)) if shares else None,
            "options": "OPTION" in detail or (k is not None and k.group(1) == "OP"),
            "detail": detail.capitalize(),
        })
    return rows


def all_trades():
    out = []
    for doc, year, filed in filings():
        try:
            out.extend(parse(pdf_text(doc, year), doc, filed))
        except Exception as e:  # scanned or malformed filings are skipped, and counted
            out.append({"doc": doc, "filed": filed.isoformat(), "error": str(e)})
    return out


if __name__ == "__main__":
    t = all_trades()
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "pelosi_trades.json").write_text(json.dumps(t, indent=1))
    ok = [r for r in t if "error" not in r]
    print(len(ok), "transactions,", sum(1 for r in ok if r["ticker"]), "with tickers,", len(t) - len(ok), "errors")
    for r in ok[-8:]:
        print(r["filed"], r["traded"], r["type"], r["ticker"], r["kind"], r["amount_lo"], r["amount_hi"], r["shares"], r["detail"][:70])
