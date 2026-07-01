import streamlit as st
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
import subprocess
import tempfile
import os
import base64
import re
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, Image as RLImage
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from PIL import Image as PILImage

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Glass Atlas · AI Visibility Audit",
    page_icon="🔍",
    layout="wide",
)

# ─── Glass Atlas brand colours (from Brand Book 2025) ─────────────────────────
SUMMIT_RED   = "#F2BC63"   # Yellow Gold (primary accent)
SUMMIT_DARK  = "#25242B"   # Charcoal
SUMMIT_GREY  = "#6B6B6B"   # neutral grey (kept)
SUMMIT_LIGHT = "#F9F5F1"   # Paper White
WHITE        = "#FFFFFF"

LOGO_PATH = os.path.join(os.path.dirname(__file__), "glassatlas_badge.png")

# ─── Gemini setup ─────────────────────────────────────────────────────────────
def get_gemini_client():
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        st.error("GEMINI_API_KEY not found. Add it to Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

# ─── Fetch URL HTML ───────────────────────────────────────────────────────────
def extract_page_signals(url: str, html: str) -> str:
    """
    Compress a raw HTML page into a compact signal summary for Gemini.
    Instead of sending thousands of tokens of raw HTML, we extract exactly
    what an AI auditor needs: meta, schema, headings, links, alt text, ARIA.
    This keeps each page under ~2000 tokens while preserving all audit signals.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "path"]):
        tag.decompose()

    out = [f"URL: {url}"]

    # ── Meta & Open Graph ──────────────────────────────────────────────
    meta_items = []
    title = soup.find("title")
    if title: meta_items.append(f"title: {title.get_text().strip()[:120]}")
    for m in soup.find_all("meta"):
        name = m.get("name","") or m.get("property","")
        val  = m.get("content","")
        if name and val and name.lower() in [
            "description","robots","author","article:author",
            "article:published_time","article:modified_time",
            "og:title","og:type","og:description","og:url","og:image",
            "og:locale","og:site_name","twitter:card","twitter:title",
            "twitter:description","viewport"
        ]:
            meta_items.append(f"{name}: {val[:120]}")
    link_tags = []
    for l in soup.find_all("link", rel=True):
        rel = " ".join(l.get("rel",[]))
        if rel in ["canonical","alternate"]:
            link_tags.append(f"<link rel={rel} href={l.get('href','')[:80]}>")
    out.append("META:\n" + "\n".join(meta_items[:20] + link_tags[:5]))

    # ── Schema / JSON-LD ──────────────────────────────────────────────
    schemas = []
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        txt = s.string or s.get_text()
        if txt and txt.strip():
            schemas.append(txt.strip()[:500])
    # Fallback: regex search raw HTML in case BeautifulSoup missed it
    if not schemas:
        import re as _re
        raw_schemas = _re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, _re.DOTALL | _re.IGNORECASE
        )
        schemas = [s.strip()[:500] for s in raw_schemas if s.strip()]
    out.append("SCHEMA_JSON_LD: " + ("; ".join(schemas) if schemas else "NONE FOUND"))

    # ── Heading structure ─────────────────────────────────────────────
    headings = []
    for tag in soup.find_all(["h1","h2","h3","h4"]):
        txt = tag.get_text(" ", strip=True)[:80]
        if txt:
            headings.append(f"<{tag.name}>{txt}</{tag.name}>")
        if len(headings) >= 20: break
    out.append("HEADINGS:\n" + ("\n".join(headings) if headings else "NONE FOUND"))

    # ── ARIA usage ────────────────────────────────────────────────────
    aria_items = []
    for tag in soup.find_all(True):
        role  = tag.get("role","")
        label = tag.get("aria-label","") or tag.get("aria-labelledby","")
        curr  = tag.get("aria-current","")
        if role:   aria_items.append(f"role={role}")
        if label:  aria_items.append(f"aria-label={label[:60]}")
        if curr:   aria_items.append(f"aria-current={curr}")
        if len(aria_items) >= 15: break
    out.append("ARIA: " + ("; ".join(aria_items) if aria_items else "NONE FOUND"))

    # ── Navigation / links sample ─────────────────────────────────────
    nav = soup.find("nav")
    all_links = soup.find_all("a", href=True)
    link_sample = []
    for a in all_links[:40]:
        href = a.get("href","")
        txt  = a.get_text(" ",strip=True)[:50]
        link_sample.append(f"{txt} -> {href[:80]}")
    out.append(f"LINKS (total found: {len(all_links)}):\n" + "\n".join(link_sample[:25]))

    # ── Images / alt text ─────────────────────────────────────────────
    imgs = soup.find_all("img")
    img_sample = []
    for img in imgs[:20]:
        alt = img.get("alt", "MISSING")
        src = img.get("src","")[:60]
        img_sample.append(f"alt={repr(alt)} src={src}")
    out.append(f"IMAGES (total: {len(imgs)}):\n" + "\n".join(img_sample))

    # ── Inline JS data stores (Next.js, Nuxt, etc.) ─────────────────
    # Even JS-heavy sites embed page data in __NEXT_DATA__ or similar
    next_data = soup.find("script", id="__NEXT_DATA__")
    if next_data and next_data.string:
        try:
            import json as _json
            nd = _json.loads(next_data.string)
            # Extract text-like values up to 600 chars
            def _extract_strings(obj, depth=0):
                if depth > 4: return []
                if isinstance(obj, str) and len(obj) > 20:
                    return [obj[:120]]
                if isinstance(obj, dict):
                    vals = []
                    for v in obj.values():
                        vals.extend(_extract_strings(v, depth+1))
                    return vals[:10]
                if isinstance(obj, list):
                    vals = []
                    for v in obj[:5]:
                        vals.extend(_extract_strings(v, depth+1))
                    return vals[:10]
                return []
            strings = _extract_strings(nd)
            if strings:
                out.append("NEXT_DATA_CONTENT:\n" + "\n".join(strings[:15]))
        except Exception:
            pass

    # ── Body content sample (for LLM signal) ─────────────────────────
    body = soup.find("body")
    if body:
        text = " ".join(body.get_text(" ", strip=True).split())[:1500]
        out.append(f"BODY_TEXT_SAMPLE:\n{text}")

    return "\n\n".join(out)


def _detect_block_or_empty(status: int, html: str, url: str) -> tuple[bool, str]:
    """
    Combined block and emptiness detector. Returns (is_bad, reason).
    Runs BEFORE any signal extraction. If this returns True, we do not score,
    we do not send to Gemini, we surface an error and prompt the user to paste
    raw HTML instead.
    """
    # Signal 1: HTTP status
    if status in (401, 403, 429, 503):
        return True, f"HTTP {status} response (likely bot block or rate limit)"

    # Signal 2: response too small to be a real page
    if len(html) < 2000:
        return True, f"Response body only {len(html)} bytes (likely a block or error page)"

    # Signal 3: known WAF and bot-challenge fingerprints
    snippet = html[:5000].lower()
    fingerprints = {
        "just a moment":              "Cloudflare bot challenge",
        "_cf_chl_":                   "Cloudflare bot challenge",
        "cf-browser-verification":    "Cloudflare bot challenge",
        "cf_chl_opt":                 "Cloudflare bot challenge",
        "checking your browser":      "Cloudflare bot challenge",
        "attention required":         "Cloudflare block",
        "ray id":                     "Cloudflare block page",
        "access denied":              "Access denied response",
        "403 forbidden":              "403 Forbidden response",
        "enable javascript and cookies to continue": "Bot challenge (JS+cookies required)",
        "ak-bmsc":                    "Akamai bot manager challenge",
        "reference #18.":             "Akamai bot manager challenge",
        "px-captcha":                 "PerimeterX challenge",
        "dd-protection":              "DataDome challenge",
        "incapsula":                  "Imperva Incapsula challenge",
        "distil_r_captcha":           "Distil Networks challenge",
    }
    for marker, description in fingerprints.items():
        if marker in snippet:
            return True, description

    # Signal 4: title heuristics
    _soup = BeautifulSoup(html, "html.parser")
    _title_tag = _soup.find("title")
    if _title_tag and _title_tag.string:
        _title = _title_tag.string.strip().lower()
        block_titles = [
            "just a moment", "access denied", "attention required",
            "please wait", "403 forbidden", "you have been blocked",
            "security check", "checking your browser",
        ]
        if any(bt in _title for bt in block_titles):
            return True, f"Blocking page title: {_title_tag.string.strip()[:80]}"

    # Signal 5: structural emptiness
    # A real page has headings, links, and paragraphs. Challenge pages do not.
    # This catches blocks even when no fingerprint matches, which is what
    # went wrong on Bettys.
    for t in _soup(["script", "style", "noscript"]):
        t.decompose()
    body = _soup.find("body")
    if body:
        h1_count   = len(body.find_all("h1"))
        h_count    = len(body.find_all(["h1", "h2", "h3"]))
        link_count = len(body.find_all("a", href=True))
        p_count    = len(body.find_all("p"))
        body_text  = body.get_text(" ", strip=True)
        if h_count == 0 and link_count < 3 and p_count < 3:
            return True, (
                f"Page appears empty (h1={h1_count}, headings={h_count}, "
                f"links={link_count}, paragraphs={p_count}, body chars={len(body_text)}). "
                "Likely a bot-block, WAF challenge, or JS-only shell."
            )

    return False, ""


def _is_js_shell(html: str) -> bool:
    """Return True if page has almost no body text but many script tags."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script","style","noscript","svg"]): t.decompose()
    body_text = soup.get_text(" ", strip=True)
    script_count = html.lower().count("<script")
    return len(body_text) < 300 and script_count > 3


def _fetch_direct(url: str, session: requests.Session) -> tuple:
    """Returns (html, status, source_label)."""
    r = session.get(url, timeout=15, allow_redirects=True)
    return r.text, r.status_code, "direct"


def _fetch_wayback(url: str, session: requests.Session) -> tuple:
    """
    Fetch via Wayback Machine (web.archive.org).
    Wayback crawls with Googlebot, so it bypasses most bot-protection
    and returns what search/AI crawlers actually saw.
    Returns (html, status, source_label) or raises on failure.
    """
    # Step 1: find the latest snapshot URL
    avail_api = f"https://archive.org/wayback/available?url={url}"
    meta = session.get(avail_api, timeout=10).json()
    snapshots = meta.get("archived_snapshots", {})
    closest   = snapshots.get("closest", {})
    snap_url  = closest.get("url", "")
    if not snap_url:
        raise ValueError("No Wayback snapshot found")
    # Convert to raw snapshot (remove Wayback toolbar injection)
    # id_ suffix returns the original page without Wayback toolbar
    snap_url = snap_url.replace("/web/", "/web/") 
    raw_url  = snap_url.replace("http://web.archive.org/web/",
                                 "https://web.archive.org/web/")
    # Insert 'id_' flag to get clean original HTML
    parts = raw_url.split("/web/", 1)
    if len(parts) == 2:
        ts_and_url = parts[1].split("/", 1)
        if len(ts_and_url) == 2:
            raw_url = f"https://web.archive.org/web/{ts_and_url[0]}id_/{ts_and_url[1]}"
    r = session.get(raw_url, timeout=20, allow_redirects=True)
    return r.text, r.status_code, f"Wayback Machine ({closest.get('timestamp','')})"


def fetch_single_page(url: str, session: requests.Session) -> tuple:
    """
    Multi-strategy fetcher. Returns (signals_text, fetch_note).
    Strategy: direct, then Wayback fallback. If both fail or return
    blocked/empty content, returns a FETCH_BLOCKED marker that the
    orchestrator surfaces to the user with a prompt to paste raw HTML.
    """
    fetch_note = ""
    html = ""
    label = ""
    block_reason = ""
    direct_error = ""

    # Strategy 1: direct fetch
    try:
        html, status, label = _fetch_direct(url, session)
        is_bad, reason = _detect_block_or_empty(status, html, url)
        if is_bad:
            block_reason = reason
            raise ValueError(reason)
    except Exception as e:
        direct_error = str(e)
        # Strategy 2: Wayback Machine
        try:
            html, status, label = _fetch_wayback(url, session)
            is_bad, reason = _detect_block_or_empty(status, html, url)
            if is_bad:
                # Wayback also returned junk. Give up and surface the block.
                return (
                    f"URL: {url}\n\n[FETCH_BLOCKED: {block_reason or direct_error}. "
                    f"Wayback fallback also returned unusable content: {reason}. "
                    "Paste the raw HTML for this URL using the 'Site blocked?' "
                    "expander below the URL inputs.]",
                    "blocked"
                )
            fetch_note = (
                f"SOURCE_NOTE: Direct fetch was blocked ({block_reason or direct_error}). "
                f"Content retrieved from {label}, reflects how AI crawlers "
                "that use cached versions see this page."
            )
        except Exception as e2:
            return (
                f"URL: {url}\n\n[FETCH_BLOCKED: Direct fetch failed ({direct_error}). "
                f"Wayback fetch also failed ({e2}). "
                "The site is either heavily bot-protected or unreachable. "
                "Paste the raw HTML for this URL using the 'Site blocked?' "
                "expander below the URL inputs.]",
                "blocked"
            )

    # Detect JS shell (still useful for legitimate React/Next/Vue sites)
    is_shell = _is_js_shell(html)
    signals  = extract_page_signals(url, html)

    if is_shell:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style","noscript","svg"]): t.decompose()
        body_len = len(soup.get_text(" ", strip=True))
        sc       = html.lower().count("<script")
        fetch_note += (
            f"\nJS_SHELL_NOTE: Page appears client-side rendered "
            f"(body text {body_len} chars, {sc} script tags). "
            "Score CRAWL 1-3. Other dimensions scored from shell content above."
        )

    if fetch_note:
        signals += "\n\n" + fetch_note

    return signals, label


def fetch_pages(domain: str, extra_urls: list[str],
                pasted_html: dict | None = None) -> dict[str, str]:
    """
    Fetch up to 4 pages. pasted_html is an optional {url: html} dict
    for manually pasted content (bypasses fetch entirely for those URLs).

    URL priority order:
    1. All pasted URLs (already have HTML, no fetch needed)
    2. The base domain homepage
    3. Any extra_urls typed in the text area
    Up to 4 total, deduped, preserving order.
    """
    base = domain.rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base

    # Build the master URL list — pasted URLs come first so they're
    # guaranteed a slot; then homepage; then any manually typed extras.
    seen = {}
    ordered_urls = []

    def add_url(u):
        u = u.strip().rstrip("/")
        if u and u not in seen:
            seen[u] = True
            ordered_urls.append(u)

    # 1. Pasted URLs (guaranteed to be processed)
    for u in (pasted_html or {}).keys():
        add_url(u)

    # 2. Homepage
    add_url(base)

    # 3. Extra typed URLs
    for u in extra_urls:
        add_url(u)

    # Cap at 4
    urls = ordered_urls[:4]

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9",
        "Cache-Control":   "no-cache",
    })

    pages = {}
    for url in urls:
        # Normalise for lookup (strip trailing slash)
        lookup = url.rstrip("/")
        pasted_match = None
        if pasted_html:
            for pu, ph in pasted_html.items():
                if pu.rstrip("/") == lookup:
                    pasted_match = ph
                    break

        if pasted_match:
            signals = extract_page_signals(url, pasted_match)
            signals += "\n\nSOURCE_NOTE: HTML was manually provided."
            pages[url] = signals
        else:
            signals, _label = fetch_single_page(url, session)
            pages[url] = signals

    return pages

# ─── Gemini audit ─────────────────────────────────────────────────────────────
AUDIT_PROMPT = """
You are an expert AI visibility auditor. Audit the provided HTML pages exactly like a senior technical SEO and AI readiness consultant would.

Score EACH page 1-10 across these 9 dimensions:
1. ARIA – landmark roles, aria-labels, accessibility for AI parsers
2. SCHEMA – schema.org JSON-LD structured data presence and quality
3. HEADINGS – H1-H6 hierarchy, clarity, topic signal
4. META – title, description, canonical, Open Graph, Twitter Card
5. LINKS – internal link quality, anchor text, protocol consistency, density
6. ALT TEXT – image alt attribute quality and completeness
7. CRAWL – server-rendered static HTML vs JS dependency
8. LLM – first-hand expertise, named entities, dates, citations, authority signals
9. CONTENT QUALITY – Score this dimension rigorously. Most commercial pages score 3-5, not 7-9. Use these specific criteria:
   SCORE 1-3 (Poor): Content is purely a list of features, specs, or product names with no explanation of why they matter to the buyer. No benefit statements. No answers to "why should I choose this?" or "what problem does this solve?". Thin content that simply labels things (e.g. "Thermostatic shower kit. Chrome finish. 200mm head.").
   SCORE 4-5 (Moderate): Some benefit language present but mostly feature-led. A few "what you get" statements but little "why it matters" context. No use case or buyer scenario addressed. Standard e-commerce copy that describes rather than sells.
   SCORE 6-7 (Good): Clearly benefit-led for most of the page. Addresses "what's in it for me" for the target buyer. Explains why features matter (e.g. not just "thermostatic" but "thermostatic control keeps water at a safe, consistent temperature"). Answers common pre-purchase questions.
   SCORE 8-9 (Excellent): Genuinely user-centric throughout. Addresses the buyer's specific situation, problem or goal. Uses outcome language ("achieve a spa-like experience at home", "save X", "eliminate Y problem"). Anticipates and answers specific buying questions. Cites proof points (reviews, stats, expert recommendation).
   SCORE 10: Reserved for editorial or guide content that is comprehensive, cites sources, names experts, and fully answers a user's question with no gaps.
   IMPORTANT: A product page that simply lists specifications and says "free delivery" scores NO HIGHER than 4. A category page that lists products with no explanatory copy scores 1-2.

CRITICAL RULES FOR JSON:
- Return ONLY raw JSON. No markdown, no ```json fences, no preamble, no explanation.
- All string values must use double quotes. Never use single quotes inside JSON strings.
- Escape any double quotes inside string values with a backslash.
- Do not include trailing commas after the last item in any array or object.
- Every string value must be on a single line with no literal newlines inside strings.

CRITICAL RULES FOR CONTENT:
- Write in British English throughout (use "optimise" not "optimize", "colour" not "color", "programme" not "program" etc.). Tone: clear-headed expert, calm authority, no jargon, no overpromising. Direct but never blunt. Explain, guide, challenge constructively. Anchor recommendations in business outcomes.
- Never use long dashes (em dashes or en dashes used as sentence breaks). Use a comma, colon, or rewrite the sentence instead.
- Never use the phrase "AI Visibility Practice".
- The executive_summary field MUST be structured with these exact pipe-delimited sections:
  OVERVIEW: 2-3 sentences on overall AI readiness. | STRENGTHS: up to 3 short bullet points of what is working (use * prefix). | GAPS: up to 3 short bullet points of the critical gaps (use * prefix). | VERDICT: 1 sentence on the single most impactful next step.
  Example: "OVERVIEW: Healix.com is built on strong technical foundations. AI crawlers receive full server-rendered HTML on first request. | STRENGTHS: * Comprehensive Open Graph and meta tags on every page. * Server-rendered HTML with no JS dependency. * Verifiable first-hand expertise with named clinicians and clients. | GAPS: * Zero schema.org structured data across all audited pages. * Tab and disclosure widgets are invisible to AI parsers. * og:type defaults to website on product pages. | VERDICT: Shipping Organisation, Service and BreadcrumbList schema sitewide is the single biggest unlock available."

Return ONLY valid JSON in exactly this structure:
{
  "company_name": "string",
  "domain": "string",
  "executive_summary": "3-4 paragraph string describing overall AI readiness",
  "average_score": number,
  "dimension_averages": {
    "aria": number, "schema": number, "headings": number, "meta": number,
    "links": number, "alt_text": number, "crawl": number, "llm": number, "content_quality": number
  },
  "pages": [
    {
      "url": "string",
      "title": "string",
      "score": number,
      "verdict": "string (1-2 sentences)",
      "headline_finding": "string (3-5 sentences)",
      "dimensions": {
        "aria": {"score": number, "detail": "string"},
        "schema": {"score": number, "detail": "string"},
        "headings": {"score": number, "detail": "string"},
        "meta": {"score": number, "detail": "string"},
        "links": {"score": number, "detail": "string"},
        "alt_text": {"score": number, "detail": "string"},
        "crawl": {"score": number, "detail": "string"},
        "llm": {"score": number, "detail": "string"},
        "content_quality": {"score": number, "detail": "string"}
      },
      "specific_findings": ["string", "string", "string"]
    }
  ],
  "cross_cutting_themes": [
    {"title": "string", "detail": "string (2-3 paragraphs)"}
  ],
  "recommendations": [
    {
      "priority": "P1|P2|P3",
      "action": "string",
      "impact": "string",
      "effort": "string",
      "owner": "string"
    }
  ],
  "three_quick_wins": [
    {"number": "1", "title": "string", "detail": "string"},
    {"number": "2", "title": "string", "detail": "string"},
    {"number": "3", "title": "string", "detail": "string"}
  ],
  "whats_working": [
    {"point": "string", "detail": "string"}
  ],
  "whats_holding_back": [
    {"point": "string", "detail": "string"},
    {"point": "string", "detail": "string"},
    {"point": "string", "detail": "string"}
  ]
}

IMPORTANT SCORING RULES:
- Each page "score" is the SUM of its 9 dimension scores (each 1-10), so the maximum is 90.
- "average_score" is the average of all page scores (i.e. average of those sums), so it is also out of 90. Do NOT return the average of dimension averages — that would give a number out of 10, which is wrong.
- Example: if one page scores aria:6, schema:2, headings:7, meta:8, links:6, alt_text:4, crawl:8, llm:7, content_quality:5 — its score is 53/90, not 6/10.
- "dimension_averages" are the averages of each dimension across all pages, each still out of 10.

IMPORTANT: The "recommendations" array MUST contain 8-12 items ranked by impact. Every audit has recommendations.
Use priority P1 for highest impact quick wins, P2 for this quarter, P3 for backlog.
Example recommendation format:
{"priority": "P1", "action": "Ship Organization + WebSite JSON-LD sitewide", "impact": "Very high - foundation for all AI citation", "effort": "Low - single template insert", "owner": "Dev"}

The "whats_working" and "whats_holding_back" arrays MUST each contain exactly 3 items.
"""

def clean_json_string(raw: str) -> str:
    """Strip markdown fences and extract the outermost JSON object."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end+1]
    return raw


def repair_and_parse(raw: str) -> dict:
    """Try several strategies to parse potentially malformed JSON."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        import json_repair  # type: ignore
        return json_repair.repair_json(raw, return_objects=True)
    except Exception:
        pass

    # Manual fixes: trailing commas, curly/smart quotes
    fixed = raw
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"Could not parse Gemini response as JSON. First 300 chars:\n{raw[:300]}")


def run_audit(model, pages: dict) -> dict:
    # pages values are already compact signal summaries from extract_page_signals
    pages_text = ""
    for url, signals in pages.items():
        pages_text += f"\n\n{'='*60}\n{signals}\n"

    response = model.generate_content(
        AUDIT_PROMPT + "\n\nPages to audit:\n" + pages_text,
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 65536,   # 2.5-flash supports up to 65k output tokens
        },
    )
    raw = clean_json_string(response.text)

    try:
        result = repair_and_parse(raw)
    except ValueError:
        # Retry: ask Gemini to fix its own output
        fix_prompt = (
            "The following text should be valid JSON but contains errors. "
            "Return ONLY the corrected JSON object with no other text, "
            "no markdown fences, no explanation:\n\n" + raw[:6000]
        )
        retry = model.generate_content(
            fix_prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 8192},
        )
        raw2 = clean_json_string(retry.text)
        result = repair_and_parse(raw2)

    # ── Post-process: fill missing recommendations from page findings ──────
    if not result.get("recommendations"):
        recs = []
        dim_keys = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
        dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM"]
        action_map = {
            "schema":          ("Ship schema.org JSON-LD structured data sitewide",
                                "Very high — biggest single AI citation unlock", "Low — template insert", "Dev"),
            "aria":            ("Add ARIA landmark roles to navigation and content regions",
                                "Medium — improves AI page structure parsing", "Low — template tweak", "Dev"),
            "alt_text":        ("Audit and fix all image alt attributes",
                                "High — dual accessibility and AI win", "Low — CMS field fix", "Content"),
            "headings":        ("Ensure every page has a unique, descriptive H1",
                                "High — primary topic signal for AI crawlers", "Low — template fix", "Dev"),
            "links":           ("Normalise internal links to consistent https:// protocol",
                                "Medium — removes redirect noise for crawlers", "Medium — site-wide pass", "Dev"),
            "meta":            ("Add bespoke meta description to every page",
                                "Medium — strengthens per-page topic signal", "Low — content pass", "Content"),
            "crawl":           ("Audit JS-dependent content and ensure static HTML fallbacks",
                                "High — critical for AI crawler access", "High — architecture review", "Dev"),
            "llm":             ("Add named experts, dates and first-hand detail to key pages",
                                "High — converts pages into citable authority content", "Medium — content pass", "Content"),
            "content_quality": ("Rewrite key pages to lead with user benefits rather than features",
                                "High — benefit-led content is more likely to be cited by AI in answers", "Medium — content rewrite", "Content"),
        }
        # Find lowest-scoring dimensions across all pages
        avg_scores = result.get("dimension_averages", {})
        sorted_dims = sorted(dim_keys, key=lambda k: avg_scores.get(k, 10))
        priority_map = ["P1","P1","P2","P2","P2","P3","P3","P3"]
        for i, dk in enumerate(sorted_dims):
            if dk in action_map:
                action, impact, effort, owner = action_map[dk]
                recs.append({
                    "priority": priority_map[i],
                    "action": action,
                    "impact": impact,
                    "effort": effort,
                    "owner": owner,
                })
        result["recommendations"] = recs

    # Fill whats_working / whats_holding_back if empty
    if not result.get("whats_working"):
        result["whats_working"] = [
            {"point": "Review audit details", "detail": "See per-page dimension breakdown for strengths."}
        ]
    if not result.get("whats_holding_back"):
        dim_keys = ["aria","schema","headings","meta","links","alt_text","crawl","llm"]
        avg_scores = result.get("dimension_averages", {})
        worst = sorted(dim_keys, key=lambda k: avg_scores.get(k, 10))[:3]
        labels = {"aria":"ARIA","schema":"Schema","headings":"Headings","meta":"Meta",
                  "links":"Links","alt_text":"Alt Text","crawl":"Crawl","llm":"LLM content"}
        result["whats_holding_back"] = [
            {"point": f"{labels.get(dk,'Unknown')} gap",
             "detail": f"Scoring {avg_scores.get(dk,0)}/10 — a priority improvement area."}
            for dk in worst
        ]

    # ── Always recalculate scores from dimension data (don't trust Gemini's maths) ──
    dim_keys = ["aria", "schema", "headings", "meta", "links", "alt_text", "crawl", "llm", "content_quality"]
    for page in result.get("pages", []):
        dims = page.get("dimensions", {})
        dim_scores = [dims.get(dk, {}).get("score", 0) for dk in dim_keys]
        if any(s > 0 for s in dim_scores):
            page["score"] = sum(dim_scores)   # sum of 8 dims = score out of 80

    # Recalculate dimension averages across all pages
    pages = result.get("pages", [])
    if pages:
        for dk in dim_keys:
            scores = [p.get("dimensions", {}).get(dk, {}).get("score", 0) for p in pages]
            result.setdefault("dimension_averages", {})[dk] = round(sum(scores) / len(scores), 1)
        # Recalculate average_score as average of page totals (out of 80)
        page_totals = [p.get("score", 0) for p in pages]
        result["average_score"] = round(sum(page_totals) / len(page_totals), 1)

    return result

# ─── Score colour ─────────────────────────────────────────────────────────────
def score_color(s):
    """Glass Atlas score band colours."""
    if s <= 2: return "#C26450"   # deep orange — warning
    if s <= 5: return "#E08A5F"   # orange earth — caution
    return "#6D8370"              # atlas green — good

# ─── Build Word doc ───────────────────────────────────────────────────────────
def build_docx(data: dict, month_year: str) -> bytes:
    """
    Writes audit data to a JSON file, then runs a static Node.js script
    that reads the JSON. This avoids injecting any user/LLM content into
    JS source code, which was causing SyntaxErrors.
    """
    import tempfile, os, subprocess, json as _json

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm","content_quality"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM","CONTENT"]

    def score_color_hex(s):
        if s <= 2: return "C0392B"
        if s <= 5: return "E67E22"
        return "27AE60"

    # Enrich data with derived fields the JS needs
    enriched = _json.loads(_json.dumps(data))   # deep copy via json

    # Strip backticks and long dashes from all string values in the data
    def _clean_strings(obj):
        if isinstance(obj, str):
            return obj.replace("`", "").replace("—", ",").replace("–", ",")
        if isinstance(obj, dict):
            return {k: _clean_strings(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_strings(v) for v in obj]
        return obj
    enriched = _clean_strings(enriched)
    enriched["month_year"]   = month_year
    enriched["logo_path"]    = LOGO_PATH
    enriched["dim_keys"]     = dim_keys
    enriched["dim_labels"]   = dim_labels
    # Add colour fields
    dim_avg = enriched.get("dimension_averages", {})
    enriched["dim_colors"] = {k: score_color_hex(dim_avg.get(k, 0)) for k in dim_keys}
    for page in enriched.get("pages", []):
        dims = page.get("dimensions", {})
        page["dim_colors"] = {k: score_color_hex(dims.get(k, {}).get("score", 0)) for k in dim_keys}
        page["score_color"] = score_color_hex(page.get("score", 0))

    # Ensure docx npm package is available — install locally if not found globally
    app_dir = os.path.dirname(os.path.abspath(__file__))
    node_modules = os.path.join(app_dir, "node_modules")
    if not os.path.exists(os.path.join(node_modules, "docx")):
        subprocess.run(
            ["npm", "install", "docx", "--prefix", app_dir],
            capture_output=True, timeout=120
        )

    # Write data JSON to temp file
    data_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    _json.dump(enriched, data_file, ensure_ascii=True)
    data_file.close()

    out_file = "/tmp/audit_output.docx"

    # Static JS template — reads ALL content from the JSON file, zero f-string injection
    js_script = r"""
'use strict';
const fs   = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat
} = require('docx');

const DATA_FILE = process.argv[2];
const OUT_FILE  = process.argv[3];

const d = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));

const company   = d.company_name  || 'Client';
const domain    = d.domain        || '';
const avg       = d.average_score || 0;
const dimAvg    = d.dimension_averages || {};
const pages     = d.pages         || [];
const themes    = d.cross_cutting_themes || [];
const recs      = d.recommendations || [];
const summary   = d.executive_summary || '';
const monthYear = d.month_year    || '';
const logoPath  = d.logo_path     || '';
const dimKeys   = d.dim_keys;
const dimLabels = d.dim_labels;
const dimColors = d.dim_colors;

const logoData = fs.existsSync(logoPath) ? fs.readFileSync(logoPath) : null;

const THIN  = { style: BorderStyle.SINGLE, size: 1,  color: 'DDDDDD' };
const NONE  = { style: BorderStyle.NONE };
const RED_LINE = { style: BorderStyle.SINGLE, size: 4, color: 'F2BC63' };
const borders   = { top: THIN, bottom: THIN, left: THIN, right: THIN };
const noBorders = { top: NONE, bottom: NONE, left: NONE, right: NONE };

function txt(text, opts) {
  return new TextRun(Object.assign({ text: String(text), font: 'Poppins', size: 20 }, opts || {}));
}
function para(children, opts) {
  if (!Array.isArray(children)) children = [children];
  return new Paragraph(Object.assign({ children }, opts || {}));
}
function cell(children, width, opts) {
  if (!Array.isArray(children)) children = [children];
  return new TableCell(Object.assign({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children
  }, opts || {}));
}
function hdrCell(label, width, bg) {
  return cell(
    [para([txt(label, { bold: true, color: 'FFFFFF', size: 18 })])],
    width,
    { shading: { fill: bg || 'F2BC63', type: ShadingType.CLEAR } }
  );
}
function scoreCell(score, width) {
  const col = score <= 2 ? 'C26450' : score <= 5 ? 'E08A5F' : '6D8370';
  return cell(
    [para([txt(score + '/10', { bold: true, color: 'FFFFFF', size: 22 })])],
    width,
    { shading: { fill: col, type: ShadingType.CLEAR } }
  );
}
function textCell(text, width, opts) {
  return cell([para([txt(text, { size: 18 })])], width, opts || {});
}
function colorCell(text, width, fill) {
  return cell(
    [para([txt(text, { bold: true, size: 18 })])],
    width,
    { shading: { fill: fill, type: ShadingType.CLEAR } }
  );
}

// ── Header ──────────────────────────────────────────────────────────────────
const NO_TABLE_BORDER = {
  top: NONE, bottom: NONE, left: NONE, right: NONE,
  insideH: NONE, insideV: NONE,
};
const headerChildren = [
  new Table({
    width: { size: 9906, type: WidthType.DXA },
    columnWidths: [1200, 8706],
    borders: NO_TABLE_BORDER,
    rows: [new TableRow({ children: [
      new TableCell({ borders: noBorders, width: { size: 1200, type: WidthType.DXA },
        children: logoData ? [para([new ImageRun({ data: logoData, transformation: { width: 50, height: 50 }, type: 'png' })])] : [para([txt('')])] }),
      new TableCell({ borders: noBorders, width: { size: 8706, type: WidthType.DXA },
        verticalAlign: VerticalAlign.CENTER,
        children: [para([txt('AI VISIBILITY AUDIT \u00b7 ' + company.toUpperCase() + ' \u00b7 ' + monthYear.toUpperCase(), { size: 16, color: '6B6B6B' })], { alignment: AlignmentType.RIGHT })] }),
    ]})]
  }),
  para([txt('')], { border: { bottom: RED_LINE } }),
];

// ── Footer ──────────────────────────────────────────────────────────────────
const footerChildren = [
  para([txt('')], { border: { top: RED_LINE } }),
  para([
    txt('GLASSATLAS.COM', { size: 16, color: '6B6B6B' }),
    txt('\t\tGLASSATLAS.COM', { size: 16, color: '6B6B6B' }),
    txt('\t\t', { size: 16 }),
    new TextRun({ children: [PageNumber.CURRENT], font: 'Poppins', size: 16, color: '6B6B6B' }),
  ]),
];

// ── Scorecard table ──────────────────────────────────────────────────────────
const scorecardRows = [
  new TableRow({ tableHeader: true, children: [
    hdrCell('PAGE', 3500), hdrCell('SCORE', 900), hdrCell('VERDICT', 4960)
  ]})
];
pages.forEach(function(p, i) {
  const fill = i % 2 === 0 ? 'F9F5F1' : 'FFFFFF';
  const shade = { shading: { fill, type: ShadingType.CLEAR } };
  scorecardRows.push(new TableRow({ children: [
    cell([para([txt(p.title || ('Page '+(i+1)), { bold: true, size: 18 })])], 3500, shade),
    cell([para([txt((p.score||0)+'/90', { bold: true, size: 18 })], { alignment: AlignmentType.CENTER })], 900, shade),
    cell([para([txt(p.verdict || '', { size: 18 })])], 4960, shade),
  ]}));
});

// ── Dimension averages row ───────────────────────────────────────────────────
const dimAvgCells = dimKeys.map(function(dk, i) {
  const s = dimAvg[dk] || 0;
  const col = dimColors[dk] || '6D8370';
  return cell([
    para([txt(dimLabels[i], { size: 13, bold: true, color: 'FFFFFF' })], { alignment: AlignmentType.CENTER }),
    para([txt(s+'/10', { size: 22, bold: true, color: 'FFFFFF' })], { alignment: AlignmentType.CENTER }),
  ], 1040, { shading: { fill: col, type: ShadingType.CLEAR } });
});

// ── Per-page sections ────────────────────────────────────────────────────────
const pageSections = [];
pages.forEach(function(p, i) {
  const dims = p.dimensions || {};
  const dimRows = dimKeys.map(function(dk, di) {
    const dim = dims[dk] || {};
    const s   = dim.score || 0;
    const col = (p.dim_colors || {})[dk] || '6D8370';
    return new TableRow({ children: [
      cell([para([txt(dimLabels[di], { bold: true, color: 'FFFFFF', size: 18 })])], 1200,
        { shading: { fill: col, type: ShadingType.CLEAR } }),
      cell([para([txt(s+'/10', { bold: true, size: 20 })], { alignment: AlignmentType.CENTER })], 600, {}),
      cell([para([txt(dim.detail || '', { size: 18 })])], 7160, {}),
    ]});
  });

  const sfParas = (p.specific_findings || []).map(function(sf) {
    return para([txt(sf, { size: 20 })], { numbering: { reference: 'bullets', level: 0 } });
  });
  if (!sfParas.length) sfParas.push(para([txt('No specific findings recorded.', { size: 20 })]));

  pageSections.push(
    para([txt('PAGE '+(i+1)+': '+(p.title||'').toUpperCase(), { size: 28, bold: true, color: 'F2BC63' })],
      { heading: HeadingLevel.HEADING_1 }),
    para([txt('URL: ', { bold: true, size: 20 }), txt(p.url||'', { size: 20, color: 'F2BC63' })]),
    para([txt('Total score: '+(p.score||0)+'/90', { bold: true, size: 20 })]),
    para([txt(p.headline_finding || '', { size: 20, italics: true })], {
      spacing: { before: 160 },
      border: { left: { style: BorderStyle.SINGLE, size: 20, color: 'F2BC63' } },
      indent: { left: 360 }
    }),
    para([txt('Dimension Breakdown', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
    new Table({ width: { size: 8960, type: WidthType.DXA }, columnWidths: [1200, 600, 7160], rows: dimRows }),
    para([txt('Specific Findings', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
    ...sfParas,
    para([new PageBreak()])
  );
});

// ── Themes ───────────────────────────────────────────────────────────────────
const themeParas = [];
themes.forEach(function(t) {
  themeParas.push(
    para([txt(t.title||'', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
    para([txt(t.detail||'', { size: 20 })], { spacing: { before: 80, after: 160 } })
  );
});
if (!themeParas.length) themeParas.push(para([txt('No cross-cutting themes identified.', { size: 20 })]));

// ── Recommendations table ────────────────────────────────────────────────────
const recRows = [
  new TableRow({ tableHeader: true, children: [
    hdrCell('PRI.', 600), hdrCell('ACTION', 4200), hdrCell('IMPACT', 1800),
    hdrCell('EFFORT', 1500), hdrCell('OWNER', 1260)
  ]})
];
recs.forEach(function(r, i) {
  const fill = i % 2 === 0 ? 'F9F5F1' : 'FFFFFF';
  const shade = { shading: { fill, type: ShadingType.CLEAR } };
  recRows.push(new TableRow({ children: [
    cell([para([txt(r.priority||'', { bold: true, size: 18, color: 'F2BC63' })])], 600, shade),
    cell([para([txt(r.action||'', { size: 18 })])], 4200, shade),
    cell([para([txt(r.impact||'', { size: 18 })])], 1800, shade),
    cell([para([txt(r.effort||'', { size: 18 })])], 1500, shade),
    cell([para([txt(r.owner||'', { size: 18 })])], 1260, shade),
  ]}));
});

// ── Executive Summary renderer ───────────────────────────────────────────────
function buildExecutiveSummary(raw) {
  var paras = [];
  // Parse pipe-delimited sections: OVERVIEW: ... | STRENGTHS: ... | GAPS: ... | VERDICT: ...
  var sections = raw.split('|').map(function(s) { return s.trim(); });
  sections.forEach(function(section) {
    var colonIdx = section.indexOf(':');
    if (colonIdx === -1) {
      // No section label — just output as normal paragraph
      if (section.trim()) {
        paras.push(para([txt(section.trim(), { size: 20 })], { spacing: { before: 80, after: 100 } }));
      }
      return;
    }
    var label = section.substring(0, colonIdx).trim().toUpperCase();
    var body  = section.substring(colonIdx + 1).trim();

    // Section heading
    paras.push(para([txt(label, { size: 20, bold: true, color: 'F2BC63' })],
      { spacing: { before: 200, after: 60 } }));

    // Split body into bullet lines (* prefix) and normal lines
    var lines = body.split('*').map(function(l) { return l.trim(); }).filter(Boolean);
    if (lines.length > 1) {
      // Multiple items = bullet list
      lines.forEach(function(line) {
        paras.push(para([txt(line, { size: 20 })],
          { numbering: { reference: 'bullets', level: 0 }, spacing: { after: 40 } }));
      });
    } else {
      // Single block = normal paragraph
      paras.push(para([txt(body, { size: 20 })],
        { spacing: { before: 40, after: 100 } }));
    }
  });

  // Fallback: if parsing produced nothing useful, just show raw text
  if (paras.length === 0) {
    paras.push(para([txt(raw, { size: 20 })], { spacing: { before: 80, after: 160 } }));
  }
  return paras;
}

// ── Document ─────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{ level: 0, format: LevelFormat.BULLET, text: '\u2022',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }]
    }]
  },
  styles: {
    default: { document: { run: { font: 'Poppins', size: 20 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Poppins', color: '25242B' },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Poppins', color: '25242B' },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1000, right: 1000, bottom: 1000, left: 1000 }
      }
    },
    headers: { default: new Header({ children: headerChildren }) },
    footers: { default: new Footer({ children: footerChildren }) },
    children: [
      // Cover
      para([txt('AI VISIBILITY AUDIT', { size: 56, bold: true })], { spacing: { before: 400 } }),
      para([txt(company, { size: 40, bold: true, color: 'F2BC63' })]),
      para([txt('Technical readiness for the AI search era', { size: 24, italics: true, color: '6B6B6B' })],
        { spacing: { after: 320 } }),
      new Table({
        width: { size: 4000, type: WidthType.DXA },
        columnWidths: [1800, 2200],
        rows: [
          new TableRow({ children: [
            cell([para([txt('Domain', { bold: true, size: 18 })])], 1800,
              { borders: noBorders, shading: { fill: 'F9F5F1', type: ShadingType.CLEAR } }),
            cell([para([txt(domain, { size: 18, color: 'F2BC63' })])], 2200,
              { borders: noBorders, shading: { fill: 'F9F5F1', type: ShadingType.CLEAR } }),
          ]}),
          new TableRow({ children: [
            cell([para([txt('Audit date', { bold: true, size: 18 })])], 1800,
              { borders: noBorders, shading: { fill: 'F9F5F1', type: ShadingType.CLEAR } }),
            cell([para([txt(monthYear, { size: 18 })])], 2200,
              { borders: noBorders, shading: { fill: 'F9F5F1', type: ShadingType.CLEAR } }),
          ]}),
          new TableRow({ children: [
            cell([para([txt('Prepared by', { bold: true, size: 18 })])], 1800,
              { borders: noBorders, shading: { fill: 'F9F5F1', type: ShadingType.CLEAR } }),
            cell([para([txt('Glass Atlas', { size: 18 })])], 2200,
              { borders: noBorders, shading: { fill: 'F9F5F1', type: ShadingType.CLEAR } }),
          ]}),
        ]
      }),
      para([txt('Confidential. Prepared for the ' + company + ' digital team.', { size: 18, italics: true, color: '6B6B6B' })],
        { spacing: { before: 160 } }),
      para([new PageBreak()]),

      // Executive Summary — parse the structured pipe-delimited format
      para([txt('EXECUTIVE SUMMARY', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      ...buildExecutiveSummary(summary),

      // Scorecard
      para([txt('Headline Scorecard', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
      new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3500, 900, 4960], rows: scorecardRows }),

      // Dimension averages
      para([txt('Dimension Averages', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2, spacing: { before: 320 } }),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040, 1040],
        rows: [new TableRow({ children: dimAvgCells })]
      }),
      para([new PageBreak()]),

      // Methodology
      para([txt('METHODOLOGY', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      para([txt('This audit treats ' + domain + '\'s site the way ChatGPT, Perplexity, Gemini and Claude actually consume it. AI crawlers fetch a URL, parse the static HTML on first request, look for structured data, and decide whether the page is citable. They do not execute a full JavaScript render in most cases.', { size: 20 })]),
      para([txt('The Eight Dimensions', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
      ...['ARIA implementation. Semantic landmarks, role attributes, descriptive aria-label values.',
          'Structured data / schema markup. Schema.org JSON-LD — the single highest-leverage signal for AI citation.',
          'Heading structure. A clean H1\u2013H6 outline that gives crawlers a topical map of the page.',
          'Meta and SEO signals. Title, description, canonical, Open Graph, Twitter Card, robots, language.',
          'Link quality. Internal consistency, descriptive anchor text, protocol uniformity, link density.',
          'Image alt text. Descriptive alt attributes \u2014 the cheapest accessibility-and-AI dual win available.',
          'Crawlability and JS dependency. Whether content is present in static HTML or requires JS execution.',
          'LLM content signals. First-hand expertise, named authors, dates, citations, accreditations.',
          'Content quality. Benefit-led writing, clear value propositions, user-focused language, and direct answers to likely user questions.',
      ].map(function(t) { return para([txt(t, { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }); }),
      para([txt('Scoring Bands', { size: 24, bold: true })], { heading: HeadingLevel.HEADING_2 }),
      para([txt('Red (1\u20132): Critical. Actively blocking AI visibility. Fix first.', { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }),
      para([txt('Amber (3\u20135): Capping. Under-performing relative to what\u2019s possible. High value to fix.', { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }),
      para([txt('Green (6+): Working. Meets the bar AI crawlers expect.', { size: 20 })], { numbering: { reference: 'bullets', level: 0 } }),
      para([new PageBreak()]),

      // Per-page sections
      ...pageSections,

      // Cross-cutting themes
      para([txt('CROSS-CUTTING THEMES', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      ...themeParas,
      para([new PageBreak()]),

      // Recommendations
      para([txt('PRIORITY RECOMMENDATIONS', { size: 28, bold: true })], { heading: HeadingLevel.HEADING_1 }),
      para([txt('Ranked by AI-citation impact relative to implementation cost. P1 = ship in the next sprint. P2 = ship this quarter. P3 = ship when convenient.', { size: 20, italics: true })],
        { spacing: { after: 160 } }),
      new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 4200, 1800, 1500, 1260], rows: recRows }),
    ]
  }]
});

Packer.toBuffer(doc).then(function(buf) {
  fs.writeFileSync(OUT_FILE, buf);
  console.log('OK');
}).catch(function(e) { console.error(e); process.exit(1); });
"""

    script_file = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
    script_file.write(js_script)
    script_file.close()

    out_path = "/tmp/audit_output.docx"
    app_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    # Include both local (app dir) and global npm paths
    local_modules  = os.path.join(app_dir, "node_modules")
    global_modules = subprocess.run(
        ["npm", "root", "-g"], capture_output=True, text=True
    ).stdout.strip()
    env["NODE_PATH"] = local_modules + os.pathsep + global_modules
    result = subprocess.run(
        ["node", script_file.name, data_file.name, out_path],
        capture_output=True, text=True, timeout=90,
        env=env
    )
    os.unlink(script_file.name)
    os.unlink(data_file.name)

    if result.returncode != 0:
        raise RuntimeError(f"docx generation failed:\n{result.stderr[-2000:]}")

    with open(out_path, "rb") as f:
        return f.read()




# ─── Build One-Pager PDF ──────────────────────────────────────────────────────
def build_onepager(data: dict, month_year: str) -> bytes:
    """
    Guaranteed single-page A4 PDF.
    All sections use fixed rowHeights so text is clipped, never expanding the page.
    """
    company      = data.get("company_name", "Client")
    domain       = data.get("domain", "")
    avg          = round(data.get("average_score", 0))
    dim_avg      = data.get("dimension_averages", {})
    exec_summary = data.get("executive_summary", "")
    working      = data.get("whats_working", [])[:3]
    holding      = data.get("whats_holding_back", [])[:3]
    wins         = data.get("three_quick_wins", [])[:3]

    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm","content_quality"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM","CONTENT"]

    GOLD    = colors.HexColor("#F2BC63")  # Yellow Gold — primary brand accent
    DARK    = colors.HexColor("#25242B")  # Charcoal
    GREY    = colors.HexColor("#6B6B6B")
    LIGHT   = colors.HexColor("#F9F5F1")  # Paper White
    GREEN   = colors.HexColor("#6D8370")  # Atlas Green — working
    WARNING = colors.HexColor("#E08A5F")  # Orange Earth — holding back
    WHITE   = colors.white
    # Aliases for compatibility with original variable names
    RED = GOLD

    def dim_color(s):
        if s <= 2: return colors.HexColor("#C26450")  # deep orange
        if s <= 5: return colors.HexColor("#E08A5F")  # orange earth
        return colors.HexColor("#6D8370")             # atlas green

    def safe(t):
        return str(t).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

    def clean(t):
        t = str(t).replace("`","").replace("\u2014",",").replace("\u2013",",").replace("  "," ")
        return safe(t.strip())

    # ── Page geometry ────────────────────────────────────────────────
    PAGE_W, PAGE_H = A4          # 595 x 842 pt
    ML = MR = 13*mm
    MT = MB = 10*mm
    W  = PAGE_W - ML - MR        # usable width
    H  = PAGE_H - MT - MB        # usable height ~822pt

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
    )

    # ── Style factory ────────────────────────────────────────────────
    _sc = {}
    def S(name, **kw):
        key = name + str(sorted(kw.items()))
        if key not in _sc:
            base = dict(fontName="Helvetica", fontSize=8, leading=10,
                        textColor=DARK, spaceAfter=0, spaceBefore=0)
            base.update(kw)
            _sc[key] = ParagraphStyle(name + str(len(_sc)), **base)
        return _sc[key]

    def P(markup, **kw):
        return Paragraph(markup, S("p", **kw))

    # ── Fixed height budget (pts) ───────────────────────────────────
    # Coded values are scaled so total ~= usable page height (785pt).
    # ReportLab adds internal padding on top, so these values are tuned
    # so the content fills the page without overflowing.
    ROW_HDR  = 33    # logo header
    ROW_HERO = 84    # headline + intro paragraph
    ROW_SCOR = 84    # score box — large number + labels + verdict line
    ROW_DIMS = 28    # dimension badge strip
    ROW_WHHD = 29    # WH section header row
    ROW_WH   = 94    # each WH content row (x3) — title + 3 lines detail
    ROW_WINS = 25    # "quick wins" label row
    ROW_WNUM = 146   # wins content row — number + title + detail
    ROW_CTA  = 35    # CTA bar
    ROW_FOOT = 13    # footer

    # Spacers (pts)
    SP1 = 5   # after header
    SP2 = 4   # after hero
    SP3 = 4   # after score box
    SP4 = 4   # after dims
    SP5 = 4   # after WH section
    SP6 = 3   # after wins

    # ═══════════════════════════════════════════════════════════════
    # HELPER: Table cell that clips content to a fixed height
    # ReportLab clips automatically when rowHeights is specified and
    # splitByRow=0 is not set — we just need to specify the heights.
    # ═══════════════════════════════════════════════════════════════

    story = []

    # ═══════════════════════════════════════════════════════════════
    # 1. HEADER
    # ═══════════════════════════════════════════════════════════════
    LOGO_SZ = 26
    logo_cell = RLImage(LOGO_PATH, width=LOGO_SZ, height=LOGO_SZ) if os.path.exists(LOGO_PATH) else P("")
    hdr_t = Table(
        [[logo_cell,
          P(f'<font name="Helvetica" size="6" color="#6B6B6B">AI VISIBILITY SNAPSHOT<br/>'
            f'{safe(company).upper()} &middot; {safe(month_year).upper()}</font>',
            alignment=TA_RIGHT, leading=8)]],
        colWidths=[LOGO_SZ + 3*mm, W - LOGO_SZ - 3*mm],
        rowHeights=[ROW_HDR - 4],
    )
    hdr_t.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),0),
        ("BOTTOMPADDING", (0,0),(-1,-1),2),
        ("LINEBELOW",     (0,0),(-1,-1),1.5,RED),
    ]))
    story.append(hdr_t)
    story.append(Spacer(1, SP1))

    # ═══════════════════════════════════════════════════════════════
    # 2. HERO HEADLINE + INTRO  (fixed height table so it cannot grow)
    # ═══════════════════════════════════════════════════════════════
    if 'OVERVIEW:' in exec_summary.upper():
        ov = exec_summary.split('|')[0]
        ci = ov.find(':')
        intro_raw = clean(ov[ci+1:].strip()) if ci != -1 else clean(ov)
    else:
        intro_raw = clean(exec_summary)
    # 165 chars fits ~2 lines at 7pt/9lead within ROW_HERO
    intro_txt = intro_raw  # no cap — hero row height clips naturally

    hero_t = Table(
        [[P('<font name="Helvetica-Bold" size="17">Is your site ready '
            f'for the <font color="#F2BC63"><i>AI search era?</i></font></font>',
            leading=20)],
         [P(f'We audited <b>{safe(domain)}</b> the way ChatGPT, Perplexity, Gemini and Claude see it. '
            + intro_txt, fontSize=7, leading=9)]],
        colWidths=[W],
        rowHeights=[22, ROW_HERO - 22],
    )
    hero_t.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),0),
        ("BOTTOMPADDING", (0,0),(-1,-1),2),
    ]))
    story.append(hero_t)
    story.append(Spacer(1, SP2))

    # ═══════════════════════════════════════════════════════════════
    # 3. SCORE BOX  — fixed height, verdict clipped to 1 line
    # ═══════════════════════════════════════════════════════════════
    verdict_text = ""
    if "VERDICT:" in exec_summary.upper():
        for part in exec_summary.split("|"):
            if "VERDICT:" in part.upper():
                ci = part.find(":")
                verdict_text = clean(part[ci+1:].strip()) if ci != -1 else ""
                break

    label_html = (
        '<font name="Helvetica" size="6.5" color="#6B6B6B">AVERAGE PAGE SCORE</font><br/>'
        '<font name="Helvetica-Bold" size="12" color="#25242B">A solid foundation. </font>'
        '<font name="Helvetica-Bold" size="12" color="#F2BC63">A clear path forward.</font>'
    )
    if verdict_text:
        label_html += f'<br/><font name="Helvetica" size="6.5" color="#6B6B6B">{verdict_text}</font>'

    score_t = Table(
        [[P(f'<font name="Helvetica-Bold" size="36" color="#F2BC63">{avg}</font>'
            f'<font name="Helvetica" size="12" color="#6B6B6B">/90</font>',
            alignment=TA_CENTER, leading=40),
          P(label_html, leading=13)]],
        colWidths=[34*mm, W - 34*mm],
        rowHeights=[ROW_SCOR],
    )
    score_t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1),LIGHT),
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1),4),
        ("BOTTOMPADDING", (0,0),(-1,-1),4),
        ("LEFTPADDING",   (0,0),(0,0),  6),
        ("LEFTPADDING",   (1,0),(1,0),  8),
        ("RIGHTPADDING",  (0,0),(-1,-1),6),
    ]))
    story.append(score_t)
    story.append(Spacer(1, SP3))

    # ═══════════════════════════════════════════════════════════════
    # 4. DIMENSION BADGES  — fixed height
    # ═══════════════════════════════════════════════════════════════
    cw = W / 9
    dim_cells = []
    for dk, dl in zip(dim_keys, dim_labels):
        s = dim_avg.get(dk, 0)
        dim_cells.append(P(
            f'<font name="Helvetica" size="4.5" color="#FFFFFF">{dl}<br/></font>'
            f'<font name="Helvetica-Bold" size="10" color="#FFFFFF">{s}</font>'
            f'<font name="Helvetica" size="5.5" color="#FFFFFF">/10</font>',
            alignment=TA_CENTER, leading=9,
        ))
    dim_t = Table([dim_cells], colWidths=[cw]*9, rowHeights=[ROW_DIMS])
    ds = [("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
          ("LEFTPADDING",(0,0),(-1,-1),1),("RIGHTPADDING",(0,0),(-1,-1),1)]
    for i, dk in enumerate(dim_keys):
        ds.append(("BACKGROUND",(i,0),(i,0),dim_color(dim_avg.get(dk,0))))
    dim_t.setStyle(TableStyle(ds))
    story.append(dim_t)
    story.append(Spacer(1, SP4))

    # ═══════════════════════════════════════════════════════════════
    # 5. WHAT'S WORKING / HOLDING BACK
    #    Fixed rowHeights on every row — content is clipped, never grows
    # ═══════════════════════════════════════════════════════════════
    GAP  = 4*mm
    HALF = (W - GAP) / 2
    # detail text char limit: ~100 chars fits 2 lines at 6.5pt within ROW_WH
    DET_CAP = 999  # no cap — row height clips naturally

    def build_side(items, hdr_text, hdr_color, icon, icon_color_hex):
        row_heights = [ROW_WHHD]
        rows = [[P(f'<font name="Helvetica-Bold" size="8" color="#FFFFFF">{hdr_text}</font>',
                   leading=10)]]
        for it in (items + [{}, {}, {}])[:3]:
            pt  = clean(it.get("point","")) if it else ""
            det = clean(it.get("detail",""))[:DET_CAP] if it else ""
            content = (
                f'<font name="Helvetica-Bold" size="7.5" color="{icon_color_hex}">{icon} </font>'
                f'<font name="Helvetica-Bold" size="7.5">{pt}</font>'
                + (f'<br/><font name="Helvetica" size="6.5" color="#4A4A4A">{det}</font>' if det else "")
            ) if pt else " "
            rows.append([P(content, leading=9)])
            row_heights.append(ROW_WH)

        t = Table(rows, colWidths=[HALF], rowHeights=row_heights)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  hdr_color),
            ("BACKGROUND",    (0,1),(-1,-1), colors.HexColor("#F9F9F9")),
            ("TOPPADDING",    (0,0),(-1,-1), 3),
            ("BOTTOMPADDING", (0,0),(-1,-1), 3),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
            ("VALIGN",        (0,0),(-1,-1), "TOP"),
            ("LINEBELOW",     (0,1),(-1,-2), 0.4, colors.HexColor("#E0E0E0")),
        ]))
        return t

    working_t = build_side(working, "\u271a WHAT\u2019S WORKING",     GREEN, "\u271a", "#6D8370")
    holding_t = build_side(holding, "! WHAT\u2019S HOLDING YOU BACK", WARNING, "!",   "#E08A5F")

    sides = Table([[working_t, Spacer(GAP,1), holding_t]],
                  colWidths=[HALF, GAP, HALF])
    sides.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),0),
        ("BOTTOMPADDING", (0,0),(-1,-1),0),
    ]))
    story.append(sides)
    story.append(Spacer(1, SP5))

    # ═══════════════════════════════════════════════════════════════
    # 6. THREE QUICK WINS  — fixed height per win cell
    # ═══════════════════════════════════════════════════════════════
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#DDDDDD"), spaceAfter=2))
    # Section label + heading in one fixed-height row
    label_t = Table(
        [[P('<font name="Helvetica" size="5.5" color="#6B6B6B">THREE MOVES THAT MOVE THE NEEDLE</font><br/>'
            '<font name="Helvetica-Bold" size="11">Quick wins, big impact</font>',
            leading=11)]],
        colWidths=[W], rowHeights=[ROW_WINS],
    )
    label_t.setStyle(TableStyle([
        ("LEFTPADDING",   (0,0),(-1,-1),0),
        ("RIGHTPADDING",  (0,0),(-1,-1),0),
        ("TOPPADDING",    (0,0),(-1,-1),2),
        ("BOTTOMPADDING", (0,0),(-1,-1),2),
    ]))
    story.append(label_t)

    THIRD = W / 3
    # detail cap: ~115 chars fits ~3 lines at 7pt/9lead within ROW_WNUM
    WIN_DET_CAP = 999  # no cap — row height clips naturally
    win_cells = []
    for w in (wins + [{},{},{}])[:3]:
        num    = clean(w.get("number","")) if w else ""
        title  = clean(w.get("title",""))  if w else ""
        detail = clean(w.get("detail",""))[:WIN_DET_CAP] if w else ""
        win_cells.append(P(
            f'<font name="Helvetica-Bold" size="20" color="#F2BC63">{num}</font><br/>'
            f'<font name="Helvetica-Bold" size="7.5">{title}</font><br/>'
            f'<font name="Helvetica" size="7" color="#4A4A4A">{detail}</font>',
            leading=10,
        ) if num else P(" "))

    wins_t = Table([win_cells], colWidths=[THIRD]*3, rowHeights=[ROW_WNUM])
    wins_t.setStyle(TableStyle([
        ("VALIGN",        (0,0),(-1,-1),"TOP"),
        ("TOPPADDING",    (0,0),(-1,-1),2),
        ("BOTTOMPADDING", (0,0),(-1,-1),0),
        ("LEFTPADDING",   (0,0),(0,0),  0),
        ("LEFTPADDING",   (1,0),(-1,-1),8),
        ("RIGHTPADDING",  (0,0),(-1,-1),6),
        ("LINEAFTER",     (0,0),(1,-1), 0.5, colors.HexColor("#DDDDDD")),
    ]))
    story.append(wins_t)
    story.append(Spacer(1, SP6))

    # ═══════════════════════════════════════════════════════════════
    # 7. CTA BAR  — fixed height
    # ═══════════════════════════════════════════════════════════════
    cta_t = Table(
        [[P('<font name="Helvetica-Bold" size="9.5">We\u2019ll walk your team through<br/>'
            'every finding. <font color="#F2BC63"><i>No obligation.</i></font></font>',
            leading=13),
          P('<font name="Helvetica" size="6" color="#6B6B6B">BOOK A SESSION<br/></font>'
            '<font name="Helvetica-Bold" size="10">hello@glassatlas.com</font>',
            alignment=TA_CENTER, leading=12)]],
        colWidths=[W*0.52, W*0.48],
        rowHeights=[ROW_CTA],
    )
    cta_t.setStyle(TableStyle([
        ("BACKGROUND",    (1,0),(1,0),  LIGHT),
        ("TOPPADDING",    (0,0),(-1,-1),6),
        ("BOTTOMPADDING", (0,0),(-1,-1),6),
        ("LEFTPADDING",   (0,0),(0,0),  10),
        ("LEFTPADDING",   (1,0),(1,0),  8),
        ("RIGHTPADDING",  (0,0),(-1,-1),8),
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
        ("LINEABOVE",     (0,0),(-1,0), 2, RED),
    ]))
    story.append(cta_t)

    # Footer removed — CTA bar is the final element

    doc.build(story)
    return buf.getvalue()




# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700;800&display=swap');
  html, body, [class*="css"] {{ font-family: 'Poppins', Helvetica, Arial, sans-serif; }}
  .summit-header {{ background: {SUMMIT_RED}; padding: 1.2rem 1.5rem; border-radius: 8px;
    color: white; margin-bottom: 1.5rem; }}
  .summit-header h1 {{ margin: 0; font-size: 1.4rem; font-weight: 700; }}
  .summit-header p {{ margin: 0; font-size: 0.85rem; opacity: 0.85; }}
  .score-card {{ background: {SUMMIT_LIGHT}; border-radius: 8px; padding: 1rem;
    text-align: center; }}
  .score-big {{ font-size: 3rem; font-weight: 700; color: {SUMMIT_RED}; }}
  .dim-badge {{ display: inline-block; border-radius: 4px; padding: 4px 8px;
    font-size: 0.75rem; font-weight: 700; color: white; margin: 2px; }}
  .stButton > button {{ background: {SUMMIT_RED}; color: white; border: none;
    border-radius: 6px; font-weight: 600; padding: 0.6rem 1.4rem; }}
  .stButton > button:hover {{ background: #d9a548; }}
  .stDownloadButton > button {{ background: {SUMMIT_RED}; color: white; border: none;
    border-radius: 6px; font-weight: 600; }}
  .stDownloadButton > button:hover {{ background: #d9a548; }}
</style>
<div class="summit-header">
  <h1>🔍 Glass Atlas · AI Visibility Audit</h1>
  <p>Audit any website the way ChatGPT, Perplexity, Gemini and Claude see it. Clarity, craft and conviction.</p>
</div>
""", unsafe_allow_html=True)

# ─── Inputs ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    domain = st.text_input("Domain to audit", placeholder="e.g. example.com")
with col2:
    month_year = st.text_input("Audit month / year", value=datetime.now().strftime("%B %Y"))

st.markdown("**Additional pages to audit** (up to 3 internal URLs, one per line)")
extra_raw = st.text_area("", placeholder="https://example.com/category\nhttps://example.com/product\nhttps://example.com/blog", height=80)
extra_urls = [u for u in extra_raw.strip().splitlines() if u.strip()]

with st.expander("⚠️ Site blocked by bot protection? Paste HTML manually"):
    st.markdown(
        "Some sites (Cloudflare, Akamai etc.) block automated fetches. "
        "If the audit shows no content, open the page in Chrome, press **Ctrl+U** "
        "to view source, copy all, and paste below. You can paste up to 4 pages."
    )
    paste_entries = []
    for _i in range(4):
        _label = "Homepage" if _i == 0 else f"Page {_i + 1}"
        _pc1, _pc2 = st.columns([1, 2])
        with _pc1:
            _u = st.text_input(f"URL ({_label})", key=f"paste_url_{_i}",
                               placeholder="https://example.com/" if _i == 0 else f"https://example.com/page-{_i+1}")
        with _pc2:
            _h = st.text_area(f"HTML ({_label})", key=f"paste_html_{_i}",
                              height=100, placeholder="<!DOCTYPE html>..." if _i == 0 else "")
        if _u.strip() and _h.strip():
            paste_entries.append((_u.strip(), _h.strip()))

run = st.button("🚀 Run Audit", use_container_width=True)

# ── Run audit and store everything in session_state ────────────────────────
if run:
    if not domain:
        st.error("Please enter a domain.")
        st.stop()

    model = get_gemini_client()

    # Build pasted HTML dict from all filled-in entries
    pasted_html = {url: html for url, html in paste_entries}

    fetch_status = st.empty()
    fetch_status.info("Fetching pages…")

    pages = fetch_pages(domain, extra_urls, pasted_html or None)

    # Show what actually happened per URL and separate blocked from usable
    fetch_log = []
    blocked_urls = []
    for url, signals in pages.items():
        if "FETCH_BLOCKED" in signals:
            fetch_log.append(f"🚫 **{url}** — blocked, cannot be scored")
            blocked_urls.append(url)
        elif "FETCH_FAILED" in signals:
            fetch_log.append(f"❌ **{url}** — fetch failed")
            blocked_urls.append(url)
        elif "Wayback Machine" in signals:
            fetch_log.append(f"🗄️ **{url}** — retrieved via Wayback Machine cache")
        elif "manually provided" in signals:
            fetch_log.append(f"📋 **{url}** — using pasted HTML")
        elif "JS_SHELL" in signals:
            fetch_log.append(f"⚠️ **{url}** — JS-rendered site (shell only)")
        else:
            fetch_log.append(f"✅ **{url}** — fetched successfully")

    # Hard gate: if any URLs are blocked, halt before Gemini
    if blocked_urls:
        fetch_status.error(
            f"❌ **{len(blocked_urls)} of {len(pages)} pages could not be fetched.**\n\n"
            + "\n".join(fetch_log)
            + "\n\n---\n\n"
            + "**These pages appear to be blocked by bot protection (Cloudflare, Akamai, etc.) "
            + "or returned an empty response.**\n\n"
            + "To continue, get the raw HTML for the blocked URLs:\n"
            + "1. Open each blocked URL in your browser\n"
            + "2. Right-click, then **View Page Source** (or press Ctrl+U / Cmd+Option+U)\n"
            + "3. Select all (Ctrl+A / Cmd+A), copy (Ctrl+C / Cmd+C)\n"
            + "4. Open the **'Site blocked?'** expander above and paste the HTML against the URL\n"
            + "5. Click **Run audit** again\n\n"
            + "Scoring these pages without their real HTML would produce false results, "
            + "so the audit has been stopped."
        )
        st.stop()

    fetch_status.success(
        f"Fetched {len(pages)} page(s). Running Gemini audit…\n\n" +
        "\n".join(fetch_log)
    )

    with st.spinner("Analysing with Gemini — this takes 20–40 seconds…"):
        try:
            audit = run_audit(model, pages)
        except Exception as e:
            st.error(f"Gemini audit failed: {e}")
            st.stop()

    # Pre-generate both files while we have the data
    with st.spinner("Building Word document…"):
        try:
            docx_bytes = build_docx(audit, month_year)
        except Exception as e:
            docx_bytes = None
            st.warning(f"Word doc error: {e}")

    with st.spinner("Building one-pager PDF…"):
        try:
            pdf_bytes = build_onepager(audit, month_year)
        except Exception as e:
            pdf_bytes = None
            st.warning(f"PDF error: {e}")

    # Store everything — survives download-button reruns
    st.session_state["audit"]      = audit
    st.session_state["month_year"] = month_year
    st.session_state["docx_bytes"] = docx_bytes
    st.session_state["pdf_bytes"]  = pdf_bytes

# ── Display results from session_state (persists across reruns) ────────────
if "audit" in st.session_state:
    audit      = st.session_state["audit"]
    month_year = st.session_state["month_year"]
    docx_bytes = st.session_state["docx_bytes"]
    pdf_bytes  = st.session_state["pdf_bytes"]

    company  = audit.get("company_name", domain)
    avg      = round(audit.get("average_score", 0))
    dim_avg  = audit.get("dimension_averages", {})
    dim_keys   = ["aria","schema","headings","meta","links","alt_text","crawl","llm","content_quality"]
    dim_labels = ["ARIA","SCHEMA","HEADINGS","META","LINKS","ALT TEXT","CRAWL","LLM","CONTENT"]

    st.markdown(f"## 📊 Results: {company}")

    # Score overview
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown(f"""
        <div class="score-card">
          <div style="font-size:0.7rem;font-weight:600;color:{SUMMIT_GREY};letter-spacing:1px">AVERAGE PAGE SCORE</div>
          <div class="score-big">{avg}</div>
          <div style="color:{SUMMIT_GREY}">/90</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        n_pages = len(audit.get("pages", []))
        page_label = f"Average across {n_pages} page{'s' if n_pages != 1 else ''}"
        badges = f'<div style="font-size:0.7rem;color:{SUMMIT_GREY};margin-bottom:4px">{page_label}</div>'
        for dk, dl in zip(dim_keys, dim_labels):
            s = dim_avg.get(dk, 0)
            badges += f'<span class="dim-badge" style="background:{score_color(s)}">{dl}: {s}/10</span>'
        st.markdown(f'<div style="padding:1rem">{badges}</div>', unsafe_allow_html=True)
        summary_full = audit.get('executive_summary','')
        # Render the structured pipe-delimited format nicely in Streamlit
        if '|' in summary_full:
            sections = [s.strip() for s in summary_full.split('|') if s.strip()]
            for sec in sections:
                colon = sec.find(':')
                if colon != -1:
                    label = sec[:colon].strip()
                    body  = sec[colon+1:].strip()
                    # Bullet lines start with *
                    if '*' in body:
                        bullets = [b.strip() for b in body.split('*') if b.strip()]
                        st.markdown(f"**{label}**")
                        for b in bullets:
                            st.markdown(f"- {b}")
                    else:
                        st.markdown(f"**{label}:** {body}")
                else:
                    st.markdown(sec)
        else:
            st.markdown(f"**Executive summary:** {summary_full[:300]}{'...' if len(summary_full)>300 else ''}")
            if len(summary_full) > 300:
                with st.expander("Read full summary"):
                    st.markdown(summary_full)

    # Page tabs
    pages_data = audit.get("pages", [])
    if pages_data:
        tabs = st.tabs([p.get("title", f"Page {i+1}") for i, p in enumerate(pages_data)])
        for tab, page in zip(tabs, pages_data):
            with tab:
                st.markdown(f"**Score:** {page.get('score',0)}/90 &nbsp;|&nbsp; *{page.get('verdict','')}*")
                st.info(page.get("headline_finding",""))
                dims = page.get("dimensions", {})
                rows = []
                for dk, dl in zip(dim_keys, dim_labels):
                    d = dims.get(dk, {})
                    rows.append({"Dimension": dl, "Score": f"{d.get('score',0)}/10", "Detail": d.get("detail","")})
                st.table(rows)
                sfinds = page.get("specific_findings", [])
                if sfinds:
                    st.markdown("**Specific findings:**")
                    for sf in sfinds:
                        st.markdown(f"- {sf}")

    # Recommendations
    recs = audit.get("recommendations", [])
    if recs:
        st.markdown("### 📋 Priority Recommendations")
        st.table([{
            "Priority": r.get("priority",""),
            "Action": r.get("action","")[:120],
            "Impact": r.get("impact",""),
            "Effort": r.get("effort",""),
            "Owner": r.get("owner",""),
        } for r in recs])

    # ── Download buttons — data already in memory, no recompute ───────────
    st.markdown("---")
    st.markdown("### 📥 Download Outputs")
    col_d, col_p = st.columns(2)

    slug = company.lower().replace(' ', '-').replace('.', '')

    with col_d:
        if docx_bytes:
            st.download_button(
                "⬇️ Download Full Audit (.docx)",
                data=docx_bytes,
                file_name=f"glassatlas-ai-audit-{slug}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="dl_docx",
            )
        else:
            st.error("Word document could not be generated.")

    with col_p:
        if pdf_bytes:
            st.download_button(
                "⬇️ Download One-Pager (.pdf)",
                data=pdf_bytes,
                file_name=f"glassatlas-ai-snapshot-{slug}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_pdf",
            )
        else:
            st.error("PDF could not be generated.")

    # Clear results button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear & run new audit"):
        for k in ["audit","month_year","docx_bytes","pdf_bytes"]:
            st.session_state.pop(k, None)
        st.rerun()

