"""
literature_live.py

OPTIONAL live literature lookup for the general-background tier, sitting
ALONGSIDE the hand-verified store in literature.json (never replacing it).

Design rules - each one exists because of a way this can go wrong:

  * OFF by default. Enable with the environment variable LIVE_LITERATURE=1
    (an Azure App Setting in deployment). Nothing here runs otherwise.
  * Only consulted when the curated store has NO match for the question, so
    every answer that had a vetted citation before still gets exactly that.
  * The citation string is BUILT IN CODE from the API's structured fields
    (authors, year, title, venue, DOI), never written by the model, so the
    model cannot invent or garble a reference.
  * Results are always marked vetted=False, and the UI shows them under an
    "Unvetted" tag: nobody has checked that the paper supports the claim.
  * Titles and abstracts are UNTRUSTED text from an external database (the
    OpenAlex docs say plainly that they pass such text through unsanitised).
    They are stripped of markup and control characters, length-capped, and
    the prompt tells the model they are data, not instructions.
  * The abstract goes to the model only. It is never returned to the
    browser (copyright, and it keeps the UI to a citation + link).
  * It must never make the app worse: every failure (timeout, HTTP 429
    from the shared public pool, bad JSON) returns [] and trips a short
    circuit breaker, so a slow or throttled API costs at most one timeout,
    not one per question.

Providers (choose with LIVE_LITERATURE_PROVIDER, default "openalex"):

  * openalex          OpenAlex works search (title + abstract + full text,
                      relevance-ranked, CC0 data). Works with no key for light
                      use; a FREE key (instant, openalex.org/settings/api)
                      raises the daily budget 10x - set OPENALEX_API_KEY.
                      Search costs $1 per 1,000 calls, so this project's
                      volume is effectively free. Abstracts arrive as an
                      inverted index (word -> positions) and are rebuilt here.
  * semantic_scholar  Semantic Scholar paper search. Unauthenticated callers
                      share one public pool that is often throttled (HTTP
                      429); a key is issued by email. Optional
                      SEMANTIC_SCHOLAR_API_KEY.

Both are normalised to one paper shape, so everything after the fetch
(filtering, sanitising, citation building, caching, breaker) is shared.

Try the real call from your own machine (this sandbox cannot reach it):

    python literature_live.py "hemodynamic response function"
"""

import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,year,authors,venue,abstract,externalIds,url,citationCount"

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
OPENALEX_SELECT = ("id,doi,display_name,title,publication_year,authorships,"
                   "primary_location,abstract_inverted_index,cited_by_count")
PROVIDERS = ("openalex", "semantic_scholar")

TIMEOUT_SEC = 4.0             # hard cap on what a lookup can add to a request
RESULTS_FETCHED = 8           # ask for a few, keep the best MAX_RESULTS
MAX_RESULTS = 2
MIN_ABSTRACT_CHARS = 200      # a citation with nothing to summarise is useless
MAX_ABSTRACT_CHARS = 1200
CACHE_TTL_SEC = 24 * 3600
NEGATIVE_CACHE_TTL_SEC = 10 * 60
BREAKER_SEC = 60
CACHE_MAX_ENTRIES = 200

_TRUE = {"1", "true", "yes", "on"}

_STOP = {
    "the", "and", "for", "with", "that", "this", "what", "whats", "which", "who", "whom", "how",
    "why", "when", "where", "does", "did", "are", "was", "were", "you", "your", "can", "could",
    "would", "should", "tell", "explain", "describe", "define", "about", "into", "from", "than",
    "then", "there", "their", "them", "its", "has", "have", "had", "not", "but", "any", "all",
    "some", "more", "most", "much", "many", "very", "just", "also", "give", "show", "list",
    "mean", "means", "meaning", "please", "know", "like", "research", "neuroscience", "study",
}
_ANCHOR = "fnirs"  # this tool is about fNIRS, so bias the search toward it
_ANCHOR_WORDS = {"fnirs", "nirs", "near-infrared", "nearinfrared"}

_cache = {}                      # query -> (expires_at, results)
_state = {"blocked_until": 0.0, "last_error": ""}  # circuit breaker + why the last lookup failed


def live_enabled() -> bool:
    return os.environ.get("LIVE_LITERATURE", "").strip().lower() in _TRUE


def reset_state() -> None:
    """Clears the cache and breaker (used by tests)."""
    _cache.clear()
    _state["blocked_until"] = 0.0
    _state["last_error"] = ""


def last_error() -> str:
    """Why the most recent live lookup failed ('' if it did not). Lookups
    fail silently by design, so this is how you find out what happened."""
    return _state["last_error"]


# ---------------------------------------------------------
# Text hygiene - everything from the API is untrusted
# ---------------------------------------------------------
_TAGS = re.compile(r"<[^>]{0,200}>")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2066-\u2069]")


def sanitize(text, limit: int) -> str:
    """Strip markup and control/bidi characters, collapse whitespace, cap the
    length (cutting at a sentence or word boundary)."""
    if not isinstance(text, str):
        return ""
    text = _TAGS.sub(" ", text)
    text = _CONTROL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end >= limit * 0.5:
        return cut[:end + 1]
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "\u2026"


# ---------------------------------------------------------
# Citation built in code
# ---------------------------------------------------------
def _format_author(name: str) -> str:
    parts = [p for p in re.split(r"\s+", sanitize(name, 80)) if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    surname = [parts.pop()]
    while parts and parts[-1].islower() and len(parts) >= 1:   # von, van, de ...
        surname.insert(0, parts.pop())
    if not parts:
        return " ".join(surname)
    initials = " ".join(p[0].upper() + "." for p in parts if p[0].isalpha())
    return f"{' '.join(surname)}, {initials}".strip().rstrip(",")


def build_citation(paper: dict) -> str:
    names = [_format_author(a.get("name", "")) for a in (paper.get("authors") or []) if isinstance(a, dict)]
    names = [n for n in names if n]
    if len(names) > 6:
        authors = ", ".join(names[:6]) + ", et al."
    elif len(names) > 2:
        authors = ", ".join(names[:-1]) + ", & " + names[-1]
    elif len(names) == 2:
        authors = f"{names[0]} & {names[1]}"
    else:
        authors = names[0] if names else "Unknown authors"
    year = paper.get("year")
    year_txt = f"({year})" if isinstance(year, int) else "(n.d.)"
    title = sanitize(paper.get("title"), 300).rstrip(".")
    venue = sanitize(paper.get("venue"), 150)
    out = f"{authors} {year_txt}. {title}."
    if venue:
        out += f" {venue}."
    return out


def _doi(paper: dict) -> str:
    ids = paper.get("externalIds") or {}
    doi = ids.get("DOI") if isinstance(ids, dict) else None
    return doi.strip().lower() if isinstance(doi, str) else ""


def _paper_url(paper: dict) -> str:
    doi = _doi(paper)
    if doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi):
        return "https://doi.org/" + urllib.parse.quote(doi, safe="/:()-._;")
    url = paper.get("url")
    if isinstance(url, str) and url.startswith("https://www.semanticscholar.org/"):
        return url
    return ""


# ---------------------------------------------------------
# Query building and relevance
# ---------------------------------------------------------
def content_words(question: str) -> list:
    seen, out = set(), []
    for w in re.findall(r"[a-z][a-z\-]{2,}", (question or "").lower()):
        if w in _STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out[:8]


def get_provider() -> str:
    """LIVE_LITERATURE_PROVIDER if valid; else Semantic Scholar only when
    that is the only key configured; else OpenAlex."""
    chosen = os.environ.get("LIVE_LITERATURE_PROVIDER", "").strip().lower()
    if chosen in PROVIDERS:
        return chosen
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip() and not os.environ.get("OPENALEX_API_KEY", "").strip():
        return "semantic_scholar"
    return "openalex"


def build_query(words: list, anchor: bool = True) -> str:
    """Search string from the question's content words. `anchor` appends
    "fnirs" to bias results toward this tool's field - right for Semantic
    Scholar's ranker, wrong for OpenAlex, whose search ANDs every word (an
    extra required word would needlessly shrink the result set; topical
    relevance is enforced afterwards by the overlap filter instead)."""
    if not words:
        return ""
    if anchor and not (set(words) & _ANCHOR_WORDS):
        words = words + [_ANCHOR]
    return " ".join(words)


def _overlap(paper: dict, words: list) -> int:
    haystack = f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()
    return sum(1 for w in words if w in haystack)


# ---------------------------------------------------------
# OpenAlex -> the common paper shape
# ---------------------------------------------------------
def _abstract_from_inverted_index(inv) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text."""
    if not isinstance(inv, dict) or not inv:
        return ""
    slots = {}
    for word, positions in inv.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int) and 0 <= p < 4000:
                slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))


def normalize_openalex(work: dict) -> dict:
    """One OpenAlex work in the same shape Semantic Scholar uses, so the
    rest of the pipeline is provider-agnostic. Every field is optional."""
    if not isinstance(work, dict):
        return {}
    authors = []
    for a in work.get("authorships") or []:
        name = ((a or {}).get("author") or {}).get("display_name") if isinstance(a, dict) else None
        if isinstance(name, str) and name.strip():
            authors.append({"name": name})
    source = ((work.get("primary_location") or {}).get("source") or {}) if isinstance(work.get("primary_location"), dict) else {}
    doi = work.get("doi") if isinstance(work.get("doi"), str) else ""
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi.strip(), flags=re.IGNORECASE)
    return {
        "paperId": (work.get("id") or "").rsplit("/", 1)[-1] if isinstance(work.get("id"), str) else "",
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year") if isinstance(work.get("publication_year"), int) else None,
        "authors": authors,
        "venue": source.get("display_name") if isinstance(source, dict) else None,
        "abstract": _abstract_from_inverted_index(work.get("abstract_inverted_index")),
        "externalIds": {"DOI": doi} if doi else {},
        "url": "",
        "citationCount": work.get("cited_by_count") if isinstance(work.get("cited_by_count"), int) else 0,
    }


def _request_urls(provider: str, query: str) -> list:
    """URLs to try, in order. OpenAlex gets a second, select-free URL: if my
    field list were ever rejected (HTTP 400) the lookup still works."""
    q = urllib.parse.quote(query)
    if provider == "semantic_scholar":
        return [f"{S2_SEARCH_URL}?query={q}&limit={RESULTS_FETCHED}&fields={S2_FIELDS}"]
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    tail = f"&api_key={urllib.parse.quote(key)}" if key else ""
    base = f"{OPENALEX_WORKS_URL}?search={q}&per_page={RESULTS_FETCHED}"
    return [f"{base}&select={OPENALEX_SELECT}{tail}", f"{base}{tail}"]


def _extract_papers(provider: str, payload) -> list:
    if provider == "semantic_scholar":
        papers = payload.get("data") if isinstance(payload, dict) else None
    else:
        raw = payload.get("results") if isinstance(payload, dict) else None
        papers = [normalize_openalex(w) for w in raw] if isinstance(raw, list) else None
    if not isinstance(papers, list):
        raise ValueError("unexpected response shape")
    return papers


# ---------------------------------------------------------
# HTTP (replaceable in tests)
# ---------------------------------------------------------
def _http_get_json(url: str, headers: dict, timeout: float):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


HTTP_GET = _http_get_json


def search_live(question: str, exclude_dois=None, fetch=None, now=time.time) -> list:
    """Up to MAX_RESULTS unvetted papers relevant to `question`, or [].

    Each result: {"id", "citation", "url", "summary" (sanitised abstract,
    for the model only), "vetted": False, "source"}. Never raises."""
    provider = get_provider()
    words = content_words(question)
    query = build_query(words, anchor=(provider == "semantic_scholar"))
    if not query:
        return []
    t = now()
    cache_key = (provider, query)
    hit = _cache.get(cache_key)
    if hit and hit[0] > t:
        return hit[1]
    if _state["blocked_until"] > t:
        return []

    _state["last_error"] = ""
    exclude = {d.lower() for d in (exclude_dois or set()) if d}
    topical = [w for w in words if w not in _ANCHOR_WORDS]
    need = max(1, math.ceil(0.6 * len(topical))) if topical else 1

    headers = {"User-Agent": "fnirs-rule-explainer/1.0 (academic project)", "Accept": "application/json"}
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if provider == "semantic_scholar" and s2_key:
        headers["x-api-key"] = s2_key

    urls = _request_urls(provider, query)
    papers = None
    for i, url in enumerate(urls):
        try:
            papers = _extract_papers(provider, (fetch or HTTP_GET)(url, headers, TIMEOUT_SEC))
            break
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", "replace")[:200].replace("\n", " ")
            except Exception:
                body = ""
            _state["last_error"] = f"{provider}: HTTP {e.code} {e.reason}" + (f" - {body}" if body else "")
            if e.code == 400 and i + 1 < len(urls):
                continue                      # retry without the optional `select`
            retry = e.headers.get("Retry-After") if getattr(e, "headers", None) else None
            wait = int(retry) if retry and str(retry).isdigit() else BREAKER_SEC
            _state["blocked_until"] = t + min(wait, 600)
            return []
        except Exception as e:
            _state["last_error"] = f"{provider}: {type(e).__name__}: {e}"
            _state["blocked_until"] = t + BREAKER_SEC
            return []
    if papers is None:
        return []
    _state["last_error"] = ""

    scored = []
    for p in papers:
        if not isinstance(p, dict):
            continue
        abstract = sanitize(p.get("abstract"), MAX_ABSTRACT_CHARS)
        if len(abstract) < MIN_ABSTRACT_CHARS or not sanitize(p.get("title"), 300):
            continue
        if _doi(p) and _doi(p) in exclude:
            continue
        ov = _overlap(p, topical) if topical else 1
        if ov < need:
            continue
        cites = p.get("citationCount") if isinstance(p.get("citationCount"), int) else 0
        scored.append((ov, 1 if _doi(p) else 0, cites, p, abstract))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

    results = []
    for _, _, _, p, abstract in scored[:MAX_RESULTS]:
        results.append({
            "id": "live:" + (sanitize(p.get("paperId"), 60) or _doi(p) or str(len(results))),
            "citation": build_citation(p),
            "url": _paper_url(p),
            "summary": abstract,
            "vetted": False,
            "source": provider,
        })

    if len(_cache) >= CACHE_MAX_ENTRIES:
        _cache.clear()
    _cache[cache_key] = (t + (CACHE_TTL_SEC if results else NEGATIVE_CACHE_TTL_SEC), results)
    return results


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "hemodynamic response function"
    prov = get_provider()
    has_key = bool(os.environ.get("OPENALEX_API_KEY" if prov == "openalex" else "SEMANTIC_SCHOLAR_API_KEY", "").strip())
    print(f"provider : {prov}  (api key set: {'yes' if has_key else 'no'})")
    print(f"question : {q}\nquery    : {build_query(content_words(q), anchor=(prov == 'semantic_scholar'))}")
    start = time.perf_counter()
    found = search_live(q)
    print(f"elapsed  : {time.perf_counter() - start:.2f}s  |  results: {len(found)}")
    if last_error():
        print(f"FAILED   : {last_error()}")
        print("(lookups fail silently in the app by design; the circuit breaker is now open for a minute)")
    elif not found:
        print("the API answered, but nothing passed the filters (no abstract / not on-topic enough)")
    for r in found:
        print(f"\n- {r['citation']}\n  {r['url']}\n  abstract ({len(r['summary'])} chars): {r['summary'][:200]}...")