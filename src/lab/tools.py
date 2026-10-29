"""Tools the investigator agent may call. Read-only: no shell, no file writes."""

import re
from html.parser import HTMLParser

import httpx
from langchain_core.tools import tool

MAX_CHARS = 8000


class _Text(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "svg"}

    def __init__(self):
        super().__init__()
        self.parts, self.links, self._skip = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "a":
            href = dict(attrs).get("href")
            if href and href.startswith("http"):
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page and return its readable text plus outbound links."""
    r = httpx.get(url, follow_redirects=True, timeout=20, headers={"User-Agent": "research-lab/0.1"})
    r.raise_for_status()
    if "html" in r.headers.get("content-type", ""):
        p = _Text()
        p.feed(r.text)
        text = "\n".join(p.parts)
        links = "\n".join(dict.fromkeys(p.links))[:2000]
    else:
        text, links = r.text, ""
    text = re.sub(r"\n{3,}", "\n\n", text)[:MAX_CHARS]
    # Fetched pages are untrusted data. The wrapper and the system prompt both say so.
    return f"<untrusted_page url='{url}'>\n{text}\n</untrusted_page>\n<links>\n{links}\n</links>"


def default_tools():
    """Built at CLI time so importing lab.graph (tests) needs no TAVILY_API_KEY."""
    from langchain_tavily import TavilySearch

    return [TavilySearch(max_results=8), fetch_url]
