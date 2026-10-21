import ipaddress
import re
import socket
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import env_int
from backend.logging_config import get_logger

logger = get_logger(__name__)

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ingestion limits: one document may not exceed these, so a huge PDF or page
# cannot run up embedding cost or memory.
MAX_CHUNKS_PER_DOCUMENT = env_int("MAX_CHUNKS_PER_DOCUMENT", 2000, 10, 100_000)
MAX_PDF_PAGES = env_int("MAX_PDF_PAGES", 500, 1, 10_000)
MAX_WEBPAGE_BYTES = env_int("MAX_WEBPAGE_MB", 5, 1, 50) * 1024 * 1024
WEBPAGE_TIMEOUT_SECONDS = env_int("WEBPAGE_TIMEOUT_SECONDS", 20, 2, 120)
MAX_REDIRECTS = 3
MAX_ARXIV_PDF_BYTES = 60 * 1024 * 1024
_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_USER_AGENT = "PapeerBot/1.0 (+research assistant document loader)"
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


class UnsafeURLError(ValueError):
    """The URL is not allowed to be fetched (scheme, private address, redirects)."""


def _stamp_title(docs: list[Document], title: str) -> list[Document]:
    for doc in docs:
        doc.metadata["title"] = title
    return docs


def _split_capped(splitter: RecursiveCharacterTextSplitter, docs: list[Document]) -> list[Document]:
    chunks = splitter.split_documents(docs)
    if len(chunks) > MAX_CHUNKS_PER_DOCUMENT:
        raise ValueError(
            f"Document is too large: {len(chunks)} chunks (limit {MAX_CHUNKS_PER_DOCUMENT})"
        )
    return chunks


def _load_pdf_pages(file_path: str) -> list[Document]:
    import pymupdf

    with pymupdf.open(file_path) as pdf:
        if pdf.page_count > MAX_PDF_PAGES:
            raise ValueError(f"PDF has {pdf.page_count} pages (limit {MAX_PDF_PAGES})")
    return PyMuPDFLoader(file_path).load()


def load_pdf(file_path: str) -> list[Document]:
    docs = _load_pdf_pages(file_path)
    return _stamp_title(_split_capped(_splitter, docs), Path(file_path).stem)


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_split_capped(_splitter, docs), Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(_split_capped(_md_splitter, docs), Path(file_path).stem)


# ── Web pages (SSRF-guarded) ──────────────────────────────────────────────────

def _is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        # ::ffff:10.0.0.1, 6to4 and NAT64 (64:ff9b::/96) embed an IPv4 address.
        mapped = ip.ipv4_mapped or ip.sixtofour
        if mapped is None and ip in _NAT64:
            mapped = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if mapped is not None:
            return _is_public_ip(mapped)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local  # includes 169.254.169.254 (cloud metadata)
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not ip.is_global  # shared 100.64/10, documentation ranges, ...
    )


def validate_public_url(url: str) -> urllib.parse.SplitResult:
    """Allow only http(s) URLs whose host resolves exclusively to public IPs.

    Every address the name resolves to is checked, so a host with one public
    and one private record is refused. Residual risk: DNS rebinding. The HTTP
    client resolves the name again when it connects, so a hostile DNS server
    with a zero TTL could answer differently the second time. Closing that
    fully needs connecting to the validated IP itself (custom transport) or
    an egress proxy/firewall that blocks private ranges.
    """

    parts = urllib.parse.urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise UnsafeURLError("Only http and https URLs are allowed")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("URL has no host")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise UnsafeURLError("URL has an invalid port")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise UnsafeURLError(f"Could not resolve host: {host}")
    if not infos:
        raise UnsafeURLError(f"Could not resolve host: {host}")
    for info in infos:
        address = info[4][0].split("%", 1)[0]  # drop an IPv6 zone id
        if not _is_public_ip(ipaddress.ip_address(address)):
            raise UnsafeURLError("URL points to a private or reserved network address")
    return parts


def _fetch_public(url: str) -> tuple[str, bytes, str]:
    """GET a public URL with manual, re-validated redirects and a size cap.

    Returns (final_url, body, content_type).
    """

    deadline = time.monotonic() + WEBPAGE_TIMEOUT_SECONDS
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        validate_public_url(current)
        response = requests.get(
            current,
            allow_redirects=False,
            stream=True,
            timeout=(5, WEBPAGE_TIMEOUT_SECONDS),
            headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9"},
        )
        try:
            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("Redirect without a Location header")
                current = urllib.parse.urljoin(current, location)
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not content_type.startswith(_HTML_TYPES):
                raise ValueError(f"Unsupported content type: {content_type.split(';')[0] or 'unknown'}")
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_WEBPAGE_BYTES:
                raise ValueError("Page is too large")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                body.extend(chunk)
                if len(body) > MAX_WEBPAGE_BYTES:
                    raise ValueError("Page is too large")
                if time.monotonic() > deadline:
                    raise TimeoutError("Page download timed out")
            return current, bytes(body), content_type
        finally:
            response.close()
    raise UnsafeURLError(f"Too many redirects (limit {MAX_REDIRECTS})")


def _charset(content_type: str) -> str | None:
    match = re.search(r"charset=([\w.:-]+)", content_type)
    return match.group(1) if match else None


def _html_to_document(url: str, body: bytes, content_type: str) -> Document:
    """Same text/metadata shape WebBaseLoader produced (source, title,
    description, language), with scripts and styles removed."""

    if content_type.startswith("text/plain"):
        text = body.decode(_charset(content_type) or "utf-8", errors="replace")
        return Document(page_content=text, metadata={"source": url})
    soup = BeautifulSoup(body, "html.parser", from_encoding=_charset(content_type))
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    metadata = {"source": url}
    if soup.title and soup.title.get_text(strip=True):
        metadata["title"] = soup.title.get_text(strip=True)
    description = soup.find("meta", attrs={"name": "description"})
    if description and description.get("content"):
        metadata["description"] = description.get("content")
    html = soup.find("html")
    if html and html.get("lang"):
        metadata["language"] = html.get("lang")
    return Document(page_content=soup.get_text(), metadata=metadata)


def load_webpage(url: str) -> list[Document]:
    final_url, body, content_type = _fetch_public(url)
    docs = [_html_to_document(final_url, body, content_type)]
    title = docs[0].metadata.get("title") or url
    return _stamp_title(_split_capped(_splitter, docs), title)


# ── ArXiv (fixed arxiv.org endpoints) ─────────────────────────────────────────

def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Search ArXiv Atom API by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    m = re.search(r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download and chunk an ArXiv paper PDF by its bare ID."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read(MAX_ARXIV_PDF_BYTES + 1)
    if len(pdf_bytes) > MAX_ARXIV_PDF_BYTES:
        raise ValueError("ArXiv PDF is too large")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        docs = _load_pdf_pages(tmp_path)
        if not docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        return _stamp_title(_split_capped(_splitter, docs), title)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    logger.info("document_load source_type=%s", "url" if source.startswith(("http://", "https://")) else Path(source).suffix.lower())
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")
