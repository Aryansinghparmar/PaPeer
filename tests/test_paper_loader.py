"""SSRF guard, redirect handling and size/chunk caps in the document loaders.

DNS (``socket.getaddrinfo``) and HTTP (``requests.get``) are faked, so no test
touches the network.
"""

from __future__ import annotations

import socket

import pytest

from backend import paper_loader
from backend.paper_loader import UnsafeURLError, load_webpage, validate_public_url

PUBLIC_IP = "93.184.215.14"


def _fake_dns(monkeypatch, table: dict[str, list[str]]):
    def getaddrinfo(host, port, *args, **kwargs):
        if host not in table:
            raise socket.gaierror("unknown host")
        out = []
        for ip in table[host]:
            if ":" in ip:
                out.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0)))
            else:
                out.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)))
        return out

    monkeypatch.setattr(paper_loader.socket, "getaddrinfo", getaddrinfo)


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b""):
        self.status_code = status
        self.headers = headers or {}
        self._body = body
        self.closed = False

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def raise_for_status(self):
        if self.status_code >= 400:
            raise paper_loader.requests.HTTPError(f"{self.status_code}")

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        self.closed = True


def _fake_http(monkeypatch, responses: dict[str, FakeResponse]):
    calls: list[str] = []

    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"]
        calls.append(url)
        return responses[url]

    monkeypatch.setattr(paper_loader.requests, "get", get)
    return calls


@pytest.mark.parametrize(
    "ip",
    [
        "169.254.169.254",  # cloud metadata (link-local)
        "10.0.0.1",
        "127.0.0.1",
        "192.168.1.10",
        "100.64.0.1",  # carrier-grade NAT
        "0.0.0.0",
        "224.0.0.1",  # multicast
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:10.0.0.1",  # IPv4-mapped private
        "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254
    ],
)
def test_private_and_reserved_addresses_are_refused(monkeypatch, ip):
    _fake_dns(monkeypatch, {"target.example": [ip]})
    with pytest.raises(UnsafeURLError):
        validate_public_url("http://target.example/")


def test_public_address_is_allowed(monkeypatch):
    _fake_dns(monkeypatch, {"paper.example": [PUBLIC_IP, "2606:2800:21f:cb07:6820:80da:af6b:8b2c"]})
    assert validate_public_url("https://paper.example/x").hostname == "paper.example"


def test_any_private_record_refuses_the_host(monkeypatch):
    _fake_dns(monkeypatch, {"mixed.example": [PUBLIC_IP, "10.0.0.1"]})
    with pytest.raises(UnsafeURLError):
        validate_public_url("http://mixed.example/")


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://paper.example/x", "gopher://x/", "http:///nohost", "https://paper.example:99999/"]
)
def test_bad_schemes_and_hosts_are_refused(monkeypatch, url):
    _fake_dns(monkeypatch, {"paper.example": [PUBLIC_IP]})
    with pytest.raises(UnsafeURLError):
        validate_public_url(url)


def test_literal_ip_urls_are_checked(monkeypatch):
    monkeypatch.setattr(
        paper_loader.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))],
    )
    with pytest.raises(UnsafeURLError):
        validate_public_url("http://169.254.169.254/metadata/instance")


def test_redirect_to_private_host_is_refused_before_fetching(monkeypatch):
    _fake_dns(monkeypatch, {"paper.example": [PUBLIC_IP], "internal.example": ["10.0.0.1"]})
    calls = _fake_http(
        monkeypatch,
        {"https://paper.example/a": FakeResponse(302, {"Location": "http://internal.example/secret"})},
    )
    with pytest.raises(UnsafeURLError):
        load_webpage("https://paper.example/a")
    assert calls == ["https://paper.example/a"]


def test_too_many_redirects(monkeypatch):
    _fake_dns(monkeypatch, {"paper.example": [PUBLIC_IP]})
    responses = {
        f"https://paper.example/{i}": FakeResponse(301, {"Location": f"/{i + 1}"}) for i in range(10)
    }
    calls = _fake_http(monkeypatch, responses)
    with pytest.raises(UnsafeURLError, match="redirects"):
        load_webpage("https://paper.example/0")
    assert len(calls) == paper_loader.MAX_REDIRECTS + 1


def test_html_page_becomes_titled_chunks_after_safe_redirect(monkeypatch):
    _fake_dns(monkeypatch, {"paper.example": [PUBLIC_IP]})
    html = (
        b"<html lang='en'><head><title>Attention Is All You Need</title>"
        b"<meta name='description' content='Transformers'><script>var x=1;</script></head>"
        b"<body><p>" + b"The Transformer relies on attention. " * 60 + b"</p></body></html>"
    )
    _fake_http(
        monkeypatch,
        {
            "https://paper.example/old": FakeResponse(301, {"Location": "/new"}),
            "https://paper.example/new": FakeResponse(
                200, {"Content-Type": "text/html; charset=utf-8"}, html
            ),
        },
    )
    docs = load_webpage("https://paper.example/old")
    assert len(docs) > 1
    assert {d.metadata["title"] for d in docs} == {"Attention Is All You Need"}
    assert docs[0].metadata["source"] == "https://paper.example/new"
    assert docs[0].metadata["language"] == "en"
    assert all("var x" not in d.page_content for d in docs)


def test_response_size_cap(monkeypatch):
    _fake_dns(monkeypatch, {"paper.example": [PUBLIC_IP]})
    monkeypatch.setattr(paper_loader, "MAX_WEBPAGE_BYTES", 1000)
    _fake_http(
        monkeypatch,
        {"https://paper.example/big": FakeResponse(200, {"Content-Type": "text/html"}, b"a" * 5000)},
    )
    with pytest.raises(ValueError, match="too large"):
        load_webpage("https://paper.example/big")


def test_non_html_content_type_is_refused(monkeypatch):
    _fake_dns(monkeypatch, {"paper.example": [PUBLIC_IP]})
    _fake_http(
        monkeypatch,
        {"https://paper.example/x.zip": FakeResponse(200, {"Content-Type": "application/zip"}, b"PK")},
    )
    with pytest.raises(ValueError, match="content type"):
        load_webpage("https://paper.example/x.zip")


def test_chunk_cap_per_document(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_loader, "MAX_CHUNKS_PER_DOCUMENT", 2)
    path = tmp_path / "long.txt"
    path.write_text("word " * 2000, encoding="utf-8")
    with pytest.raises(ValueError, match="too large"):
        paper_loader.load_text(str(path))
    monkeypatch.setattr(paper_loader, "MAX_CHUNKS_PER_DOCUMENT", 100)
    assert len(paper_loader.load_text(str(path))) > 2
