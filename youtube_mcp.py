#!/usr/bin/env python3
"""
YouTube Video MCP Server with Secure Cookie Management UI
=========================================================

Features:
- MCP Server mounted on /mcp (Streamable HTTP)
- Web Admin Portal on /admin to upload/paste/verify YouTube cookies
- Password protected by ADMIN_PASSWORD (via session cookie)
- Real-time cookie verification with yt-dlp
- DNS rebinding checks disabled to prevent Railway 421 errors
"""

from __future__ import annotations

import html
import logging
import os
import re
import secrets
import tempfile
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import uvicorn
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route, Mount

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings


# ---------------------------------------------------------------------------
# Configuration & Security
# ---------------------------------------------------------------------------

SERVER_NAME = "youtube-video-mcp"

CACHE_TTL_SECONDS = int(os.getenv("YOUTUBE_MCP_CACHE_TTL", "1800"))
CACHE_MAX_ITEMS = int(os.getenv("YOUTUBE_MCP_CACHE_SIZE", "10"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("YOUTUBE_MCP_MAX_TRANSCRIPT_CHARS", "400000"))
MAX_DESCRIPTION_CHARS = int(os.getenv("YOUTUBE_MCP_MAX_DESCRIPTION_CHARS", "20000"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("YOUTUBE_MCP_TIMEOUT", "30"))

DEFAULT_PREFERRED_LANGUAGES = ("en",)

ALLOWED_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

COOKIES_FILE_PATH = Path(os.getenv("YOUTUBE_COOKIES_PATH", "/app/cookies.txt"))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1234")
SESSION_COOKIE_NAME = "mcp_admin_session"
ADMIN_SESSION_TOKEN = secrets.token_hex(32)

COOKIE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(SERVER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

logger.setLevel(
    getattr(logging, os.getenv("YOUTUBE_MCP_LOG_LEVEL", "INFO").upper(), logging.INFO)
)
logger.propagate = False


# ---------------------------------------------------------------------------
# Cookie Verification & Management Helpers
# ---------------------------------------------------------------------------

def is_netscape_format(content: str) -> bool:
    lines = content.strip().splitlines()
    valid_lines = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            valid_lines += 1
    return valid_lines > 0


def verify_cookies_content(content: str) -> tuple[bool, str]:
    if not is_netscape_format(content):
        return False, "Invalid cookie format. Netscape tab-separated cookies require at least 7 columns per entry."

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        temp_path = tmp.name

    test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "cookiefile": temp_path,
        "socket_timeout": 15,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(test_video_url, download=False)
        return True, "Cookies successfully authenticated and verified with YouTube!"
    except Exception as exc:
        err_msg = str(exc)
        if "Sign in to confirm you’re not a bot" in err_msg:
            return False, "YouTube rejected these cookies (Bot confirmation failed)."
        return False, f"Verification failed: {err_msg[:200]}"
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def get_active_cookie_path() -> str | None:
    with COOKIE_LOCK:
        if COOKIES_FILE_PATH.is_file() and COOKIES_FILE_PATH.stat().st_size > 0:
            return str(COOKIES_FILE_PATH)
        return None


def save_active_cookies(content: str) -> None:
    with COOKIE_LOCK:
        COOKIES_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        COOKIES_FILE_PATH.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Models / Cache
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TranscriptResult:
    language: str
    language_code: str
    is_generated: bool
    source: str
    segments: list[dict[str, Any]]

    @property
    def transcript_text(self) -> str:
        lines: list[str] = []
        for item in self.segments:
            timestamp = item["timestamp"]
            text = item["text"].strip()
            if text:
                lines.append(f"[{timestamp}] {text}")
        return "\n".join(lines)


@dataclass
class CacheEntry:
    created_at: float
    value: dict[str, Any]


_CACHE: OrderedDict[str, CacheEntry] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def cache_get(video_id: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(video_id)
        if entry is None:
            return None

        if time.monotonic() - entry.created_at > CACHE_TTL_SECONDS:
            _CACHE.pop(video_id, None)
            return None

        _CACHE.move_to_end(video_id)
        return entry.value


def cache_put(video_id: str, value: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _CACHE[video_id] = CacheEntry(time.monotonic(), value)
        _CACHE.move_to_end(video_id)
        while len(_CACHE) > CACHE_MAX_ITEMS:
            _CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Formatting & Extraction
# ---------------------------------------------------------------------------

def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    try:
        parsed = urlparse(value)
    except ValueError as exc:
        raise ValueError("Invalid URL.") from exc

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("YouTube URL must use http:// or https://.")

    host = parsed.hostname.lower() if parsed.hostname else ""
    if host not in ALLOWED_YOUTUBE_HOSTS:
        raise ValueError("Only YouTube URLs are supported.")

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        query = parse_qs(parsed.query)
        video_ids = query.get("v")
        if video_ids and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_ids[0]):
            return video_ids[0]

        for pattern in (r"^/embed/([A-Za-z0-9_-]{11})", r"^/shorts/([A-Za-z0-9_-]{11})", r"^/live/([A-Za-z0-9_-]{11})"):
            match = re.match(pattern, parsed.path)
            if match:
                return match.group(1)

    if host in {"youtu.be", "www.youtu.be"}:
        match = re.match(r"^/([A-Za-z0-9_-]{11})(?:/|$)", parsed.path)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a valid YouTube video ID from: {value}")


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis >= 1000:
        whole += 1
        millis = 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_vtt_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    raise ValueError(f"Invalid VTT timestamp: {value}")


def clean_caption_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<\d{2}:\d{2}:\d{2}(?:\.\d{3})?>", "", text)
    text = re.sub(r"</?c(?:\.[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?v(?:\s+[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?lang(?:\s+[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    clipped = text[:max_chars]
    last_space = clipped.rfind(" ")
    if last_space > max_chars * 0.85:
        clipped = clipped[:last_space]
    return clipped.rstrip() + "\n[TRANSCRIPT TRUNCATED]", True


def language_priority(code: str, preferred: tuple[str, ...]) -> int:
    normalized = code.lower().replace("_", "-")
    for index, pref in enumerate(preferred):
        pref_norm = pref.lower().replace("_", "-")
        if normalized == pref_norm:
            return index * 10
        if normalized.startswith(pref_norm + "-"):
            return index * 10 + 1
    return len(preferred) * 10 + 100


def choose_transcript(transcripts: list[Any], preferred: tuple[str, ...]) -> Any:
    if not transcripts:
        raise ValueError("No transcripts were returned.")
    ranked = sorted(
        enumerate(transcripts),
        key=lambda pair: (
            language_priority(getattr(pair[1], "language_code", ""), preferred),
            bool(getattr(pair[1], "is_generated", True)),
            pair[0],
        ),
    )
    return ranked[0][1]


def fetch_with_youtube_transcript_api(video_id: str, preferred_languages: tuple[str, ...]) -> TranscriptResult:
    api = YouTubeTranscriptApi()
    cookie_path = get_active_cookie_path()

    transcript_list = api.list(video_id, cookies=cookie_path) if cookie_path else api.list(video_id)
    available = list(transcript_list)
    if not available:
        raise RuntimeError("youtube-transcript-api returned no transcript tracks.")

    selected = choose_transcript(available, preferred_languages)
    language = getattr(selected, "language", "Unknown")
    language_code = getattr(selected, "language_code", "unknown")
    is_generated = bool(getattr(selected, "is_generated", False))

    fetched = selected.fetch()
    segments: list[dict[str, Any]] = []
    for snippet in fetched:
        start = float(getattr(snippet, "start", 0.0))
        duration = float(getattr(snippet, "duration", 0.0))
        text = clean_caption_text(str(getattr(snippet, "text", "")))
        if text:
            segments.append({
                "start": start,
                "duration": duration,
                "timestamp": format_timestamp(start),
                "text": text,
            })

    if not segments:
        raise RuntimeError("Transcript track was fetched but contained no text.")

    return TranscriptResult(
        language=language,
        language_code=language_code,
        is_generated=is_generated,
        source="youtube-transcript-api",
        segments=segments,
    )


def choose_ytdlp_language(tracks: dict[str, Any], preferred: tuple[str, ...]) -> str | None:
    if not tracks:
        return None
    languages = list(tracks.keys())
    ranked = sorted(languages, key=lambda lang: (language_priority(lang, preferred), lang.lower()))
    return ranked[0] if ranked else None


def parse_vtt_file(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: list[dict[str, Any]] = []
    index = 0

    timestamp_pattern = re.compile(
        r"^\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{3})?)\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{3})?)"
    )

    while index < len(lines):
        line = lines[index].strip()
        if not line or line.upper() == "WEBVTT":
            index += 1
            continue

        if line.startswith("NOTE"):
            index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            continue

        match = timestamp_pattern.match(line)
        if not match and index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            match = timestamp_pattern.match(next_line)
            if match:
                index += 1
                line = next_line

        if not match:
            index += 1
            continue

        start = format_vtt_timestamp(match.group(1))
        end = format_vtt_timestamp(match.group(2))

        index += 1
        text_lines: list[str] = []
        while index < len(lines) and lines[index].strip():
            text_lines.append(lines[index])
            index += 1

        text = clean_caption_text(" ".join(text_lines))
        if text and end >= start:
            if not segments or segments[-1]["text"] != text:
                segments.append({
                    "start": start,
                    "end": end,
                    "duration": max(0.0, end - start),
                    "timestamp": format_timestamp(start),
                    "text": text,
                })

    if not segments:
        raise RuntimeError(f"No captions found in VTT file: {path}")
    return segments


def fetch_with_yt_dlp(url: str, video_id: str, preferred_languages: tuple[str, ...]) -> TranscriptResult:
    with tempfile.TemporaryDirectory(prefix="youtube_mcp_") as temp_dir:
        output_template = str(Path(temp_dir) / "%(id)s.%(ext)s")

        common_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "socket_timeout": REQUEST_TIMEOUT_SECONDS,
            "outtmpl": output_template,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "vtt",
        }

        cookie_path = get_active_cookie_path()
        if cookie_path:
            common_opts["cookiefile"] = cookie_path

        inspect_opts = {**common_opts, "writesubtitles": False, "writeautomaticsub": False}
        with yt_dlp.YoutubeDL(inspect_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        manual_tracks = info.get("subtitles") or {}
        auto_tracks = info.get("automatic_captions") or {}

        selected_source = None
        selected_language = None
        is_generated = False

        if manual_tracks:
            selected_language = choose_ytdlp_language(manual_tracks, preferred_languages)
            if selected_language:
                selected_source = manual_tracks
                is_generated = False

        if not selected_language and auto_tracks:
            selected_language = choose_ytdlp_language(auto_tracks, preferred_languages)
            if selected_language:
                selected_source = auto_tracks
                is_generated = True

        if not selected_language or selected_source is None:
            raise RuntimeError("yt-dlp found no usable subtitle tracks.")

        download_opts = {**common_opts, "subtitleslangs": [selected_language]}
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])

        vtt_files = sorted(Path(temp_dir).glob("*.vtt"))
        if not vtt_files:
            raise RuntimeError("yt-dlp produced no VTT file.")

        segments = parse_vtt_file(vtt_files[0])
        return TranscriptResult(
            language=selected_language,
            language_code=selected_language,
            is_generated=is_generated,
            source="yt-dlp",
            segments=segments,
        )


def fetch_metadata(url: str) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": REQUEST_TIMEOUT_SECONDS,
    }

    cookie_path = get_active_cookie_path()
    if cookie_path:
        options["cookiefile"] = cookie_path

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    description, description_truncated = trim_text(str(info.get("description") or ""), MAX_DESCRIPTION_CHARS)
    duration_seconds = info.get("duration")

    return {
        "video_id": info.get("id"),
        "url": info.get("webpage_url") or url,
        "title": info.get("title"),
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "uploader": info.get("uploader"),
        "description": description,
        "description_truncated": description_truncated,
        "duration_seconds": duration_seconds,
        "duration": format_timestamp(float(duration_seconds)) if duration_seconds else None,
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
    }


def normalize_preferred_languages(preferred_languages: list[str] | None) -> tuple[str, ...]:
    if not preferred_languages:
        return DEFAULT_PREFERRED_LANGUAGES
    cleaned: list[str] = []
    for language in preferred_languages:
        if isinstance(language, str) and language.strip():
            lang = language.strip().lower().replace("_", "-")
            if re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,4})?", lang) and lang not in cleaned:
                cleaned.append(lang)
    return tuple(cleaned) or DEFAULT_PREFERRED_LANGUAGES


def build_result(url: str, preferred_languages: tuple[str, ...]) -> dict[str, Any]:
    video_id = extract_video_id(url)
    canonical = canonical_url(video_id)

    cached = cache_get(video_id)
    if cached is not None:
        return cached

    started = time.monotonic()
    metadata: dict[str, Any]
    metadata_error: str | None = None

    try:
        metadata = fetch_metadata(canonical)
    except Exception as exc:
        metadata_error = f"{type(exc).__name__}: {exc}"
        metadata = {"video_id": video_id, "url": canonical, "metadata_error": metadata_error}

    transcript: TranscriptResult | None = None
    transcript_errors: list[str] = []

    try:
        transcript = fetch_with_youtube_transcript_api(video_id, preferred_languages)
    except Exception as exc:
        transcript_errors.append(f"youtube-transcript-api: {exc}")

    if transcript is None:
        try:
            transcript = fetch_with_yt_dlp(canonical, video_id, preferred_languages)
        except Exception as exc:
            transcript_errors.append(f"yt-dlp: {exc}")

    if transcript is None:
        elapsed = time.monotonic() - started
        return {
            "ok": False,
            "video": metadata,
            "transcript": None,
            "transcript_error": "No captions could be retrieved. Provide fresh cookies via the admin page if blocked.",
            "diagnostics": {"errors": transcript_errors, "elapsed_seconds": round(elapsed, 2)},
        }

    transcript_text, transcript_truncated = trim_text(transcript.transcript_text, MAX_TRANSCRIPT_CHARS)
    elapsed = time.monotonic() - started

    result = {
        "ok": True,
        "video": metadata,
        "transcript": {
            "language": transcript.language,
            "language_code": transcript.language_code,
            "is_generated": transcript.is_generated,
            "source": transcript.source,
            "segment_count": len(transcript.segments),
            "truncated": transcript_truncated,
            "segments": transcript.segments,
            "timestamped_text": transcript_text,
        },
        "diagnostics": {
            "elapsed_seconds": round(elapsed, 2),
            "preferred_languages": list(preferred_languages),
            "cookie_used": bool(get_active_cookie_path()),
        },
    }

    cache_put(video_id, result)
    return result


# ---------------------------------------------------------------------------
# MCP Setup & Tools
# ---------------------------------------------------------------------------

mcp = MCPServer(SERVER_NAME)


@mcp.tool()
def get_youtube_video(url: str, preferred_languages: list[str] | None = None) -> dict[str, Any]:
    """Fetch YouTube metadata and timestamped transcript segments."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL is required.")
    languages = normalize_preferred_languages(preferred_languages)
    return build_result(url.strip(), languages)


# ---------------------------------------------------------------------------
# Secure Admin HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP YouTube Cookie Manager</title>
    <style>
        :root {
            --bg: #0f172a; --card: #1e293b; --border: #334155;
            --text: #f8fafc; --muted: #94a3b8; --accent: #3b82f6;
            --success: #10b981; --error: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 2rem 1rem; display: flex; justify-content: center; }
        .container { max-width: 650px; width: 100%; background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 2rem; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
        h1 { font-size: 1.5rem; margin-bottom: 0.5rem; font-weight: 600; }
        p.subtitle { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
        .status-badge { display: inline-flex; align-items: center; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 500; margin-bottom: 1.5rem; }
        .status-badge.active { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); }
        .status-badge.inactive { background: rgba(239, 68, 68, 0.15); color: var(--error); border: 1px solid var(--error); }
        .form-group { margin-bottom: 1.25rem; }
        label { display: block; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem; color: var(--muted); }
        input[type="password"], input[type="file"], textarea {
            width: 100%; background: #0b1120; border: 1px solid var(--border);
            border-radius: 8px; color: var(--text); padding: 0.75rem; font-size: 0.9rem;
        }
        textarea { height: 160px; font-family: monospace; resize: vertical; }
        input:focus, textarea:focus { outline: none; border-color: var(--accent); }
        .btn { width: 100%; padding: 0.8rem; background: var(--accent); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn:hover { background: #2563eb; }
        .alert { margin-top: 1.25rem; padding: 0.85rem; border-radius: 8px; font-size: 0.875rem; display: none; }
        .alert.error { background: rgba(239, 68, 68, 0.15); color: var(--error); border: 1px solid var(--error); }
        .alert.success { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); }
        .divider { display: flex; align-items: center; text-align: center; margin: 1.25rem 0; color: var(--border); }
        .divider::before, .divider::after { content: ''; flex: 1; border-bottom: 1px solid var(--border); }
        .divider span { padding: 0 10px; color: var(--muted); font-size: 0.8rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>YouTube Cookies Setup</h1>
        <p class="subtitle">Authenticate YouTube requests directly from this server instance.</p>

        <div id="statusIndicator">
            __STATUS_BADGE__
        </div>

        <form id="cookieForm">
            <div id="authSection" style="__AUTH_DISPLAY__">
                <div class="form-group">
                    <label for="adminPass">Admin Master Password</label>
                    <input type="password" id="adminPass" placeholder="Enter ADMIN_PASSWORD">
                </div>
            </div>

            <div class="form-group">
                <label>Option A: Select Netscape cookies.txt</label>
                <input type="file" id="cookieFile" accept=".txt">
            </div>

            <div class="divider"><span>OR</span></div>

            <div class="form-group">
                <label for="cookieText">Option B: Paste Netscape Format Raw Text</label>
                <textarea id="cookieText" placeholder="# Netscape HTTP Cookie File&#10;.youtube.com\tTRUE\t/\tTRUE\t...\tSID\txxxxx"></textarea>
            </div>

            <button type="submit" id="submitBtn" class="btn">Verify & Activate Cookies</button>
        </form>

        <div id="alertBox" class="alert"></div>
    </div>

    <script>
        const form = document.getElementById('cookieForm');
        const alertBox = document.getElementById('alertBox');
        const submitBtn = document.getElementById('submitBtn');
        const fileInput = document.getElementById('cookieFile');
        const textInput = document.getElementById('cookieText');
        const passInput = document.getElementById('adminPass');

        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => { textInput.value = e.target.result; };
                reader.readAsText(file);
            }
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            alertBox.style.display = 'none';
            alertBox.className = 'alert';

            const cookiesContent = textInput.value.trim();
            const password = passInput ? passInput.value.trim() : '';

            if (!cookiesContent) {
                showAlert('Please provide cookie content by file or text area.', 'error');
                return;
            }

            submitBtn.disabled = true;
            submitBtn.innerText = 'Verifying with YouTube...';

            try {
                const res = await fetch('/admin/update-cookies', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ password: password, content: cookiesContent })
                });

                const data = await res.json();
                if (res.ok && data.ok) {
                    showAlert(data.message, 'success');
                    setTimeout(() => window.location.reload(), 1500);
                } else {
                    showAlert(data.error || 'Failed to update cookies', 'error');
                }
            } catch (err) {
                showAlert('Network error occurred: ' + err.message, 'error');
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerText = 'Verify & Activate Cookies';
            }
        });

        function showAlert(msg, type) {
            alertBox.innerText = msg;
            alertBox.className = 'alert ' + type;
            alertBox.style.display = 'block';
        }
    </script>
</body>
</html>
"""


def _is_authenticated(request: Request) -> bool:
    return request.cookies.get(SESSION_COOKIE_NAME) == ADMIN_SESSION_TOKEN


async def admin_dashboard(request: Request):
    has_active_cookie = get_active_cookie_path() is not None
    status_badge = (
        '<span class="status-badge active">&#x25CF; Active Cookies Loaded</span>'
        if has_active_cookie
        else '<span class="status-badge inactive">&#x25CF; No Cookies Active (YouTube may block requests)</span>'
    )
    is_auth = _is_authenticated(request)
    auth_display = "none" if is_auth else "block"

    page = HTML_TEMPLATE.replace("__STATUS_BADGE__", status_badge).replace("__AUTH_DISPLAY__", auth_display)
    return HTMLResponse(page)


async def update_cookies(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body."}, status_code=400)

    password = data.get("password", "")
    content = data.get("content", "").strip()

    if not _is_authenticated(request):
        if not password or password != ADMIN_PASSWORD:
            return JSONResponse({"ok": False, "error": "Incorrect admin password."}, status_code=401)

    if not content:
        return JSONResponse({"ok": False, "error": "Cookie content cannot be empty."}, status_code=400)

    verified, message = verify_cookies_content(content)
    if not verified:
        return JSONResponse({"ok": False, "error": message}, status_code=422)

    save_active_cookies(content)

    response = JSONResponse({"ok": True, "message": message})
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=ADMIN_SESSION_TOKEN,
        httponly=True,
        samesite="lax",
        max_age=86400 * 30,
    )
    return response


# ---------------------------------------------------------------------------
# ASGI App Construction & Server Launch
# ---------------------------------------------------------------------------

def create_app() -> Starlette:
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    
    # Generate Streamable-HTTP ASGI sub-app for /mcp endpoint
    mcp_subapp = mcp.streamable_http_app(
        transport_security=security,
        json_response=True
    )

    routes = [
        Route("/admin", endpoint=admin_dashboard, methods=["GET"]),
        Route("/admin/update-cookies", endpoint=update_cookies, methods=["POST"]),
        Mount("/mcp", app=mcp_subapp),
    ]

    return Starlette(debug=False, routes=routes)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info("Starting %s | host=%s | port=%d", SERVER_NAME, host, port)
    logger.info("MCP endpoint live on: /mcp")
    logger.info("Admin UI accessible on: /admin")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
