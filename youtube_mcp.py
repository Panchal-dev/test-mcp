"""
YouTube Video MCP Server
========================

Purpose
-------
Give an MCP-compatible LLM access to a YouTube video's:
- metadata (title, description, channel, duration, dates, etc.)
- timestamped transcript/captions

Transcript strategy
-------------------
1. youtube-transcript-api
2. yt-dlp subtitle tracks (manual + automatic)
3. If neither source has captions, return a clear structured error.

Language strategy
-----------------
- Prefer English when available.
- Otherwise select the best available transcript automatically.
- Manual captions are preferred over auto-generated captions.
- No language is hard-coded as the only accepted language.

This server uses stdio by default, which is the appropriate transport for
a local personal MCP server. IMPORTANT: do not use print() for diagnostics;
stdout is the MCP protocol channel. All logs go to stderr.

Python: 3.10+
Install:
    pip install "mcp[cli]" youtube-transcript-api yt-dlp

Run directly:
    python youtube_mcp.py

Development / Inspector:
    uv run mcp dev youtube_mcp.py
"""

from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from cryptography.fernet import Fernet, InvalidToken
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_NAME = "youtube-video-mcp"

# Personal-use defaults. Environment variables can override these.
CACHE_TTL_SECONDS = int(os.getenv("YOUTUBE_MCP_CACHE_TTL", "1800"))
CACHE_MAX_ITEMS = int(os.getenv("YOUTUBE_MCP_CACHE_SIZE", "10"))
MAX_TRANSCRIPT_CHARS = int(
    os.getenv("YOUTUBE_MCP_MAX_TRANSCRIPT_CHARS", "400000")
)
MAX_DESCRIPTION_CHARS = int(
    os.getenv("YOUTUBE_MCP_MAX_DESCRIPTION_CHARS", "20000")
)
REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("YOUTUBE_MCP_TIMEOUT", "30")
)

DEFAULT_PREFERRED_LANGUAGES = ("en",)

# Only allow YouTube URLs. This avoids turning the MCP into a generic
# arbitrary-URL downloader.
ALLOWED_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

# ---------------------------------------------------------------------------
# Cookie management configuration
# ---------------------------------------------------------------------------

ADMIN_PASSWORD = os.getenv("YOUTUBE_MCP_ADMIN_PASSWORD", "")
SESSION_SECRET = os.getenv("YOUTUBE_MCP_SESSION_SECRET", "")
COOKIE_ENCRYPTION_KEY = os.getenv("YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY", "")

# On Railway, /app is ephemeral unless a Volume is attached. The cookie store
# is encrypted at rest, but the encryption key must be kept in Railway Variables.
COOKIE_STORAGE_PATH = Path(
    os.getenv("YOUTUBE_MCP_COOKIE_STORAGE_PATH", "/app/data/youtube_cookies.enc")
)

MAX_COOKIE_FILE_BYTES = int(
    os.getenv("YOUTUBE_MCP_MAX_COOKIE_FILE_BYTES", str(2 * 1024 * 1024))
)

ACTIVE_COOKIE_SOURCE = "none"
_ACTIVE_COOKIES = ""
_ACTIVE_COOKIE_SHA256 = ""

# Admin browser-session cookie. HttpOnly prevents JavaScript from reading it.
ADMIN_COOKIE_NAME = "youtube_mcp_admin"
COOKIE_CSRF_FIELD = "csrf_token"

# Streamable HTTP for Railway; stdio can still be selected for local MCP use.
TRANSPORT = os.getenv("YOUTUBE_MCP_TRANSPORT", "streamable-http").strip().lower()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(SERVER_NAME)

if not logger.handlers:
    handler = logging.StreamHandler()  # stderr by default
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
# Models / cache
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


def cache_get(video_id: str) -> dict[str, Any] | None:
    entry = _CACHE.get(video_id)
    if entry is None:
        return None

    if time.monotonic() - entry.created_at > CACHE_TTL_SECONDS:
        _CACHE.pop(video_id, None)
        logger.debug("Cache expired | video_id=%s", video_id)
        return None

    _CACHE.move_to_end(video_id)
    logger.info("Cache hit | video_id=%s", video_id)
    return entry.value


def cache_put(video_id: str, value: dict[str, Any]) -> None:
    _CACHE[video_id] = CacheEntry(time.monotonic(), value)
    _CACHE.move_to_end(video_id)

    while len(_CACHE) > CACHE_MAX_ITEMS:
        removed_id, _ = _CACHE.popitem(last=False)
        logger.debug("Cache evicted | video_id=%s", removed_id)


# ---------------------------------------------------------------------------
# URL / video ID handling
# ---------------------------------------------------------------------------

def extract_video_id(url_or_id: str) -> str:
    """
    Extract a YouTube video ID from common YouTube URL formats.

    Accepted:
      - 11-character video ID
      - youtube.com/watch?v=...
      - youtu.be/...
      - youtube.com/embed/...
      - youtube.com/shorts/...
      - youtube.com/live/...
    """
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
        raise ValueError(
            "Only YouTube URLs are supported "
            "(youtube.com / youtu.be)."
        )

    # youtube.com/watch?v=VIDEO_ID
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
        query = parse_qs(parsed.query)
        video_ids = query.get("v")
        if video_ids:
            candidate = video_ids[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                return candidate

        path_patterns = (
            r"^/embed/([A-Za-z0-9_-]{11})",
            r"^/shorts/([A-Za-z0-9_-]{11})",
            r"^/live/([A-Za-z0-9_-]{11})",
        )
        for pattern in path_patterns:
            match = re.match(pattern, parsed.path)
            if match:
                return match.group(1)

    # youtu.be/VIDEO_ID
    if host in {"youtu.be", "www.youtu.be"}:
        match = re.match(r"^/([A-Za-z0-9_-]{11})(?:/|$)", parsed.path)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a valid YouTube video ID from: {value}")


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# ---------------------------------------------------------------------------
# Formatting / sanitization
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or HH:MM:SS.mmm when useful."""
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))

    if millis >= 1000:
        whole += 1
        millis = 0

    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)

    # For normal transcript display, second precision is easier to read.
    # Keep milliseconds only for sub-second timestamps.
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_vtt_timestamp(value: str) -> float:
    """
    Parse WebVTT timestamp:
      HH:MM:SS.mmm
      MM:SS.mmm
      HH:MM:SS,mmm
    """
    value = value.strip().replace(",", ".")
    parts = value.split(":")

    try:
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds

        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except ValueError as exc:
        raise ValueError(f"Invalid VTT timestamp: {value}") from exc

    raise ValueError(f"Invalid VTT timestamp: {value}")


def clean_caption_text(text: str) -> str:
    """Clean common WebVTT/YouTube markup while preserving readable text."""
    text = html.unescape(text)

    # Remove YouTube word-level timing tags.
    text = re.sub(r"<\d{2}:\d{2}:\d{2}(?:\.\d{3})?>", "", text)

    # Remove HTML/WebVTT tags.
    text = re.sub(r"</?c(?:\.[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?v(?:\s+[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?lang(?:\s+[^>]*)?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def trim_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    # Prefer a clean character boundary.
    clipped = text[:max_chars]
    last_space = clipped.rfind(" ")
    if last_space > max_chars * 0.85:
        clipped = clipped[:last_space]

    return clipped.rstrip() + "\n[TRANSCRIPT TRUNCATED]", True


# ---------------------------------------------------------------------------
# Transcript language selection
# ---------------------------------------------------------------------------

def language_priority(code: str, preferred: tuple[str, ...]) -> int:
    """
    Lower number = higher priority.

    Exact language matches beat regional variants:
      en > en-US > en-GB
    """
    normalized = code.lower().replace("_", "-")

    for index, pref in enumerate(preferred):
        pref_norm = pref.lower().replace("_", "-")
        if normalized == pref_norm:
            return index * 10
        if normalized.startswith(pref_norm + "-"):
            return index * 10 + 1

    return len(preferred) * 10 + 100


def choose_transcript(transcripts: list[Any], preferred: tuple[str, ...]) -> Any:
    """
    Choose:
      1. preferred language, manual first
      2. preferred language, auto-generated
      3. any language, manual first
      4. any language, generated

    This means a video with only Hindi captions still works when English
    captions are unavailable.
    """
    if not transcripts:
        raise ValueError("No transcripts were returned.")

    ranked = sorted(
        enumerate(transcripts),
        key=lambda pair: (
            language_priority(
                getattr(pair[1], "language_code", ""),
                preferred,
            ),
            bool(getattr(pair[1], "is_generated", True)),
            pair[0],
        ),
    )

    return ranked[0][1]


# ---------------------------------------------------------------------------
# Transcript source #1: youtube-transcript-api
# ---------------------------------------------------------------------------

def fetch_with_youtube_transcript_api(
    video_id: str,
    preferred_languages: tuple[str, ...],
) -> TranscriptResult:
    logger.info(
        "Transcript attempt 1/2 | source=youtube-transcript-api | video_id=%s",
        video_id,
    )

    api = YouTubeTranscriptApi()

    transcript_list = api.list(video_id)
    available = list(transcript_list)

    if not available:
        raise RuntimeError("youtube-transcript-api returned no transcript tracks.")

    logger.info(
        "Transcript tracks discovered | count=%d | video_id=%s",
        len(available),
        video_id,
    )

    selected = choose_transcript(available, preferred_languages)

    language = getattr(selected, "language", "Unknown")
    language_code = getattr(selected, "language_code", "unknown")
    is_generated = bool(getattr(selected, "is_generated", False))

    logger.info(
        "Selected transcript | language=%s | code=%s | generated=%s",
        language,
        language_code,
        is_generated,
    )

    fetched = selected.fetch()

    segments: list[dict[str, Any]] = []
    for snippet in fetched:
        start = float(getattr(snippet, "start", 0.0))
        duration = float(getattr(snippet, "duration", 0.0))
        text = clean_caption_text(str(getattr(snippet, "text", "")))

        if not text:
            continue

        segments.append(
            {
                "start": start,
                "duration": duration,
                "timestamp": format_timestamp(start),
                "text": text,
            }
        )

    if not segments:
        raise RuntimeError("Transcript track was fetched but contained no text.")

    return TranscriptResult(
        language=language,
        language_code=language_code,
        is_generated=is_generated,
        source="youtube-transcript-api",
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Transcript source #2: yt-dlp
# ---------------------------------------------------------------------------

def choose_ytdlp_language(
    tracks: dict[str, Any],
    preferred: tuple[str, ...],
) -> str | None:
    if not tracks:
        return None

    languages = list(tracks.keys())

    ranked = sorted(
        languages,
        key=lambda lang: (
            language_priority(lang, preferred),
            lang.lower(),
        ),
    )
    return ranked[0] if ranked else None


def parse_vtt_file(path: Path) -> list[dict[str, Any]]:
    """
    Minimal, dependency-free WebVTT parser.

    Handles normal YouTube VTT cues and ignores:
      - WEBVTT header
      - NOTE blocks
      - cue IDs
      - cue settings
    """
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    segments: list[dict[str, Any]] = []
    index = 0

    timestamp_pattern = re.compile(
        r"^\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{3})?)"
        r"\s*-->\s*"
        r"(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{3})?)"
    )

    while index < len(lines):
        line = lines[index].strip()

        if not line:
            index += 1
            continue

        if line.upper() == "WEBVTT":
            index += 1
            continue

        if line.startswith("NOTE"):
            index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            continue

        match = timestamp_pattern.match(line)

        # Cue IDs may occur immediately before the timestamp line.
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
            # YouTube auto captions can contain repeated rolling text.
            # Avoid only exact consecutive duplicates.
            if not segments or segments[-1]["text"] != text:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": max(0.0, end - start),
                        "timestamp": format_timestamp(start),
                        "text": text,
                    }
                )

    if not segments:
        raise RuntimeError(f"No usable captions found in VTT file: {path}")

    return segments


def fetch_with_yt_dlp(
    url: str,
    video_id: str,
    preferred_languages: tuple[str, ...],
) -> TranscriptResult:
    logger.info(
        "Transcript attempt 2/2 | source=yt-dlp | video_id=%s",
        video_id,
    )

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

        # First inspect metadata + available caption tracks.
        inspect_opts = {
            **common_opts,
            "writesubtitles": False,
            "writeautomaticsub": False,
        }

        with yt_dlp.YoutubeDL(inspect_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        manual_tracks = info.get("subtitles") or {}
        auto_tracks = info.get("automatic_captions") or {}

        selected_source = None
        selected_language = None
        is_generated = False

        if manual_tracks:
            selected_language = choose_ytdlp_language(
                manual_tracks,
                preferred_languages,
            )
            if selected_language:
                selected_source = manual_tracks
                is_generated = False

        if not selected_language and auto_tracks:
            selected_language = choose_ytdlp_language(
                auto_tracks,
                preferred_languages,
            )
            if selected_language:
                selected_source = auto_tracks
                is_generated = True

        if not selected_language or selected_source is None:
            raise RuntimeError(
                "yt-dlp found no usable manual or automatic subtitle tracks."
            )

        logger.info(
            "yt-dlp subtitle selected | language=%s | generated=%s",
            selected_language,
            is_generated,
        )

        download_opts = {
            **common_opts,
            "subtitleslangs": [selected_language],
        }

        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])

        vtt_files = sorted(Path(temp_dir).glob("*.vtt"))
        if not vtt_files:
            raise RuntimeError(
                "yt-dlp selected a subtitle track but produced no VTT file."
            )

        # Normally there is exactly one selected file.
        selected_vtt = vtt_files[0]
        logger.info("Parsing VTT | file=%s", selected_vtt.name)

        segments = parse_vtt_file(selected_vtt)

        # Find a readable language name from yt-dlp's metadata if available.
        language_name = selected_language

        track_metadata = selected_source.get(selected_language)
        if isinstance(track_metadata, list) and track_metadata:
            # yt-dlp entries don't consistently expose a human language name,
            # so keep the language code as the stable identifier.
            language_name = selected_language

        return TranscriptResult(
            language=language_name,
            language_code=selected_language,
            is_generated=is_generated,
            source="yt-dlp",
            segments=segments,
        )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def fetch_metadata(url: str) -> dict[str, Any]:
    logger.info("Fetching metadata | url=%s", url)

    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": REQUEST_TIMEOUT_SECONDS,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    description = info.get("description") or ""
    description, description_truncated = trim_text(
        str(description),
        MAX_DESCRIPTION_CHARS,
    )

    duration_seconds = info.get("duration")
    duration_formatted = (
        format_timestamp(float(duration_seconds))
        if duration_seconds is not None
        else None
    )

    upload_date = info.get("upload_date")
    if upload_date and len(str(upload_date)) == 8:
        upload_date = (
            f"{upload_date[0:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
        )

    return {
        "video_id": info.get("id"),
        "url": info.get("webpage_url") or url,
        "title": info.get("title"),
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "uploader": info.get("uploader"),
        "uploader_id": info.get("uploader_id"),
        "description": description,
        "description_truncated": description_truncated,
        "duration_seconds": duration_seconds,
        "duration": duration_formatted,
        "upload_date": upload_date,
        "timestamp": info.get("timestamp"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "categories": info.get("categories") or [],
        "tags": info.get("tags") or [],
        "thumbnail": info.get("thumbnail"),
        "live_status": info.get("live_status"),
        "availability": info.get("availability"),
    }


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def normalize_preferred_languages(
    preferred_languages: list[str] | None,
) -> tuple[str, ...]:
    if not preferred_languages:
        return DEFAULT_PREFERRED_LANGUAGES

    cleaned: list[str] = []
    for language in preferred_languages:
        if not isinstance(language, str):
            continue

        language = language.strip().lower().replace("_", "-")
        if not language:
            continue

        # Accept common language code forms, e.g. en, en-US, hi, gu.
        if re.fullmatch(r"[a-z]{2,3}(?:-[a-z]{2,4})?", language):
            if language not in cleaned:
                cleaned.append(language)

    return tuple(cleaned) or DEFAULT_PREFERRED_LANGUAGES


def build_result(
    url: str,
    preferred_languages: tuple[str, ...],
) -> dict[str, Any]:
    video_id = extract_video_id(url)
    canonical = canonical_url(video_id)

    cached = cache_get(video_id)
    if cached is not None:
        return cached

    started = time.monotonic()

    # Metadata is independent from captions. If metadata works but captions
    # don't, return the metadata plus a precise transcript error.
    metadata: dict[str, Any]
    metadata_error: str | None = None

    try:
        metadata = fetch_metadata(canonical)
    except Exception as exc:
        metadata_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Metadata failed | video_id=%s", video_id)
        metadata = {
            "video_id": video_id,
            "url": canonical,
            "title": None,
            "description": None,
            "duration_seconds": None,
            "duration": None,
            "metadata_error": metadata_error,
        }

    transcript: TranscriptResult | None = None
    transcript_errors: list[str] = []

    # Primary transcript source.
    try:
        transcript = fetch_with_youtube_transcript_api(
            video_id,
            preferred_languages,
        )
    except Exception as exc:
        message = f"youtube-transcript-api: {type(exc).__name__}: {exc}"
        transcript_errors.append(message)
        logger.warning(
            "Primary transcript source failed | video_id=%s | %s",
            video_id,
            message,
        )

    # Fallback transcript source.
    if transcript is None:
        try:
            transcript = fetch_with_yt_dlp(
                canonical,
                video_id,
                preferred_languages,
            )
        except Exception as exc:
            message = f"yt-dlp: {type(exc).__name__}: {exc}"
            transcript_errors.append(message)
            logger.warning(
                "Fallback transcript source failed | video_id=%s | %s",
                video_id,
                message,
            )

    if transcript is None:
        elapsed = time.monotonic() - started
        logger.error(
            "Video extraction failed | video_id=%s | elapsed=%.2fs",
            video_id,
            elapsed,
        )

        result = {
            "ok": False,
            "video": metadata,
            "transcript": None,
            "transcript_error": (
                "No usable transcript/caption track could be retrieved. "
                "The video may have captions disabled, inaccessible captions, "
                "or YouTube may be blocking transcript access."
            ),
            "diagnostics": {
                "errors": transcript_errors,
                "elapsed_seconds": round(elapsed, 2),
            },
        }

        # Metadata can still be useful even if transcript extraction fails.
        if metadata_error:
            result["metadata_error"] = metadata_error

        return result

    transcript_text = transcript.transcript_text
    transcript_text, transcript_truncated = trim_text(
        transcript_text,
        MAX_TRANSCRIPT_CHARS,
    )

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
            "fallbacks_used": (
                ["yt-dlp"]
                if transcript.source == "yt-dlp"
                else []
            ),
            "transcript_errors": transcript_errors,
        },
    }

    cache_put(video_id, result)

    logger.info(
        "Video extraction successful | video_id=%s | source=%s | "
        "language=%s | segments=%d | elapsed=%.2fs",
        video_id,
        transcript.source,
        transcript.language_code,
        len(transcript.segments),
        elapsed,
    )

    return result



# ---------------------------------------------------------------------------
# Cookie validation / secure storage
# ---------------------------------------------------------------------------

NETSCAPE_HEADER = "# Netscape HTTP Cookie File"


def _cookie_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _cookie_key() -> bytes:
    if not COOKIE_ENCRYPTION_KEY:
        raise RuntimeError(
            "YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY is not configured."
        )
    # Fernet expects a URL-safe base64-encoded 32-byte key.
    try:
        key = COOKIE_ENCRYPTION_KEY.encode("ascii")
        Fernet(key)
        return key
    except Exception as exc:
        raise RuntimeError(
            "YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY must be a valid Fernet key. "
            "Generate one with: python -c "
            "\"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        ) from exc


def validate_netscape_cookies(text: str) -> tuple[str, int]:
    """Validate and normalize a Netscape-format YouTube cookie jar."""
    if not isinstance(text, str):
        raise ValueError("Cookie data must be text.")

    raw_size = len(text.encode("utf-8", errors="ignore"))
    if raw_size > MAX_COOKIE_FILE_BYTES:
        raise ValueError(
            f"Cookie data exceeds the {MAX_COOKIE_FILE_BYTES} byte limit."
        )

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise ValueError("Cookie data is empty.")

    lines = normalized.splitlines()
    has_header = False
    count = 0

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            if line.lower().startswith(NETSCAPE_HEADER.lower()):
                has_header = True
            continue

        parts = raw_line.split("\t")
        if len(parts) != 7:
            raise ValueError(
                f"Invalid Netscape cookie format on line {line_number}: "
                "expected 7 tab-separated fields."
            )

        domain, include_subdomains, path, secure, expiry, name, _value = parts

        if not domain.strip():
            raise ValueError(f"Cookie domain is empty on line {line_number}.")

        if include_subdomains not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"Invalid include-subdomains value on line {line_number}."
            )

        if not path.startswith("/"):
            raise ValueError(f"Invalid cookie path on line {line_number}.")

        if secure not in {"TRUE", "FALSE"}:
            raise ValueError(f"Invalid secure value on line {line_number}.")

        if not re.fullmatch(r"-?\d+", expiry.strip()):
            raise ValueError(
                f"Invalid expiry timestamp on line {line_number}."
            )

        if not name.strip():
            raise ValueError(f"Cookie name is empty on line {line_number}.")

        clean_domain = domain.lower().lstrip(".")
        if clean_domain != "youtube.com" and not clean_domain.endswith(
            ".youtube.com"
        ):
            raise ValueError(
                "Only youtube.com and its subdomains are accepted. "
                "This server will not store unrelated site cookies."
            )

        count += 1

    if not has_header:
        raise ValueError(
            "The required '# Netscape HTTP Cookie File' header was not found."
        )

    if count == 0:
        raise ValueError("No YouTube cookies were found.")

    return normalized + "\n", count


def save_encrypted_cookie_store(cookie_text: str) -> None:
    """Persist cookies encrypted with Fernet; never write plaintext to disk."""
    key = _cookie_key()
    encrypted = Fernet(key).encrypt(cookie_text.encode("utf-8"))

    COOKIE_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = COOKIE_STORAGE_PATH.with_name(
        COOKIE_STORAGE_PATH.name + ".tmp"
    )
    temp_path.write_bytes(encrypted)

    try:
        os.chmod(temp_path, 0o600)
    except OSError:
        pass

    temp_path.replace(COOKIE_STORAGE_PATH)

    try:
        os.chmod(COOKIE_STORAGE_PATH, 0o600)
    except OSError:
        pass

    logger.info(
        "Encrypted cookie store updated | path=%s | sha256=%s",
        COOKIE_STORAGE_PATH,
        _cookie_hash(cookie_text),
    )


def load_encrypted_cookie_store() -> str:
    if not COOKIE_STORAGE_PATH.exists():
        return ""

    if not COOKIE_ENCRYPTION_KEY:
        logger.warning(
            "Encrypted cookie store exists but encryption key is missing; "
            "cookies were not loaded."
        )
        return ""

    try:
        encrypted = COOKIE_STORAGE_PATH.read_bytes()
        cookie_text = Fernet(_cookie_key()).decrypt(encrypted).decode("utf-8")
        normalized, count = validate_netscape_cookies(cookie_text)

        logger.info(
            "Encrypted cookie store loaded | cookies=%d | sha256=%s",
            count,
            _cookie_hash(normalized),
        )
        return normalized

    except InvalidToken:
        logger.error(
            "Cookie store authentication failed. Stored cookies were not loaded."
        )
    except Exception:
        logger.exception("Failed to load encrypted cookie store.")

    return ""


def clear_active_cookies(delete_persisted: bool = True) -> None:
    global _ACTIVE_COOKIES, _ACTIVE_COOKIE_SHA256

    _ACTIVE_COOKIES = ""
    _ACTIVE_COOKIE_SHA256 = ""

    if delete_persisted:
        try:
            COOKIE_STORAGE_PATH.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to delete encrypted cookie store.")

    logger.info("YouTube cookies cleared")


def activate_cookies(cookie_text: str, persist: bool) -> dict[str, Any]:
    global _ACTIVE_COOKIES, _ACTIVE_COOKIE_SHA256

    normalized, count = validate_netscape_cookies(cookie_text)

    _ACTIVE_COOKIES = normalized
    _ACTIVE_COOKIE_SHA256 = _cookie_hash(normalized)

    if persist:
        save_encrypted_cookie_store(normalized)

    return {
        "ok": True,
        "cookie_count": count,
        "sha256": _ACTIVE_COOKIE_SHA256,
        "persisted": persist,
    }


def _write_active_cookie_tempfile() -> Path | None:
    if not _ACTIVE_COOKIES:
        return None

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="ytcookies_",
        delete=False,
    )
    try:
        handle.write(_ACTIVE_COOKIES)
        handle.flush()
    finally:
        handle.close()

    path = Path(handle.name)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    return path


def verify_active_cookies() -> dict[str, Any]:
    """
    Verify the active cookie jar by:
    1. Re-validating Netscape syntax.
    2. Passing the jar to yt-dlp for a YouTube request.

    Cookie values are never returned or logged.
    """
    if not _ACTIVE_COOKIES:
        return {
            "ok": False,
            "verified": False,
            "reason": "No active YouTube cookies.",
        }

    normalized, count = validate_netscape_cookies(_ACTIVE_COOKIES)
    path = _write_active_cookie_tempfile()

    if path is None:
        return {
            "ok": False,
            "verified": False,
            "reason": "Could not create a temporary cookie jar.",
        }

    started = time.monotonic()

    try:
        options: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "socket_timeout": REQUEST_TIMEOUT_SECONDS,
            "cookiefile": str(path),
        }

        # This checks that yt-dlp can use the supplied cookie jar for a
        # real YouTube request. It is intentionally a public video and does
        # not attempt to expose account information.
        verification_url = "https://www.youtube.com/watch?v=BaW_jenozKc"

        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(verification_url, download=False)

        if not info or not info.get("id"):
            raise RuntimeError("YouTube returned no usable video information.")

        elapsed = time.monotonic() - started

        logger.info(
            "Cookie verification succeeded | cookies=%d | elapsed=%.2fs",
            count,
            elapsed,
        )

        return {
            "ok": True,
            "verified": True,
            "cookie_count": count,
            "sha256": _cookie_hash(normalized),
            "elapsed_seconds": round(elapsed, 2),
            "message": (
                "Cookie jar is syntactically valid and yt-dlp successfully "
                "used it for a YouTube request."
            ),
        }

    except Exception as exc:
        logger.warning(
            "Cookie verification failed | type=%s | message=%s",
            type(exc).__name__,
            str(exc)[:500],
        )

        return {
            "ok": False,
            "verified": False,
            "cookie_count": count,
            "sha256": _cookie_hash(normalized),
            "reason": (
                "Cookie syntax is valid, but the YouTube request failed. "
                "The cookies may be expired/invalid, or YouTube may be "
                "blocking the request."
            ),
            "error_type": type(exc).__name__,
        }

    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Could not remove temporary cookie file.")


# ---------------------------------------------------------------------------
# Admin session security
# ---------------------------------------------------------------------------

def _admin_enabled() -> bool:
    return bool(ADMIN_PASSWORD and SESSION_SECRET)


def _session_hash(token: str) -> str:
    return hmac.new(
        SESSION_SECRET.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _cleanup_admin_sessions() -> None:
    now = time.time()
    expired = [
        key
        for key, (expires, _csrf) in _ADMIN_SESSIONS.items()
        if expires <= now
    ]
    for key in expired:
        _ADMIN_SESSIONS.pop(key, None)


def _new_admin_session() -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)

    _ADMIN_SESSIONS[_session_hash(token)] = (
        time.time() + ADMIN_SESSION_TTL_SECONDS,
        csrf,
    )

    return token, csrf


def _get_admin_session(request: Request) -> tuple[bool, str | None]:
    _cleanup_admin_sessions()

    token = request.cookies.get(ADMIN_COOKIE_NAME)
    if not token:
        return False, None

    record = _ADMIN_SESSIONS.get(_session_hash(token))
    if not record:
        return False, None

    expires, csrf = record
    if expires <= time.time():
        _ADMIN_SESSIONS.pop(_session_hash(token), None)
        return False, None

    return True, csrf


def _password_matches(candidate: str) -> bool:
    if not ADMIN_PASSWORD:
        return False

    return hmac.compare_digest(
        candidate.encode("utf-8"),
        ADMIN_PASSWORD.encode("utf-8"),
    )


def _admin_response_or_session(
    request: Request,
) -> tuple[bool, str | None, Response | None]:
    if not _admin_enabled():
        return (
            False,
            None,
            JSONResponse(
                {
                    "ok": False,
                    "error": (
                        "Cookie management is disabled. Configure "
                        "YOUTUBE_MCP_ADMIN_PASSWORD and "
                        "YOUTUBE_MCP_SESSION_SECRET."
                    ),
                },
                status_code=503,
            ),
        )

    authenticated, csrf = _get_admin_session(request)

    if not authenticated:
        return (
            False,
            None,
            RedirectResponse("/admin/cookies/login", status_code=303),
        )

    return True, csrf, None


def _valid_csrf(form: Any, csrf: str | None) -> bool:
    submitted = str(form.get(COOKIE_CSRF_FIELD) or "")

    if not csrf or not submitted:
        return False

    return hmac.compare_digest(submitted, csrf)


# ---------------------------------------------------------------------------
# Admin HTML
# ---------------------------------------------------------------------------

def _admin_html(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{
  max-width: 900px;
  margin: 40px auto;
  padding: 0 18px;
  font-family: system-ui, sans-serif;
  background: #111;
  color: #eee;
  line-height: 1.5;
}}
.card {{
  background: #181818;
  border: 1px solid #333;
  border-radius: 12px;
  padding: 20px;
  margin: 16px 0;
}}
textarea {{
  width: 100%;
  min-height: 320px;
  box-sizing: border-box;
  resize: vertical;
  background: #0c0c0c;
  color: #eee;
  border: 1px solid #444;
  border-radius: 8px;
  padding: 12px;
  font-family: ui-monospace, monospace;
}}
input[type=password], input[type=file] {{
  max-width: 100%;
  box-sizing: border-box;
  padding: 10px;
}}
button {{
  padding: 10px 15px;
  border-radius: 8px;
  border: 1px solid #555;
  cursor: pointer;
}}
.small {{ color: #aaa; }}
.ok {{ color: #7ee787; }}
.err {{ color: #ff7b72; }}
code {{ background: #222; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def _login_page(error: str = "") -> HTMLResponse:
    error_block = (
        f'<p class="err">{html.escape(error)}</p>'
        if error
        else ""
    )

    body = f"""
<h1>YouTube MCP Admin</h1>
<div class="card">
  <h2>Login</h2>
  {error_block}
  <form method="post" action="/admin/cookies/login">
    <label for="password">Admin password</label><br><br>
    <input id="password" name="password" type="password"
           autocomplete="current-password" required>
    <br><br>
    <button type="submit">Login</button>
  </form>
</div>
"""

    return HTMLResponse(_admin_html("YouTube MCP Admin Login", body))


def _cookie_manager_page(request: Request) -> Response:
    authenticated, csrf, response = _admin_response_or_session(request)

    if response is not None:
        return response

    status = (
        '<span class="ok">Configured</span>'
        if _ACTIVE_COOKIES
        else '<span class="err">Not configured</span>'
    )

    body = f"""
<h1>YouTube MCP Cookie Manager</h1>

<div class="card">
  <h2>Current status</h2>
  <p>Active cookies: {status}</p>
  <p class="small">
    Cookie values are never displayed. Only a SHA-256 fingerprint and count
    are exposed to the admin UI/API.
  </p>
</div>

<div class="card">
  <h2>Upload Netscape cookie file</h2>
  <form method="post" action="/admin/cookies/update"
        enctype="multipart/form-data">
    <input type="hidden" name="{COOKIE_CSRF_FIELD}"
           value="{html.escape(csrf or '')}">
    <input type="file" name="cookie_file" accept=".txt,text/plain" required>
    <br><br>
    <label>
      <input type="checkbox" name="persist" value="1" checked>
      Persist encrypted on server
    </label>
    <br><br>
    <button type="submit">Verify &amp; Activate</button>
  </form>
</div>

<div class="card">
  <h2>Paste Netscape cookie data</h2>
  <p class="small">
    Required first line:
    <code># Netscape HTTP Cookie File</code>
  </p>
  <form method="post" action="/admin/cookies/update">
    <input type="hidden" name="{COOKIE_CSRF_FIELD}"
           value="{html.escape(csrf or '')}">
    <textarea name="cookie_text"
      placeholder="# Netscape HTTP Cookie File"></textarea>
    <br><br>
    <label>
      <input type="checkbox" name="persist" value="1" checked>
      Persist encrypted on server
    </label>
    <br><br>
    <button type="submit">Verify &amp; Activate</button>
  </form>
</div>

<div class="card">
  <h2>Actions</h2>
  <form method="post" action="/admin/cookies/verify" style="display:inline">
    <input type="hidden" name="{COOKIE_CSRF_FIELD}"
           value="{html.escape(csrf or '')}">
    <button type="submit">Verify Active Cookies</button>
  </form>

  <form method="post" action="/admin/cookies/clear"
        style="display:inline;margin-left:8px">
    <input type="hidden" name="{COOKIE_CSRF_FIELD}"
           value="{html.escape(csrf or '')}">
    <button type="submit">Clear Cookies</button>
  </form>

  <form method="post" action="/admin/logout"
        style="display:inline;margin-left:8px">
    <button type="submit">Logout</button>
  </form>
</div>

<div class="card">
  <h2>Security</h2>
  <ul>
    <li>YouTube cookies are not stored in browser localStorage/sessionStorage.</li>
    <li>The browser receives only an HttpOnly admin session cookie.</li>
    <li>Cookies are encrypted at rest with Fernet when persistence is enabled.</li>
    <li>Cookie values are never logged or returned by the API.</li>
    <li>Use a Railway Volume if encrypted persistence must survive restarts.</li>
  </ul>
</div>
"""

    return HTMLResponse(_admin_html("YouTube MCP Cookie Manager", body))


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------

async def admin_login_get(request: Request) -> Response:
    if not _admin_enabled():
        return JSONResponse(
            {
                "ok": False,
                "error": (
                    "Admin disabled. Set YOUTUBE_MCP_ADMIN_PASSWORD and "
                    "YOUTUBE_MCP_SESSION_SECRET."
                ),
            },
            status_code=503,
        )

    authenticated, _csrf = _get_admin_session(request)

    if authenticated:
        return RedirectResponse("/admin/cookies", status_code=303)

    return _login_page()


async def admin_login_post(request: Request) -> Response:
    if not _admin_enabled():
        return JSONResponse(
            {"ok": False, "error": "Admin disabled."},
            status_code=503,
        )

    form = await request.form()
    password = str(form.get("password") or "")

    if not _password_matches(password):
        logger.warning("Admin login failed")
        return _login_page("Invalid password.")

    token, _csrf = _new_admin_session()

    response = RedirectResponse("/admin/cookies", status_code=303)

    response.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        max_age=ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        secure=os.getenv("YOUTUBE_MCP_COOKIE_SECURE", "true").lower() != "false",
        samesite="strict",
        path="/admin",
    )

    logger.info("Admin login succeeded")
    return response


async def admin_logout(request: Request) -> Response:
    token = request.cookies.get(ADMIN_COOKIE_NAME)

    if token:
        _ADMIN_SESSIONS.pop(_session_hash(token), None)

    response = RedirectResponse("/admin/cookies/login", status_code=303)
    response.delete_cookie(ADMIN_COOKIE_NAME, path="/admin")

    logger.info("Admin logout")
    return response


async def admin_cookies(request: Request) -> Response:
    return _cookie_manager_page(request)


async def admin_cookie_update(request: Request) -> Response:
    _authenticated, csrf, response = _admin_response_or_session(request)

    if response is not None:
        return response

    form = await request.form()

    if not _valid_csrf(form, csrf):
        return JSONResponse(
            {"ok": False, "error": "Invalid CSRF token."},
            status_code=403,
        )

    persist = str(form.get("persist") or "") == "1"

    cookie_text = str(form.get("cookie_text") or "")
    upload = form.get("cookie_file")

    if upload is not None and getattr(upload, "filename", None):
        raw = await upload.read()

        if len(raw) > MAX_COOKIE_FILE_BYTES:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Cookie file is too large.",
                },
                status_code=413,
            )

        try:
            cookie_text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Cookie file must be UTF-8 text.",
                },
                status_code=400,
            )

    if not cookie_text.strip():
        return JSONResponse(
            {
                "ok": False,
                "error": "Provide a cookie file or paste cookie data.",
            },
            status_code=400,
        )

    global _ACTIVE_COOKIES, _ACTIVE_COOKIE_SHA256

    old_cookies = _ACTIVE_COOKIES
    old_hash = _ACTIVE_COOKIE_SHA256

    try:
        normalized, count = validate_netscape_cookies(cookie_text)

        # Activate only temporarily for verification.
        _ACTIVE_COOKIES = normalized
        _ACTIVE_COOKIE_SHA256 = _cookie_hash(normalized)

        verification = verify_active_cookies()

        if not verification.get("verified"):
            _ACTIVE_COOKIES = old_cookies
            _ACTIVE_COOKIE_SHA256 = old_hash

            logger.warning(
                "Cookie update rejected after verification | cookies=%d",
                count,
            )

            return JSONResponse(
                {
                    "ok": False,
                    "verified": False,
                    "error": (
                        "Cookie syntax is valid, but verification failed. "
                        "The new cookies were NOT persisted or activated."
                    ),
                    "verification": verification,
                },
                status_code=400,
            )

        if persist:
            save_encrypted_cookie_store(normalized)

        # Invalidate cached video data because cookie state changed.
        _CACHE.clear()

        logger.info(
            "Cookie update accepted | cookies=%d | persisted=%s | sha256=%s",
            count,
            persist,
            _cookie_hash(normalized),
        )

        return JSONResponse(
            {
                "ok": True,
                "verified": True,
                "cookie_count": count,
                "sha256": _cookie_hash(normalized),
                "persisted": persist,
                "message": (
                    "Cookies verified and activated. Cookie values are "
                    "intentionally never returned."
                ),
            }
        )

    except Exception as exc:
        _ACTIVE_COOKIES = old_cookies
        _ACTIVE_COOKIE_SHA256 = old_hash

        logger.warning(
            "Cookie update rejected | type=%s | error=%s",
            type(exc).__name__,
            str(exc)[:500],
        )

        return JSONResponse(
            {
                "ok": False,
                "verified": False,
                "error": str(exc),
            },
            status_code=400,
        )


async def admin_cookie_verify(request: Request) -> Response:
    _authenticated, csrf, response = _admin_response_or_session(request)

    if response is not None:
        return response

    form = await request.form()

    if not _valid_csrf(form, csrf):
        return JSONResponse(
            {"ok": False, "error": "Invalid CSRF token."},
            status_code=403,
        )

    return JSONResponse(verify_active_cookies())


async def admin_cookie_clear(request: Request) -> Response:
    _authenticated, csrf, response = _admin_response_or_session(request)

    if response is not None:
        return response

    form = await request.form()

    if not _valid_csrf(form, csrf):
        return JSONResponse(
            {"ok": False, "error": "Invalid CSRF token."},
            status_code=403,
        )

    clear_active_cookies(delete_persisted=True)
    _CACHE.clear()

    return RedirectResponse("/admin/cookies", status_code=303)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = MCPServer(SERVER_NAME)


@mcp.tool()
def get_youtube_video(
    url: str,
    preferred_languages: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fetch a YouTube video's metadata and timestamped transcript.

    Language behavior:
    - English is preferred by default.
    - If English is unavailable, the best available language is selected.
    - Manual captions are preferred over auto-generated captions.
    - Custom preference examples: ["hi", "en"] or ["gu", "hi", "en"].

    If administrator-approved YouTube cookies are active, the server uses
    them automatically for metadata/transcript extraction.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("The YouTube URL is required.")

    languages = normalize_preferred_languages(preferred_languages)

    logger.info(
        "MCP tool call | tool=get_youtube_video | url=%s | preferred=%s | cookies=%s",
        url,
        ",".join(languages),
        bool(_ACTIVE_COOKIES),
    )

    return build_result(url.strip(), languages)


# ---------------------------------------------------------------------------
# HTTP application
# ---------------------------------------------------------------------------

async def health(request: Request) -> Response:
    return JSONResponse(
        {
            "status": "ok",
            "service": SERVER_NAME,
            "mcp_endpoint": "/mcp",
            "admin_endpoint": "/admin/cookies",
            "cookie_active": bool(_ACTIVE_COOKIES),
        }
    )


def build_http_app():
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
    custom_domain = os.getenv("YOUTUBE_MCP_CUSTOM_DOMAIN", "").strip()

    allowed_hosts = [
        "localhost:*",
        "127.0.0.1:*",
        "[::1]:*",
    ]

    allowed_origins = [
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://[::1]:*",
    ]

    for domain in (railway_domain, custom_domain):
        if not domain:
            continue

        domain = domain.split("://", 1)[-1].split("/", 1)[0].strip()

        allowed_hosts.extend([domain, f"{domain}:*"])

        scheme = "https" if domain not in {
            "localhost",
            "127.0.0.1",
            "[::1]",
        } else "http"

        allowed_origins.append(f"{scheme}://{domain}")

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )

    custom_routes = [
        Route("/health", health, methods=["GET"]),
        Route("/admin/cookies/login", admin_login_get, methods=["GET"]),
        Route("/admin/cookies/login", admin_login_post, methods=["POST"]),
        Route("/admin/cookies", admin_cookies, methods=["GET"]),
        Route("/admin/cookies/update", admin_cookie_update, methods=["POST"]),
        Route("/admin/cookies/verify", admin_cookie_verify, methods=["POST"]),
        Route("/admin/cookies/clear", admin_cookie_clear, methods=["POST"]),
        Route("/admin/logout", admin_logout, methods=["POST"]),
    ]

    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=security,
        custom_starlette_routes=custom_routes,
    )

    logger.info(
        "HTTP server configured | port=%d | mcp=/mcp | admin=/admin/cookies",
        PORT,
    )
    logger.info(
        "Allowed hosts | %s",
        ", ".join(allowed_hosts),
    )

    return app


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def initialize() -> None:
    global _ACTIVE_COOKIES, _ACTIVE_COOKIE_SHA256

    if COOKIE_STORAGE_PATH.exists():
        loaded = load_encrypted_cookie_store()

        if loaded:
            try:
                normalized, _count = validate_netscape_cookies(loaded)
                _ACTIVE_COOKIES = normalized
                _ACTIVE_COOKIE_SHA256 = _cookie_hash(normalized)
            except Exception:
                logger.exception(
                    "Persisted cookies failed validation and were not activated."
                )

    if not ADMIN_PASSWORD:
        logger.warning(
            "YOUTUBE_MCP_ADMIN_PASSWORD is not configured; "
            "cookie management is disabled."
        )

    if ADMIN_PASSWORD and not SESSION_SECRET:
        logger.warning(
            "YOUTUBE_MCP_SESSION_SECRET is missing; "
            "cookie management is disabled."
        )

    if COOKIE_STORAGE_PATH.exists() and not COOKIE_ENCRYPTION_KEY:
        logger.warning(
            "Encrypted cookie store exists but YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY "
            "is missing; stored cookies cannot be loaded."
        )


def main() -> None:
    initialize()

    logger.info(
        "Starting %s | transport=%s | host=%s | port=%d | cookies=%s",
        SERVER_NAME,
        TRANSPORT,
        HOST,
        PORT,
        bool(_ACTIVE_COOKIES),
    )

    if TRANSPORT == "stdio":
        logger.info("Starting MCP over stdio")
        mcp.run()
        return

    if TRANSPORT not in {"streamable-http", "streamable_http"}:
        raise ValueError(
            "YOUTUBE_MCP_TRANSPORT must be 'streamable-http' or 'stdio'."
        )

    import uvicorn

    app = build_http_app()

    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()

    logger.info(
        "MCP endpoint | http://%s:%d/mcp | railway_domain=%s",
        HOST,
        PORT,
        railway_domain or "<not-set>",
    )

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info").lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
