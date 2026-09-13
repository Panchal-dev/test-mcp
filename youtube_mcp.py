#!/usr/bin/env python3
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

This server uses Streamable HTTP for local testing and Railway deployment.
The MCP endpoint is /mcp. Do not use print() for diagnostics; application logs
must stay on stderr so they never corrupt the MCP protocol channel.

Python: 3.10+
Install:
    pip install "mcp[cli]" youtube-transcript-api yt-dlp

Run locally:
    python youtube_mcp.py

Local MCP endpoint:
    http://localhost:8000/mcp

Railway:
    https://<your-railway-domain>/mcp
"""

from __future__ import annotations

import html
import logging
import os
import re
import tempfile
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings


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
_CACHE_LOCK = threading.Lock()


def cache_get(video_id: str) -> dict[str, Any] | None:
    with _CACHE_LOCK:
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
    with _CACHE_LOCK:
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

    Use this tool whenever the user gives a YouTube URL and asks to:
    summarize it, explain it, create notes, make study notes, answer questions
    about it, extract concepts, or otherwise understand the video.

    Language selection:
    - English is preferred by default.
    - If English is unavailable, the best available language is selected.
    - Manually created captions are preferred over auto-generated captions.
    - To prefer another language, pass e.g. ["hi", "en"] or ["gu", "hi", "en"].

    Returns:
    - title, channel, description, duration, dates, views/likes when available
    - selected transcript language and source
    - timestamped transcript segments
    - a ready-to-use timestamped text representation
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("The YouTube URL is required.")

    languages = normalize_preferred_languages(preferred_languages)

    logger.info(
        "MCP tool call | tool=get_youtube_video | url=%s | preferred=%s",
        url,
        ",".join(languages),
    )

    return build_result(url.strip(), languages)


def _csv_env(name: str) -> list[str]:
    value = os.getenv(name, "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_transport_security() -> TransportSecuritySettings:
    # Railway provides RAILWAY_PUBLIC_DOMAIN automatically.
    # MCP_ALLOWED_HOSTS can override it for a custom domain or another proxy.
    public_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip().lower()
    configured_hosts = _csv_env("MCP_ALLOWED_HOSTS")

    if configured_hosts:
        allowed_hosts = configured_hosts
    elif public_domain:
        allowed_hosts = [public_domain, f"{public_domain}:*"]
    else:
        allowed_hosts = [
            "localhost:*",
            "127.0.0.1:*",
            "[::1]:*",
        ]

    configured_origins = _csv_env("MCP_ALLOWED_ORIGINS")

    if configured_origins:
        allowed_origins = configured_origins
    elif public_domain:
        allowed_origins = [f"https://{public_domain}"]
    else:
        allowed_origins = [
            "http://localhost:*",
            "http://127.0.0.1:*",
            "http://[::1]:*",
        ]

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


if __name__ == "__main__":
    # Railway injects PORT. 8000 is the local-development fallback.
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    security = _build_transport_security()

    logger.info(
        "Starting %s | transport=streamable-http | host=%s | port=%d",
        SERVER_NAME,
        host,
        port,
    )
    logger.info(
        "MCP endpoint | /mcp | railway_domain=%s",
        os.getenv("RAILWAY_PUBLIC_DOMAIN", "not-set"),
    )
    logger.info("Allowed hosts | %s", ", ".join(security.allowed_hosts))

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        json_response=True,
        transport_security=security,
    )
