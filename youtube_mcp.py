#!/usr/bin/env python3
"""
YouTube Video MCP Server
------------------------
Personal-use MCP server for ChatGPT/other MCP clients.

Features
- Streamable HTTP MCP endpoint: /mcp
- Railway-friendly PORT + RAILWAY_PUBLIC_DOMAIN
- Dynamic Host allowlist (fixes 421 Invalid Host header)
- YouTube metadata via yt-dlp
- Transcript via youtube-transcript-api
- yt-dlp subtitle fallback
- Manual captions preferred; English preferred when available
- Timestamped transcript
- Secure admin cookie manager at /admin/cookies
- Netscape cookie file upload OR paste
- Cookie validation + lightweight YouTube verification
- Optional encrypted persistent cookie storage
- HttpOnly/Secure/SameSite admin session
- Structured, real-time logging
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import uvicorn
import yt_dlp
from cryptography.fernet import Fernet, InvalidToken
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "youtube-video-mcp"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))

PUBLIC_DOMAIN = (
    os.getenv("RAILWAY_PUBLIC_DOMAIN")
    or os.getenv("RAILWAY_STATIC_URL", "").replace("https://", "").replace("http://", "").rstrip("/")
    or ""
)

ADMIN_PASSWORD = os.getenv("YOUTUBE_MCP_ADMIN_PASSWORD", "")
SESSION_SECRET = os.getenv("YOUTUBE_MCP_SESSION_SECRET", "")
COOKIE_ENCRYPTION_KEY = os.getenv("YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY", "")

# Persistent location. On Railway, mount /app/data as a Volume if persistence
# across redeploys/restarts is required.
COOKIE_STORAGE_PATH = Path(
    os.getenv("YOUTUBE_MCP_COOKIE_STORAGE_PATH", "/app/data/youtube_cookies.enc")
)

MAX_COOKIE_BYTES = int(
    os.getenv("YOUTUBE_MCP_MAX_COOKIE_BYTES", str(2 * 1024 * 1024))
)
MAX_TRANSCRIPT_CHARS = int(
    os.getenv("YOUTUBE_MCP_MAX_TRANSCRIPT_CHARS", str(2_000_000))
)
MAX_DESCRIPTION_CHARS = int(
    os.getenv("YOUTUBE_MCP_MAX_DESCRIPTION_CHARS", str(20_000))
)
COOKIE_SESSION_TTL = int(
    os.getenv("YOUTUBE_MCP_SESSION_TTL", str(12 * 60 * 60))
)

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
}

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(APP_NAME)


# ---------------------------------------------------------------------------
# Security / cookie storage
# ---------------------------------------------------------------------------

ACTIVE_COOKIES_TEXT: str | None = None
ACTIVE_COOKIE_FINGERPRINT: str | None = None


def _fernet() -> Fernet | None:
    if not COOKIE_ENCRYPTION_KEY:
        return None
    try:
        # Validate key now so a bad Railway variable fails clearly.
        return Fernet(COOKIE_ENCRYPTION_KEY.encode("utf-8"))
    except Exception as exc:
        logger.error("Invalid YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY: %s", exc)
        return None


def cookie_fingerprint(cookie_text: str) -> str:
    return hashlib.sha256(cookie_text.encode("utf-8")).hexdigest()[:16]


def load_persisted_cookies() -> None:
    global ACTIVE_COOKIES_TEXT, ACTIVE_COOKIE_FINGERPRINT

    if not COOKIE_STORAGE_PATH.exists():
        logger.info("No persisted YouTube cookies found")
        return

    if not COOKIE_ENCRYPTION_KEY:
        logger.warning(
            "Persisted cookie file exists but encryption key is not configured; "
            "cookies will not be loaded"
        )
        return

    try:
        encrypted = COOKIE_STORAGE_PATH.read_bytes()
        cipher = _fernet()
        if cipher is None:
            return

        plaintext = cipher.decrypt(encrypted).decode("utf-8")
        validate_netscape_cookies(plaintext)

        ACTIVE_COOKIES_TEXT = plaintext
        ACTIVE_COOKIE_FINGERPRINT = cookie_fingerprint(plaintext)

        logger.info(
            "Loaded encrypted YouTube cookies | fingerprint=%s",
            ACTIVE_COOKIE_FINGERPRINT,
        )
    except (InvalidToken, UnicodeDecodeError, OSError, ValueError) as exc:
        logger.error("Could not load persisted cookies: %s", exc)
    except Exception:
        logger.exception("Unexpected error while loading persisted cookies")


def persist_cookies(cookie_text: str) -> None:
    if not COOKIE_ENCRYPTION_KEY:
        raise RuntimeError(
            "YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY is not configured. "
            "Cookies can be used for this process but cannot be persisted securely."
        )

    cipher = _fernet()
    if cipher is None:
        raise RuntimeError("Cookie encryption key is invalid")

    COOKIE_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    encrypted = cipher.encrypt(cookie_text.encode("utf-8"))

    # Atomic replace.
    temp_path = COOKIE_STORAGE_PATH.with_suffix(".tmp")
    temp_path.write_bytes(encrypted)
    os.replace(temp_path, COOKIE_STORAGE_PATH)

    try:
        os.chmod(COOKIE_STORAGE_PATH, 0o600)
    except OSError:
        pass


def clear_persisted_cookies() -> None:
    try:
        COOKIE_STORAGE_PATH.unlink(missing_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Could not delete stored cookies: {exc}") from exc


def activate_cookies(cookie_text: str, persist: bool = True) -> None:
    global ACTIVE_COOKIES_TEXT, ACTIVE_COOKIE_FINGERPRINT

    validate_netscape_cookies(cookie_text)

    if persist:
        persist_cookies(cookie_text)

    ACTIVE_COOKIES_TEXT = cookie_text
    ACTIVE_COOKIE_FINGERPRINT = cookie_fingerprint(cookie_text)

    logger.info(
        "YouTube cookies activated | fingerprint=%s | persistent=%s",
        ACTIVE_COOKIE_FINGERPRINT,
        persist,
    )


def clear_active_cookies() -> None:
    global ACTIVE_COOKIES_TEXT, ACTIVE_COOKIE_FINGERPRINT

    ACTIVE_COOKIES_TEXT = None
    ACTIVE_COOKIE_FINGERPRINT = None
    clear_persisted_cookies()

    logger.info("YouTube cookies cleared")


# ---------------------------------------------------------------------------
# Netscape cookie validation
# ---------------------------------------------------------------------------

def validate_netscape_cookies(cookie_text: str) -> None:
    if not cookie_text or not cookie_text.strip():
        raise ValueError("Cookie data is empty")

    raw = cookie_text.encode("utf-8")
    if len(raw) > MAX_COOKIE_BYTES:
        raise ValueError(
            f"Cookie data is too large. Maximum is {MAX_COOKIE_BYTES} bytes."
        )

    lines = cookie_text.splitlines()

    valid_rows = 0
    youtube_rows = 0

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("#"):
            # Netscape header/comment.
            continue

        parts = raw_line.split("\t")

        if len(parts) != 7:
            raise ValueError(
                f"Invalid Netscape cookie format on line {line_number}: "
                "expected 7 tab-separated fields"
            )

        domain, include_subdomains, path, secure, expiry, name, value = parts

        if not domain:
            raise ValueError(f"Missing cookie domain on line {line_number}")

        if include_subdomains not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"Invalid includeSubdomains value on line {line_number}: "
                f"{include_subdomains!r}"
            )

        if secure not in {"TRUE", "FALSE"}:
            raise ValueError(
                f"Invalid secure value on line {line_number}: {secure!r}"
            )

        if not path.startswith("/"):
            raise ValueError(
                f"Invalid cookie path on line {line_number}: {path!r}"
            )

        try:
            int(expiry)
        except ValueError as exc:
            raise ValueError(
                f"Invalid expiry timestamp on line {line_number}: {expiry!r}"
            ) from exc

        if not name:
            raise ValueError(f"Cookie name is empty on line {line_number}")

        valid_rows += 1

        normalized_domain = domain.lstrip(".").lower()
        if normalized_domain in {"youtube.com", "www.youtube.com"}:
            youtube_rows += 1

    if valid_rows == 0:
        raise ValueError("No Netscape cookie rows were found")

    if youtube_rows == 0:
        raise ValueError(
            "No YouTube cookies found. The file must contain youtube.com "
            "or www.youtube.com cookie entries."
        )


# ---------------------------------------------------------------------------
# YouTube URL / metadata helpers
# ---------------------------------------------------------------------------

def extract_video_id(url_or_id: str) -> str:
    value = (url_or_id or "").strip()

    if VIDEO_ID_RE.fullmatch(value):
        return value

    parsed = urlparse(value)
    host = parsed.netloc.lower().split(":")[0]

    if host not in YOUTUBE_HOSTS:
        raise ValueError("Please provide a valid YouTube URL or 11-character video ID")

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif parsed.path == "/watch":
        from urllib.parse import parse_qs

        video_id = parse_qs(parsed.query).get("v", [None])[0]
    else:
        match = re.search(
            r"/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})(?:[/?#]|$)",
            parsed.path,
        )
        video_id = match.group(1) if match else None

    if not video_id or not VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError(f"Could not extract a valid YouTube video ID from: {value}")

    return video_id


def format_timestamp(seconds: float | int) -> str:
    total = max(0, int(float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _cookie_tempfile() -> str | None:
    if not ACTIVE_COOKIES_TEXT:
        return None

    fd, path = tempfile.mkstemp(prefix="ytcookies_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(ACTIVE_COOKIES_TEXT)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def _base_ydl_options(cookie_file: str | None = None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "extract_flat": False,
        "socket_timeout": 20,
        "retries": 2,
        "fragment_retries": 2,
        "concurrent_fragment_downloads": 1,
    }

    if cookie_file:
        options["cookiefile"] = cookie_file

    return options


def fetch_metadata(video_id: str) -> dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookie_file = _cookie_tempfile()

    try:
        options = _base_ydl_options(cookie_file)

        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise RuntimeError("yt-dlp returned no video information")

        description = info.get("description") or ""

        return {
            "id": video_id,
            "url": url,
            "title": info.get("title"),
            "description": description[:MAX_DESCRIPTION_CHARS],
            "channel": info.get("channel"),
            "channel_id": info.get("channel_id"),
            "uploader": info.get("uploader"),
            "uploader_id": info.get("uploader_id"),
            "upload_date": info.get("upload_date"),
            "duration_seconds": info.get("duration"),
            "duration": (
                format_timestamp(info["duration"])
                if info.get("duration") is not None
                else None
            ),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
            "categories": info.get("categories"),
            "tags": info.get("tags"),
            "language": info.get("language"),
            "thumbnail": info.get("thumbnail"),
            "live_status": info.get("live_status"),
            "availability": info.get("availability"),
        }
    finally:
        if cookie_file:
            try:
                os.unlink(cookie_file)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Transcript extraction
# ---------------------------------------------------------------------------

def _snippet_values(snippet: Any) -> tuple[float, str]:
    if isinstance(snippet, dict):
        start = snippet.get("start", 0)
        text = snippet.get("text", "")
    else:
        start = getattr(snippet, "start", 0)
        text = getattr(snippet, "text", "")

    return float(start or 0), str(text or "").strip()


def _select_transcript(transcripts: list[Any]) -> Any:
    if not transcripts:
        raise ValueError("No transcripts available")

    # Prefer manually created English.
    for transcript in transcripts:
        code = str(getattr(transcript, "language_code", "")).lower()
        generated = bool(getattr(transcript, "is_generated", False))
        if code.startswith("en") and not generated:
            return transcript

    # Then auto-generated English.
    for transcript in transcripts:
        code = str(getattr(transcript, "language_code", "")).lower()
        if code.startswith("en"):
            return transcript

    # Otherwise prefer manually created first.
    for transcript in transcripts:
        if not bool(getattr(transcript, "is_generated", False)):
            return transcript

    # Finally: first available transcript.
    return transcripts[0]


def fetch_transcript_api(video_id: str) -> dict[str, Any]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    transcript_list = list(api.list(video_id))

    selected = _select_transcript(transcript_list)
    fetched = selected.fetch()

    rows: list[str] = []
    total_chars = 0

    for snippet in fetched:
        start, text = _snippet_values(snippet)
        if not text:
            continue

        line = f"[{format_timestamp(start)}] {text}"
        total_chars += len(line) + 1

        if total_chars > MAX_TRANSCRIPT_CHARS:
            rows.append("[TRANSCRIPT TRUNCATED: maximum size reached]")
            break

        rows.append(line)

    if not rows:
        raise ValueError("Selected transcript was empty")

    return {
        "source": "youtube-transcript-api",
        "language": getattr(selected, "language", None),
        "language_code": getattr(selected, "language_code", None),
        "is_generated": bool(getattr(selected, "is_generated", False)),
        "is_translatable": bool(getattr(selected, "is_translatable", False)),
        "transcript": "\n".join(rows),
    }


def fetch_transcript_ytdlp(video_id: str) -> dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookie_file = _cookie_tempfile()

    try:
        options = _base_ydl_options(cookie_file)
        options.update(
            {
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": ["en.*", ".*"],
                "subtitlesformat": "vtt",
            }
        )

        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise RuntimeError("yt-dlp returned no information")

        # yt-dlp exposes subtitle entries in info without downloading when
        # extraction has succeeded. Select English first, then any language.
        subtitles = info.get("subtitles") or {}
        automatic = info.get("automatic_captions") or {}

        selected_lang: str | None = None
        selected_entries: list[dict[str, Any]] | None = None
        generated = False

        def choose(entries_map: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]] | None]:
            if not entries_map:
                return None, None

            for lang, entries in entries_map.items():
                if lang.lower().startswith("en"):
                    return lang, entries

            first_lang = next(iter(entries_map))
            return first_lang, entries_map[first_lang]

        selected_lang, selected_entries = choose(subtitles)

        if selected_entries is None:
            selected_lang, selected_entries = choose(automatic)
            generated = True

        if not selected_entries:
            raise ValueError("yt-dlp found no usable subtitles")

        # yt-dlp does not always provide subtitle text directly in extract_info.
        # Download a single VTT subtitle to a temporary directory.
        with tempfile.TemporaryDirectory(prefix="yt_subs_") as tmpdir:
            sub_options = _base_ydl_options(cookie_file)
            sub_options.update(
                {
                    "writesubtitles": not generated,
                    "writeautomaticsub": generated,
                    "subtitleslangs": [selected_lang or "en"],
                    "subtitlesformat": "vtt",
                    "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
                }
            )

            with yt_dlp.YoutubeDL(sub_options) as ydl:
                ydl.download([url])

            files = list(Path(tmpdir).glob("*.vtt"))
            if not files:
                raise ValueError("yt-dlp did not produce a VTT subtitle file")

            vtt = files[0].read_text(encoding="utf-8", errors="replace")
            transcript = parse_vtt_to_timestamped_text(vtt)

        if not transcript.strip():
            raise ValueError("Downloaded VTT transcript was empty")

        return {
            "source": "yt-dlp",
            "language": selected_lang,
            "language_code": selected_lang,
            "is_generated": generated,
            "is_translatable": False,
            "transcript": transcript,
        }
    finally:
        if cookie_file:
            try:
                os.unlink(cookie_file)
            except OSError:
                pass


def parse_vtt_to_timestamped_text(vtt: str) -> str:
    lines = vtt.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    output: list[str] = []
    i = 0

    timestamp_re = re.compile(
        r"(?P<start>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+-->\s+"
        r"(?P<end>\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
    )

    while i < len(lines):
        line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        match = timestamp_re.search(line)
        if not match:
            i += 1
            continue

        start_text = match.group("start")
        start_seconds = vtt_timestamp_to_seconds(start_text)

        i += 1
        text_lines: list[str] = []

        while i < len(lines) and lines[i].strip():
            current = lines[i].strip()
            if not current.startswith(("NOTE", "STYLE", "REGION")):
                text_lines.append(re.sub(r"<[^>]+>", "", current))
            i += 1

        text = " ".join(text_lines).strip()
        text = re.sub(r"\s+", " ", text)

        if text:
            output.append(f"[{format_timestamp(start_seconds)}] {html.unescape(text)}")

        i += 1

    return "\n".join(output)


def vtt_timestamp_to_seconds(value: str) -> float:
    parts = value.split(":")
    if len(parts) != 3:
        return 0.0

    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def fetch_transcript(video_id: str) -> dict[str, Any]:
    errors: list[str] = []

    try:
        logger.info("Transcript attempt 1/2 | source=youtube-transcript-api | video=%s", video_id)
        result = fetch_transcript_api(video_id)
        logger.info(
            "Transcript success | source=%s | language=%s | generated=%s",
            result["source"],
            result.get("language_code"),
            result.get("is_generated"),
        )
        return result
    except Exception as exc:
        errors.append(f"youtube-transcript-api: {type(exc).__name__}: {exc}")
        logger.warning("Transcript API failed | %s", errors[-1])

    try:
        logger.info("Transcript attempt 2/2 | source=yt-dlp | video=%s", video_id)
        result = fetch_transcript_ytdlp(video_id)
        logger.info(
            "Transcript success | source=%s | language=%s | generated=%s",
            result["source"],
            result.get("language_code"),
            result.get("is_generated"),
        )
        return result
    except Exception as exc:
        errors.append(f"yt-dlp: {type(exc).__name__}: {exc}")
        logger.error("Transcript fallback failed | %s", errors[-1])

    raise RuntimeError(
        "Could not obtain a transcript. Attempts: " + " | ".join(errors)
    )


# ---------------------------------------------------------------------------
# Cookie verification
# ---------------------------------------------------------------------------

def verify_youtube_cookies(cookie_text: str) -> dict[str, Any]:
    """
    Validate the Netscape jar and perform a real YouTube request with it.

    A successful request proves that yt-dlp can parse/use the cookie jar.
    It does not guarantee that every YouTube account feature is authenticated.
    """
    validate_netscape_cookies(cookie_text)

    fd, path = tempfile.mkstemp(prefix="verify_cookies_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(cookie_text)

        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

        options = _base_ydl_options(path)

        # Use a normal public YouTube video to verify that yt-dlp can actually
        # open YouTube with the supplied cookie jar.
        verification_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(verification_url, download=False)

        if not info:
            raise RuntimeError("YouTube returned no video information")

        return {
            "ok": True,
            "message": "Cookie file is valid and yt-dlp successfully used it with YouTube.",
            "video_id": info.get("id"),
            "cookie_fingerprint": cookie_fingerprint(cookie_text),
        }
    except Exception as exc:
        logger.warning("Cookie verification failed | %s", exc)
        return {
            "ok": False,
            "message": f"Cookie verification failed: {type(exc).__name__}: {exc}",
            "cookie_fingerprint": cookie_fingerprint(cookie_text),
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Admin session
# ---------------------------------------------------------------------------

def session_signature(timestamp: int) -> str:
    message = str(timestamp).encode("utf-8")
    key = SESSION_SECRET.encode("utf-8")
    return hashlib.sha256(key + b":" + message).hexdigest()


def make_session_value() -> str:
    timestamp = int(time.time())
    nonce = secrets.token_urlsafe(16)
    signature = session_signature(timestamp)
    return f"{timestamp}.{nonce}.{signature}"


def valid_session(request: Request) -> bool:
    if not SESSION_SECRET:
        return False

    value = request.cookies.get("youtube_mcp_admin")
    if not value:
        return False

    parts = value.split(".")
    if len(parts) != 3:
        return False

    timestamp_text, nonce, signature = parts

    try:
        timestamp = int(timestamp_text)
    except ValueError:
        return False

    if not nonce or not signature:
        return False

    if abs(int(time.time()) - timestamp) > COOKIE_SESSION_TTL:
        return False

    expected = session_signature(timestamp)
    return secrets.compare_digest(signature, expected)


def require_admin_configured() -> None:
    if not ADMIN_PASSWORD:
        raise RuntimeError(
            "YOUTUBE_MCP_ADMIN_PASSWORD is not configured on the server."
        )
    if not SESSION_SECRET:
        raise RuntimeError(
            "YOUTUBE_MCP_SESSION_SECRET is not configured on the server."
        )


# ---------------------------------------------------------------------------
# FastMCP
# ---------------------------------------------------------------------------

allowed_hosts = [
    "localhost:*",
    "127.0.0.1:*",
]

if PUBLIC_DOMAIN:
    allowed_hosts.extend(
        [
            PUBLIC_DOMAIN,
            f"{PUBLIC_DOMAIN}:*",
        ]
    )

allowed_origins = [
    "http://localhost:*",
    "http://127.0.0.1:*",
]

if PUBLIC_DOMAIN:
    allowed_origins.extend(
        [
            f"https://{PUBLIC_DOMAIN}",
            f"http://{PUBLIC_DOMAIN}",
            f"https://{PUBLIC_DOMAIN}:*",
            f"http://{PUBLIC_DOMAIN}:*",
        ]
    )

transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=allowed_hosts,
    allowed_origins=allowed_origins,
)

mcp = FastMCP(
    APP_NAME,
    host=HOST,
    port=PORT,
    transport_security=transport_security,
)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_youtube_video(
    url: str,
    include_transcript: bool = True,
) -> dict[str, Any]:
    """
    Fetch YouTube video metadata and, when available, a timestamped transcript.

    Give a YouTube URL. The result is designed for an AI client to summarize,
    explain, answer questions, or turn the video into notes.
    """
    video_id = extract_video_id(url)

    logger.info(
        "Tool get_youtube_video | video=%s | transcript=%s | cookies=%s",
        video_id,
        include_transcript,
        bool(ACTIVE_COOKIES_TEXT),
    )

    metadata_error: str | None = None
    transcript_error: str | None = None

    try:
        metadata = fetch_metadata(video_id)
    except Exception as exc:
        metadata_error = f"{type(exc).__name__}: {exc}"
        logger.error("Metadata failed | video=%s | %s", video_id, metadata_error)
        metadata = {
            "id": video_id,
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }

    result: dict[str, Any] = {
        "video_id": video_id,
        "metadata": metadata,
        "transcript": None,
        "transcript_available": False,
    }

    if metadata_error:
        result["metadata_error"] = metadata_error

    if include_transcript:
        try:
            transcript = fetch_transcript(video_id)
            result["transcript"] = transcript
            result["transcript_available"] = True
        except Exception as exc:
            transcript_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Transcript failed | video=%s | %s",
                video_id,
                transcript_error,
            )
            result["transcript_error"] = transcript_error

    if not result["transcript_available"] and metadata_error:
        raise RuntimeError(
            f"Could not retrieve useful YouTube data. "
            f"Metadata error: {metadata_error}. "
            f"Transcript error: {transcript_error or 'not requested'}."
        )

    return result


@mcp.tool()
def get_youtube_transcript(url: str) -> dict[str, Any]:
    """
    Return only the timestamped transcript for a YouTube video.
    """
    video_id = extract_video_id(url)
    logger.info("Tool get_youtube_transcript | video=%s", video_id)
    return fetch_transcript(video_id)


@mcp.tool()
def get_youtube_metadata(url: str) -> dict[str, Any]:
    """
    Return YouTube title, description, channel, duration, views, thumbnail,
    upload date, and other available metadata.
    """
    video_id = extract_video_id(url)
    logger.info("Tool get_youtube_metadata | video=%s", video_id)
    return fetch_metadata(video_id)


# ---------------------------------------------------------------------------
# Custom HTTP routes
# ---------------------------------------------------------------------------

@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": APP_NAME,
            "mcp_endpoint": "/mcp",
            "cookies_active": bool(ACTIVE_COOKIES_TEXT),
        }
    )


@mcp.custom_route("/admin", methods=["GET"])
async def admin_redirect(request: Request) -> RedirectResponse:
    return RedirectResponse("/admin/cookies", status_code=303)


def admin_page(message: str = "", error: str = "") -> str:
    message_html = (
        f'<div class="message success">{html.escape(message)}</div>' if message else ""
    )
    error_html = (
        f'<div class="message error">{html.escape(error)}</div>' if error else ""
    )

    cookie_status = (
        "ACTIVE"
        if ACTIVE_COOKIES_TEXT
        else "NOT ACTIVE"
    )

    fingerprint = ACTIVE_COOKIE_FINGERPRINT or "—"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YouTube MCP Cookie Manager</title>
<style>
body {{
    font-family: system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    line-height: 1.5;
    background: #f7f7f8;
    color: #171717;
}}
.card {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}}
textarea {{
    width: 100%;
    min-height: 300px;
    box-sizing: border-box;
    font-family: ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size: 13px;
    padding: 12px;
    border: 1px solid #bbb;
    border-radius: 8px;
}}
input[type=password],input[type=file] {{
    width: 100%;
    box-sizing: border-box;
    padding: 11px;
    margin: 7px 0 15px;
}}
button {{
    padding: 11px 18px;
    border: 0;
    border-radius: 8px;
    cursor: pointer;
    margin-right: 8px;
}}
.primary {{ background:#111; color:#fff; }}
.danger {{ background:#b42318; color:#fff; }}
.message {{
    padding: 12px;
    border-radius: 8px;
    margin: 12px 0;
}}
.success {{ background:#e8f7ee; color:#146c3a; }}
.error {{ background:#fdecec; color:#9b1c1c; }}
.small {{ color:#666; font-size:14px; }}
code {{ background:#eee; padding:2px 5px; border-radius:4px; }}
</style>
</head>
<body>

<div class="card">
<h1>YouTube MCP Cookie Manager</h1>
<p class="small">Cookie status: <strong>{cookie_status}</strong></p>
<p class="small">Fingerprint: <code>{html.escape(fingerprint)}</code></p>
{message_html}
{error_html}
</div>

<div class="card">
<h2>Upload Netscape cookie file</h2>
<form method="post" action="/admin/cookies/upload" enctype="multipart/form-data">
<label>Cookie <code>.txt</code> file</label>
<input type="file" name="cookie_file" accept=".txt,text/plain" required>
<button class="primary" type="submit">Verify &amp; Activate</button>
</form>
</div>

<div class="card">
<h2>Paste Netscape cookie data</h2>
<p class="small">
Paste the complete Netscape cookie content. Each cookie row must have
7 tab-separated fields.
</p>
<form method="post" action="/admin/cookies/paste">
<textarea name="cookie_text" placeholder="# Netscape HTTP Cookie File
.youtube.com	TRUE	/	TRUE	0	COOKIE_NAME	COOKIE_VALUE"></textarea>
<br><br>
<button class="primary" type="submit">Verify &amp; Activate</button>
</form>
</div>

<div class="card">
<h2>Current cookies</h2>
<p class="small">
Cookie values are never displayed. Only the fingerprint is shown.
</p>
<form method="post" action="/admin/cookies/clear"
      onsubmit="return confirm('Clear active and persisted YouTube cookies?')">
<button class="danger" type="submit">Clear Cookies</button>
</form>
</div>

<div class="card">
<p class="small">
MCP endpoint: <code>/mcp</code><br>
Health endpoint: <code>/health</code>
</p>
</div>

</body>
</html>"""


@mcp.custom_route("/admin/cookies", methods=["GET"])
async def admin_cookies_get(request: Request) -> Response:
    if not ADMIN_PASSWORD or not SESSION_SECRET:
        return HTMLResponse(
            "<h1>Admin configuration incomplete</h1>"
            "<p>Set YOUTUBE_MCP_ADMIN_PASSWORD and "
            "YOUTUBE_MCP_SESSION_SECRET in Railway variables.</p>",
            status_code=503,
        )

    if not valid_session(request):
        return HTMLResponse(
            """<!doctype html>
<html><head><meta charset="utf-8"><title>Admin Login</title>
<style>
body{font-family:system-ui;max-width:500px;margin:60px auto;padding:20px}
input,button{width:100%;box-sizing:border-box;padding:12px;margin:8px 0}
button{background:#111;color:white;border:0;border-radius:8px;cursor:pointer}
</style></head>
<body>
<h1>Admin Login</h1>
<form method="post" action="/admin/login">
<input type="password" name="password" placeholder="Admin password" required>
<button type="submit">Login</button>
</form>
</body></html>""",
            status_code=200,
        )

    return HTMLResponse(admin_page())


@mcp.custom_route("/admin/login", methods=["POST"])
async def admin_login(request: Request) -> Response:
    try:
        require_admin_configured()
        form = await request.form()
        password = str(form.get("password") or "")

        if not secrets.compare_digest(password, ADMIN_PASSWORD):
            logger.warning("Rejected admin login attempt")
            return HTMLResponse(
                """<h1>Access denied</h1>
<p>Invalid credentials.</p>
<p><a href="/admin/cookies">Back</a></p>""",
                status_code=401,
            )

        response = RedirectResponse("/admin/cookies", status_code=303)

        response.set_cookie(
            "youtube_mcp_admin",
            make_session_value(),
            max_age=COOKIE_SESSION_TTL,
            httponly=True,
            secure=True,
            samesite="strict",
            path="/admin",
        )

        logger.info("Admin login successful")
        return response
    except Exception as exc:
        logger.error("Admin login error | %s", exc)
        return HTMLResponse(
            "<h1>Server configuration error</h1>",
            status_code=500,
        )


async def read_uploaded_cookie_file(request: Request) -> str:
    form = await request.form()
    uploaded = form.get("cookie_file")

    if uploaded is None:
        raise ValueError("No cookie file was uploaded")

    # Starlette UploadFile interface.
    filename = getattr(uploaded, "filename", "") or ""
    if not filename.lower().endswith(".txt"):
        raise ValueError("Only .txt cookie files are accepted")

    data = await uploaded.read()

    if len(data) > MAX_COOKIE_BYTES:
        raise ValueError(
            f"Cookie file exceeds {MAX_COOKIE_BYTES} bytes"
        )

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Cookie file must be UTF-8 text") from exc


async def process_new_cookies(
    request: Request,
    cookie_text: str,
) -> Response:
    if not valid_session(request):
        return HTMLResponse("Unauthorized", status_code=401)

    try:
        validate_netscape_cookies(cookie_text)

        logger.info(
            "Starting cookie verification | fingerprint=%s",
            cookie_fingerprint(cookie_text),
        )

        verification = verify_youtube_cookies(cookie_text)

        if not verification["ok"]:
            return HTMLResponse(
                admin_page(error=verification["message"]),
                status_code=400,
            )

        # Only activate after successful verification.
        try:
            activate_cookies(cookie_text, persist=True)
        except RuntimeError as persist_exc:
            # Still allow process-level activation if persistence was not
            # configured, but make it explicit to the admin.
            if "COOKIE_ENCRYPTION_KEY" in str(persist_exc):
                activate_cookies(cookie_text, persist=False)
                return HTMLResponse(
                    admin_page(
                        message=(
                            "Cookies verified and activated for the current "
                            "process. They were NOT persisted because "
                            "YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY is not configured."
                        )
                    )
                )
            raise

        return HTMLResponse(
            admin_page(
                message=(
                    "Cookies verified successfully and activated. "
                    f"Fingerprint: {ACTIVE_COOKIE_FINGERPRINT}"
                )
            )
        )
    except Exception as exc:
        logger.error("Cookie update failed | %s", exc)
        return HTMLResponse(
            admin_page(error=f"{type(exc).__name__}: {exc}"),
            status_code=400,
        )


@mcp.custom_route("/admin/cookies/upload", methods=["POST"])
async def admin_cookies_upload(request: Request) -> Response:
    if not valid_session(request):
        return HTMLResponse("Unauthorized", status_code=401)

    try:
        cookie_text = await read_uploaded_cookie_file(request)
        return await process_new_cookies(request, cookie_text)
    except Exception as exc:
        logger.error("Cookie file upload failed | %s", exc)
        return HTMLResponse(
            admin_page(error=f"{type(exc).__name__}: {exc}"),
            status_code=400,
        )


@mcp.custom_route("/admin/cookies/paste", methods=["POST"])
async def admin_cookies_paste(request: Request) -> Response:
    if not valid_session(request):
        return HTMLResponse("Unauthorized", status_code=401)

    try:
        form = await request.form()
        cookie_text = str(form.get("cookie_text") or "")

        if len(cookie_text.encode("utf-8")) > MAX_COOKIE_BYTES:
            raise ValueError(
                f"Cookie data exceeds {MAX_COOKIE_BYTES} bytes"
            )

        return await process_new_cookies(request, cookie_text)
    except Exception as exc:
        logger.error("Cookie paste failed | %s", exc)
        return HTMLResponse(
            admin_page(error=f"{type(exc).__name__}: {exc}"),
            status_code=400,
        )


@mcp.custom_route("/admin/cookies/clear", methods=["POST"])
async def admin_cookies_clear(request: Request) -> Response:
    if not valid_session(request):
        return HTMLResponse("Unauthorized", status_code=401)

    try:
        clear_active_cookies()
        return HTMLResponse(
            admin_page(message="Active and persisted YouTube cookies cleared.")
        )
    except Exception as exc:
        logger.error("Cookie clear failed | %s", exc)
        return HTMLResponse(
            admin_page(error=f"{type(exc).__name__}: {exc}"),
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def initialize() -> None:
    logger.info(
        "Starting %s | transport=streamable-http | host=%s | port=%s",
        APP_NAME,
        HOST,
        PORT,
    )

    if PUBLIC_DOMAIN:
        logger.info("Railway public domain | %s", PUBLIC_DOMAIN)
    else:
        logger.info(
            "RAILWAY_PUBLIC_DOMAIN not set; localhost-only host validation "
            "will be configured"
        )

    if ADMIN_PASSWORD and SESSION_SECRET:
        logger.info("Admin cookie manager | configured")
    else:
        logger.warning(
            "Admin cookie manager is not fully configured. "
            "Set YOUTUBE_MCP_ADMIN_PASSWORD and "
            "YOUTUBE_MCP_SESSION_SECRET."
        )

    if COOKIE_ENCRYPTION_KEY:
        if _fernet() is None:
            raise RuntimeError(
                "YOUTUBE_MCP_COOKIE_ENCRYPTION_KEY is invalid"
            )
        logger.info("Encrypted cookie persistence | configured")
    else:
        logger.warning(
            "Cookie encryption key is not configured. "
            "Cookies cannot persist across restarts."
        )

    load_persisted_cookies()

    logger.info("MCP endpoint | /mcp")
    logger.info("Admin cookie manager | /admin/cookies")
    logger.info("Health endpoint | /health")
    logger.info(
        "Allowed hosts | %s",
        ", ".join(allowed_hosts),
    )


def main() -> None:
    initialize()

    # IMPORTANT:
    # Do not call streamable_http_app(custom_starlette_routes=...).
    # The current MCP SDK exposes custom routes through @mcp.custom_route(),
    # and mcp.run() serves those routes together with /mcp.
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
