"""Your Personal AI Chatbot - a Streamlit chat app built on LangChain + OpenAI."""

from __future__ import annotations

import base64
import html
import io
import json
import mimetypes
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import AuthenticationError, OpenAIError

# ---------------- CONFIGURATION ----------------

APP_TITLE = "Your Personal AI Chatbot"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")
BASE_DIR = Path(__file__).parent
HISTORY_FILE = BASE_DIR / "chat_history.json"
UPLOAD_DIR = BASE_DIR / "chat_uploads"  # uploaded images are stored here

MAX_CONTEXT_MESSAGES = 40  # only the latest N messages are sent to the model
MAX_TITLE_LENGTH = 40
MAX_RENAME_LENGTH = 60

# File upload limits
MAX_FILES_PER_MESSAGE = 5
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file
MAX_ATTACHMENT_CHARS = 60_000  # text extracted from one file is cut after this
MAX_IMAGES_IN_CONTEXT = 4  # older images are mentioned by name only

TEXT_EXTENSIONS = {
    "txt", "md", "csv", "json", "py", "js", "ts", "html", "css",
    "xml", "yaml", "yml", "log", "sql", "ini", "toml",
}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
DOCUMENT_EXTENSIONS = {"pdf", "docx"}
ACCEPTED_FILE_TYPES = sorted(TEXT_EXTENSIONS | IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS)

DEFAULT_FILE_PROMPT = "Please review the attached file(s) and summarize the key points."

# Always used as the system role. Users can still ask for a different role
# simply by writing it in their message (e.g. "You are a math teacher...").
SYSTEM_ROLE = (
    "You are a helpful, professional AI assistant. "
    "Explain concepts clearly and naturally. "
    "When you write math or formulas, use LaTeX with $...$ for inline math and "
    "$$...$$ for display equations (never \\( \\) or \\[ \\]), and put units inside \\text{}."
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- STYLES (White + Orange + Royal Blue) ----------------

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --blue: #1F4FD8;
    --blue-dark: #1A3FB0;
    --blue-soft: #EEF3FF;
    --orange: #F97316;
    --orange-dark: #EA580C;
    --orange-soft: #FFF3E8;
    --ink: #172033;
    --muted: #64748B;
    --line: #E2E8F0;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #FFFFFF; color: var(--ink); }

/* Header: keep it, but remove Deploy / toolbar clutter */
[data-testid="stHeader"] { background: rgba(255,255,255,0.95); }
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"], .stAppDeployButton, .stDeployButton,
[data-testid="stMainMenu"], #MainMenu, footer { display: none !important; visibility: hidden; }
/* Button that CLOSES the sidebar (inside the sidebar, top right) */
[data-testid="stSidebarCollapseButton"] button { color: var(--blue) !important; }
[data-testid="stSidebarCollapseButton"] button:hover { background: var(--blue-soft) !important; }

/* Button that OPENS the sidebar again (top-left of the page, shown only while the sidebar is closed).
   Made big, orange and fixed in place so it is always easy to find. */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    position: fixed !important; top: 0.8rem; left: 0.8rem; z-index: 999999;
}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button {
    background: var(--orange) !important; color: #FFFFFF !important; border-radius: 10px !important;
    width: 42px; height: 42px; box-shadow: 0 4px 14px rgba(249,115,22,0.35);
}
[data-testid="stExpandSidebarButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover { background: var(--orange-dark) !important; }
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg { fill: #FFFFFF !important; color: #FFFFFF !important; }

/* Top padding must clear the fixed Streamlit header, otherwise the brand gets clipped */
.block-container { max-width: 1050px; padding-top: 4.2rem; padding-bottom: 6rem; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #F6F8FC; border-right: 1px solid var(--line); }
[data-testid="stSidebar"][aria-expanded="true"] { min-width: 320px !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }

/* Streamlit pulls custom HTML blocks up slightly; padding keeps labels from being overlapped */
.brand { color: var(--ink); font-size: 21px; font-weight: 700; letter-spacing: -0.6px; line-height: 1.4; }
[data-testid="stSidebar"] .brand { padding-bottom: 1.5rem; }
.brand .blue { color: var(--blue); }
.brand .dot { color: var(--orange); }

.eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 1.5px; color: var(--orange-dark); }
.hero-accent { width: 46px; height: 4px; border-radius: 4px; background: var(--orange); margin: 6px 0 14px; }
.hero-title { font-size: clamp(30px, 4vw, 43px); font-weight: 700; letter-spacing: -1.5px; line-height: 1.15; color: var(--ink); }
.hero-title .blue { color: var(--blue); }
.hero-description { color: var(--muted); font-size: 14px; line-height: 1.8; }
.section-label { color: #8290A5; font-size: 11px; font-weight: 700; letter-spacing: 1.3px; margin-top: 10px; padding-bottom: calc(1rem + 2px); }
.group-label { color: var(--muted); font-size: 12px; font-weight: 600; margin-left: 2px; padding-bottom: calc(1rem + 2px); }

.card { border: 1px solid var(--line); border-top: 4px solid var(--orange); border-radius: 16px; padding: 22px 24px 12px; }
.card h2 { color: var(--ink); margin-top: 10px; }
.card p.muted { color: var(--muted); font-size: 14px; }

/* ---------- Buttons ---------- */
.stButton button, .stFormSubmitButton button {
    min-height: 40px; border-radius: 9px; font-weight: 600; transition: 0.15s ease;
}

/* primary = royal blue */
.stButton :is(button[kind="primary"], button[data-testid="stBaseButton-primary"]),
.stFormSubmitButton button {
    background: var(--blue) !important; color: #FFFFFF !important; border: 1px solid var(--blue) !important;
}
.stButton :is(button[kind="primary"], button[data-testid="stBaseButton-primary"]):hover,
.stFormSubmitButton button:hover {
    background: var(--blue-dark) !important; border-color: var(--blue-dark) !important;
}

/* secondary = white with blue outline */
.stButton :is(button[kind="secondary"], button[data-testid="stBaseButton-secondary"]) {
    background: #FFFFFF; color: var(--blue); border: 1px solid #C9D6F7;
}
.stButton :is(button[kind="secondary"], button[data-testid="stBaseButton-secondary"]):hover {
    background: var(--blue-soft); border-color: var(--blue); color: var(--blue);
}

/* ---------- Chat history rows ---------- */
[class*="st-key-select_"] button,
[class*="st-key-active_"] button {
    min-height: 44px; padding: 0 12px; border-radius: 11px; font-weight: 500;
}
/* force content to the left (Streamlit centres button content by default) */
[class*="st-key-select_"] button > div,
[class*="st-key-active_"] button > div {
    justify-content: flex-start !important; width: 100%; gap: 8px;
}
[class*="st-key-select_"] button [data-testid="stMarkdownContainer"],
[class*="st-key-active_"] button [data-testid="stMarkdownContainer"] { min-width: 0; flex: 1; text-align: left; }
[class*="st-key-select_"] button p,
[class*="st-key-active_"] button p {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: left; margin: 0;
}

/* normal row: white card */
[class*="st-key-select_"] button {
    background: #FFFFFF !important; color: #334155 !important; border: 1px solid #E8ECF3 !important;
}
[class*="st-key-select_"] button [data-testid="stIconMaterial"] { color: var(--blue); }
[class*="st-key-select_"] button:hover {
    background: var(--blue-soft) !important; border-color: #C9D6F7 !important;
    box-shadow: 0 3px 10px rgba(31,79,216,0.08);
}

/* active row: soft blue with orange marker */
[class*="st-key-active_"] button {
    background: var(--blue-soft) !important; color: var(--blue) !important; font-weight: 600;
    border: 1px solid #C9D6F7 !important; border-left: 4px solid var(--orange) !important;
}
[class*="st-key-active_"] button [data-testid="stIconMaterial"] { color: var(--orange); }

/* "..." menu (rename / delete) */
[data-testid="stSidebar"] [data-testid="stPopover"] { width: 100%; }
[data-testid="stSidebar"] [data-testid="stPopover"] button {
    width: 100%; min-height: 44px; padding: 0; justify-content: center;
    background: transparent !important; border: 1px solid transparent !important; color: #94A3B8 !important;
}
[data-testid="stSidebar"] [data-testid="stPopover"] button:hover { background: var(--orange-soft) !important; color: var(--orange-dark) !important; }
/* hide the little chevron next to the icon */
[data-testid="stSidebar"] [data-testid="stPopover"] button > :last-child:not(:first-child),
[data-testid="stSidebar"] [data-testid="stPopover"] button > div > :last-child:not(:first-child) { display: none; }
[data-testid="stPopoverBody"] { min-width: 280px; }

[class*="st-key-del_"] button { background: #FFFFFF !important; color: var(--orange-dark) !important; border: 1px solid var(--orange) !important; }
[class*="st-key-del_"] button:hover { background: var(--orange) !important; color: #FFFFFF !important; }

/* logout = orange outline */
.st-key-logout button { background: #FFFFFF !important; color: var(--orange-dark) !important; border: 1px solid var(--orange) !important; }
.st-key-logout button:hover { background: var(--orange) !important; color: #FFFFFF !important; }

/* keep Model + Log out pinned to the very bottom of the sidebar:
   the sidebar content fills the full height, and the footer takes all the free space above it */
[data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
[data-testid="stSidebarUserContent"] {
    flex: 1 1 auto; display: flex; flex-direction: column;
    padding-bottom: 1.2rem !important; /* Streamlit's default 6rem bottom padding left a big gap */
}
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] {
    flex: 1 1 auto; min-height: calc(100vh - 9.5rem);
}
.st-key-sidebar_footer { margin-top: auto; }

/* ---------- Inputs ---------- */
.stTextInput label { color: #344054 !important; font-weight: 600; }

[data-testid="stTextInput"] div[data-baseweb="input"] {
    background: #FFFFFF !important; border: 1px solid #CBD5E1 !important; border-radius: 9px !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"] > div,
[data-testid="stTextInput"] div[data-baseweb="base-input"] { background: #FFFFFF !important; }
[data-testid="stTextInput"] input {
    border: none !important; box-shadow: none !important; background: transparent !important; color: var(--ink) !important;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
    border-color: var(--blue) !important; box-shadow: 0 0 0 1px var(--blue) !important;
}

/* hide the "Press Enter to submit form" hint that overlaps typed text */
[data-testid="InputInstructions"] { display: none !important; }
/* hide Edge/IE built-in password eye so only one eye icon shows */
input::-ms-reveal, input::-ms-clear { display: none; }

div[data-testid="stForm"] { background: #FFFFFF; border: 1px solid var(--line); border-radius: 16px; padding: 22px; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: #FFFFFF; border: 1px solid #E8ECF3; border-radius: 14px;
    padding: 16px 19px; margin-bottom: 13px; box-shadow: 0 2px 8px rgba(23,32,51,0.025);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: var(--blue-soft); border-color: #D9E3FB; }
[data-testid="chatAvatarIcon-user"] { background: var(--orange) !important; }
[data-testid="chatAvatarIcon-assistant"] { background: var(--blue) !important; }
[data-testid="stChatMessage"] [data-testid="stImage"] img { border-radius: 10px; border: 1px solid var(--line); }

.file-chip {
    display: inline-block; margin: 4px 6px 0 0; padding: 4px 11px; border-radius: 999px;
    background: #FFFFFF; border: 1px solid #C9D6F7; color: var(--blue); font-size: 12.5px; font-weight: 600;
}

/* chat box: white background (Streamlit default is grey) */
[data-testid="stBottom"] > div { background: #FFFFFF; }
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea { background: #FFFFFF !important; color: var(--ink); }
[data-testid="stChatInput"] { border: 1px solid #CBD5E1; border-radius: 13px; box-shadow: 0 4px 18px rgba(23,32,51,0.06); }
[data-testid="stChatInput"]:focus-within { border-color: var(--blue); box-shadow: 0 4px 18px rgba(31,79,216,0.12); }
/* remove Streamlit's default red inner outline so only our blue border shows */
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="base-input"] { border-color: transparent !important; box-shadow: none !important; outline: none !important; }
[data-testid="stChatInputSubmitButton"], [data-testid="stChatInput"] button[kind="primary"] { background: var(--orange) !important; color: #FFFFFF !important; border-radius: 9px; }
[data-testid="stChatInputSubmitButton"]:hover { background: var(--orange-dark) !important; }

.topline { padding-bottom: 15px; border-bottom: 1px solid #E8ECF3; margin-bottom: 25px; }
.topline .title { font-size: 20px; font-weight: 700; color: var(--ink); }
.topline .title span { color: var(--blue); }
.topline .title .dot { color: var(--orange); }

/* ---------- Welcome block (fades out smoothly when a reply is being generated) ---------- */
.welcome-wrap { padding: 3rem 0 1rem; overflow: hidden; }
.welcome-wrap.vanish { animation: welcomeOut 0.6s ease forwards; }
@keyframes welcomeOut {
    0%   { opacity: 1; max-height: 460px; transform: translateY(0); }
    55%  { opacity: 0; max-height: 460px; transform: translateY(-10px); }
    100% { opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0; transform: translateY(-10px); }
}
.welcome-title { color: var(--ink); font-size: clamp(27px, 4vw, 36px); font-weight: 700; letter-spacing: -1.2px; text-align: center; }
.welcome-description { color: var(--muted); text-align: center; font-size: 14px; margin: 8px 0 26px; }
.welcome-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
@media (max-width: 720px) { .welcome-grid { grid-template-columns: 1fr; } }

.suggestion { border: 1px solid var(--line); border-top: 3px solid var(--blue); border-radius: 12px; padding: 15px; min-height: 82px; background: #FFFFFF; transition: 0.15s ease; }
.suggestion:hover { border-top-color: var(--orange); box-shadow: 0 6px 18px rgba(23,32,51,0.06); }
.suggestion b { color: var(--ink); }
.suggestion .small-muted { color: var(--muted); font-size: 13px; margin: 6px 0 0; }

.app-footer { font-size: 11px; color: #98A2B3; text-align: center; padding-top: 15px; }

hr { border-color: #E8ECF3; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------- ATTACHMENTS ----------------

class AttachmentError(Exception):
    """Raised when an uploaded file cannot be read."""


def _extension(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _extract_text(ext: str, data: bytes) -> str:
    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise AttachmentError("PDF support needs a package: pip install pypdf") from error
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    if ext == "docx":
        try:
            from docx import Document
        except ImportError as error:
            raise AttachmentError("Word support needs a package: pip install python-docx") from error
        document = Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    return data.decode("utf-8", errors="replace")


def process_uploads(files: list) -> tuple[list[dict], list[str]]:
    """Turn uploaded files into storable attachments. Returns (attachments, warnings)."""
    attachments: list[dict] = []
    warnings: list[str] = []

    if len(files) > MAX_FILES_PER_MESSAGE:
        warnings.append(f"Only the first {MAX_FILES_PER_MESSAGE} files were used.")

    for uploaded in files[:MAX_FILES_PER_MESSAGE]:
        name = uploaded.name
        ext = _extension(name)
        data = uploaded.getvalue()

        if len(data) > MAX_FILE_BYTES:
            warnings.append(f"{name} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB and was skipped.")
            continue

        try:
            if ext in IMAGE_EXTENSIONS:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                filename = f"{uuid.uuid4().hex}.{ext}"
                (UPLOAD_DIR / filename).write_bytes(data)
                mime = uploaded.type or mimetypes.guess_type(name)[0] or f"image/{ext}"
                attachments.append({"name": name, "kind": "image", "file": filename, "mime": mime})

            elif ext in TEXT_EXTENSIONS | DOCUMENT_EXTENSIONS:
                text = _extract_text(ext, data).strip()
                if not text:
                    warnings.append(f"No readable text found in {name} (scanned PDFs are not supported).")
                    continue
                truncated = len(text) > MAX_ATTACHMENT_CHARS
                if truncated:
                    text = text[:MAX_ATTACHMENT_CHARS]
                    warnings.append(f"{name} is very long, so only the first part was used.")
                attachments.append({"name": name, "kind": "text", "text": text, "truncated": truncated})

            else:
                warnings.append(f"{name}: this file type is not supported.")

        except AttachmentError as error:
            warnings.append(str(error))
        except OSError:
            warnings.append(f"{name} could not be saved.")
        except Exception:  # noqa: BLE001 - a broken file must never crash the app
            warnings.append(f"{name} could not be read.")

    return attachments, warnings


def remove_attachment_files(message: dict) -> None:
    for attachment in message.get("attachments", []):
        if attachment.get("kind") == "image" and attachment.get("file"):
            try:
                (UPLOAD_DIR / attachment["file"]).unlink(missing_ok=True)
            except OSError:
                pass


def image_data_url(attachment: dict) -> str | None:
    path = UPLOAD_DIR / attachment.get("file", "")
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    mime = attachment.get("mime") or mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{encoded}"


# ---------------- HISTORY ----------------

def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def history_group(updated_at: str | None) -> str:
    """Label used to group the sidebar list: Today / Yesterday / ..."""
    if not updated_at:
        return "Earlier"
    try:
        moment = datetime.fromisoformat(updated_at).astimezone()
    except ValueError:
        return "Earlier"
    days = (datetime.now().astimezone().date() - moment.date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days <= 7:
        return "Previous 7 days"
    return "Earlier"


def _is_valid_chat(chat: Any) -> bool:
    return (
        isinstance(chat, dict)
        and isinstance(chat.get("id"), str)
        and isinstance(chat.get("messages", []), list)
    )


def _normalise_attachments(raw: Any) -> list[dict]:
    clean: list[dict] = []
    if not isinstance(raw, list):
        return clean
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        if item.get("kind") == "text" and isinstance(item.get("text"), str):
            clean.append({
                "name": item["name"], "kind": "text",
                "text": item["text"], "truncated": bool(item.get("truncated")),
            })
        elif item.get("kind") == "image" and isinstance(item.get("file"), str):
            clean.append({
                "name": item["name"], "kind": "image",
                "file": item["file"], "mime": str(item.get("mime") or ""),
            })
    return clean


def _normalise_chat(chat: dict) -> dict:
    """Make sure every stored chat has all required fields and valid messages."""
    messages = []
    for m in chat.get("messages", []):
        if not (isinstance(m, dict) and m.get("role") in ("user", "assistant") and "content" in m):
            continue
        message = {"role": m["role"], "content": str(m["content"])}
        attachments = _normalise_attachments(m.get("attachments"))
        if attachments:
            message["attachments"] = attachments
        messages.append(message)

    updated_at = chat.get("updated_at")
    return {
        "id": chat["id"],
        "title": str(chat.get("title") or "New chat"),
        "messages": messages,
        "updated_at": updated_at if isinstance(updated_at, str) else None,
    }


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    chats = [_normalise_chat(chat) for chat in data if _is_valid_chat(chat)]
    return [chat for chat in chats if chat["messages"]]  # drop empty "New chat" leftovers


def save_history() -> None:
    """Persist non-empty conversations only. API keys are never written to disk."""
    temp_file = HISTORY_FILE.with_suffix(".tmp")
    to_save = [chat for chat in st.session_state.chats if chat["messages"]]
    try:
        with temp_file.open("w", encoding="utf-8") as file:
            json.dump(to_save, file, ensure_ascii=False, indent=2)
        temp_file.replace(HISTORY_FILE)
    except OSError:
        st.toast("Could not save chat history to disk.", icon="⚠️")


def create_chat() -> None:
    """Open a blank chat. If a blank one already exists, reuse it (no duplicates)."""
    for chat in st.session_state.chats:
        if not chat["messages"]:
            st.session_state.active_chat_id = chat["id"]
            return
    chat = {"id": str(uuid.uuid4()), "title": "New chat", "messages": [], "updated_at": now_iso()}
    st.session_state.chats.insert(0, chat)
    st.session_state.active_chat_id = chat["id"]


def select_chat(chat_id: str) -> None:
    # Discard blank drafts when moving to another conversation.
    st.session_state.chats = [
        c for c in st.session_state.chats if c["messages"] or c["id"] == chat_id
    ]
    st.session_state.active_chat_id = chat_id


def get_active_chat() -> dict | None:
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    return None


def delete_chat(chat_id: str) -> None:
    for chat in st.session_state.chats:
        if chat["id"] == chat_id:
            for message in chat["messages"]:
                remove_attachment_files(message)
    st.session_state.chats = [c for c in st.session_state.chats if c["id"] != chat_id]
    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = None
    save_history()


def rename_chat(chat_id: str, new_title: str) -> None:
    title = re.sub(r"\s+", " ", new_title or "").strip()[:MAX_RENAME_LENGTH]
    if not title:
        return
    for chat in st.session_state.chats:
        if chat["id"] == chat_id:
            chat["title"] = title
            save_history()
            return


def _commit_rename(chat_id: str) -> None:
    """Callback for the rename box (Enter key or Save button)."""
    rename_chat(chat_id, st.session_state.get(f"rename_{chat_id}", ""))
    st.toast("Chat renamed", icon="✅")


def make_title(prompt: str) -> str:
    text = re.sub(r"\s+", " ", prompt).strip()
    return text if len(text) <= MAX_TITLE_LENGTH else text[:MAX_TITLE_LENGTH].rstrip() + "..."


# ---------------- LLM HELPERS ----------------

def build_model(api_key: str, streaming: bool = False) -> ChatOpenAI:
    # No `temperature` is passed: GPT-5 family models only accept the default value
    # and reject custom ones with a 400 error.
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        streaming=streaming,
        timeout=60,
        max_retries=2,
    )


def build_messages(chat_messages: list[dict]) -> list[BaseMessage]:
    context = chat_messages[-MAX_CONTEXT_MESSAGES:]

    # Only the newest few images are sent again; older ones are mentioned by name.
    keep_images: set[tuple[int, int]] = set()
    for i in range(len(context) - 1, -1, -1):
        for j, attachment in enumerate(context[i].get("attachments", [])):
            if attachment["kind"] == "image" and len(keep_images) < MAX_IMAGES_IN_CONTEXT:
                keep_images.add((i, j))

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_ROLE)]
    for i, item in enumerate(context):
        if item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
            continue

        text_parts = [item["content"]]
        image_parts: list[dict] = []
        for j, attachment in enumerate(item.get("attachments", [])):
            name = attachment["name"]
            if attachment["kind"] == "text":
                note = " (truncated)" if attachment.get("truncated") else ""
                text_parts.append(
                    f"\n\n--- Attached file: {name}{note} ---\n"
                    f"{attachment['text']}\n--- End of file ---"
                )
            elif (i, j) in keep_images:
                data_url = image_data_url(attachment)
                if data_url:
                    image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                else:
                    text_parts.append(f'\n[Image "{name}" is no longer available]')
            else:
                text_parts.append(f'\n[Image "{name}" was attached earlier and is not shown again]')

        text = "".join(text_parts)
        content: str | list[dict] = (
            [{"type": "text", "text": text}, *image_parts] if image_parts else text
        )
        messages.append(HumanMessage(content=content))
    return messages


def stream_answer(model: ChatOpenAI, messages: list[BaseMessage]) -> Iterator[str]:
    for chunk in model.stream(messages):
        content = chunk.content
        if isinstance(content, str):
            if content:
                yield content
        elif isinstance(content, list):  # content blocks
            for block in content:
                if isinstance(block, str):
                    yield block
                elif isinstance(block, dict) and isinstance(block.get("text"), str):
                    yield block["text"]


def verify_api_key(key: str) -> None:
    """Raises an OpenAI error if the key or model access is invalid."""
    build_model(key).invoke([HumanMessage(content="Reply with OK.")])


# ---------------- SESSION STATE ----------------

DEFAULTS = {
    "verified": False,
    "api_key": "",
    "chats": None,
    "active_chat_id": None,
}
for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value

if st.session_state.chats is None:
    st.session_state.chats = load_history()


# ==================================================
# PAGE 1: API KEY VERIFICATION
# ==================================================

def render_login_page() -> None:
    st.markdown(
        '<div class="brand"><span class="dot">●</span> '
        'Your Personal <span class="blue">AI Chatbot</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.write("")

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.write("")
        st.write("")
        st.markdown(
            '<p class="eyebrow">YOUR PERSONAL AI WORKSPACE</p>'
            '<div class="hero-accent"></div>'
            '<div class="hero-title">A smarter way<br>'
            '<span class="blue">to get things done.</span></div>'
            '<p class="hero-description">Connect your OpenAI API key to start '
            "conversations, explore ideas and get help with your daily work.</p>"
            "<p>🔒 &nbsp; Secure API key entry</p>"
            "<p>⚡ &nbsp; Live API verification</p>"
            "<p>💬 &nbsp; Saved chat history</p>",
            unsafe_allow_html=True,
        )

    with right:
        st.write("")
        st.write("")
        st.markdown(
            '<div class="card"><p class="eyebrow">GET STARTED</p>'
            "<h2>Connect your API key</h2>"
            '<p class="muted">Verify your API access to continue.</p></div>',
            unsafe_allow_html=True,
        )

        with st.form("api_verification_form"):
            key_input = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                autocomplete="new-password",  # stops browser from showing old suggestions
            )
            submitted = st.form_submit_button("Verify & Continue", width="stretch")

        if not submitted:
            return

        key = key_input.strip()
        if not key:
            st.warning("Please enter your API key.")
            return

        try:
            with st.spinner("Verifying your API key..."):
                verify_api_key(key)
        except AuthenticationError:
            st.error("Invalid API key. Please check your key and try again.")
            return
        except OpenAIError as error:
            st.error("API request failed. Check model access, billing and usage limits.")
            st.caption(str(error)[:250])
            return
        except Exception as error:  # noqa: BLE001 - last-resort UI guard
            st.error("Verification could not be completed.")
            st.caption(str(error)[:250])
            return

        st.session_state.api_key = key
        st.session_state.verified = True
        if get_active_chat() is None:
            create_chat()
        st.rerun()


# ==================================================
# PAGE 2: CHAT INTERFACE
# ==================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="dot">●</span><br>'
            'Your Personal<br><span class="blue">AI Chatbot</span></div>',
            unsafe_allow_html=True,
        )

        if st.button("New chat", icon=":material/add:", type="primary", width="stretch"):
            create_chat()
            st.rerun()

        st.markdown('<div class="section-label">CHAT HISTORY</div>', unsafe_allow_html=True)

        # Only real conversations are listed; a blank chat is not shown as an entry.
        history = sorted(
            (c for c in st.session_state.chats if c["messages"]),
            key=lambda c: c.get("updated_at") or "",
            reverse=True,
        )

        if not history:
            st.caption("Your conversations will appear here.")

        current_group = None
        for chat in history:
            group = history_group(chat.get("updated_at"))
            if group != current_group:
                current_group = group
                st.markdown(f'<div class="group-label">{group}</div>', unsafe_allow_html=True)

            chat_id = chat["id"]
            is_active = chat_id == st.session_state.active_chat_id
            col_chat, col_menu = st.columns([6, 1.3], gap="small", vertical_alignment="center")

            with col_chat:
                # The key prefix ("active_" / "select_") is what the CSS targets.
                if st.button(
                    chat["title"],
                    key=f"{'active' if is_active else 'select'}_{chat_id}",
                    icon=":material/chat_bubble_outline:",
                    width="stretch",
                ):
                    select_chat(chat_id)
                    st.rerun()

            with col_menu:
                with st.popover("", icon=":material/more_horiz:", help="Rename or delete"):
                    st.text_input(
                        "Rename chat",
                        value=chat["title"],
                        max_chars=MAX_RENAME_LENGTH,
                        key=f"rename_{chat_id}",
                        on_change=_commit_rename,
                        args=(chat_id,),
                    )
                    st.button(
                        "Save name",
                        key=f"save_{chat_id}",
                        icon=":material/check:",
                        type="primary",
                        width="stretch",
                        on_click=_commit_rename,
                        args=(chat_id,),
                    )
                    if st.button(
                        "Delete chat",
                        key=f"del_{chat_id}",
                        icon=":material/delete_outline:",
                        width="stretch",
                    ):
                        delete_chat(chat_id)
                        st.rerun()

        # Footer block: pushed to the very bottom of the sidebar by CSS.
        with st.container(key="sidebar_footer"):
            st.divider()
            st.caption(f"Model: {MODEL_NAME}")

            if st.button("Log out", key="logout", icon=":material/logout:", width="stretch"):
                st.session_state.api_key = ""
                st.session_state.verified = False
                st.rerun()


def render_welcome(vanishing: bool = False) -> None:
    """Welcome block. When `vanishing` is True it fades out smoothly (CSS animation)."""
    cards = [
        ("Learn something", "Understand concepts easily."),
        ("Write & create", "Develop ideas and content."),
        ("Code & solve", "Understand and debug code."),
    ]
    cards_html = "".join(
        f'<div class="suggestion"><b>{title}</b><p class="small-muted">{text}</p></div>'
        for title, text in cards
    )
    st.markdown(
        f'<div class="welcome-wrap{" vanish" if vanishing else ""}">'
        '<div class="welcome-title">How can I help you today?</div>'
        '<div class="welcome-description">'
        "Ask a question, learn something new, or solve a problem.</div>"
        f'<div class="welcome-grid">{cards_html}</div></div>',
        unsafe_allow_html=True,
    )


_CODE_FENCE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_MATH_BLOCK = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


def format_math(text: str) -> str:
    """Convert LaTeX written as \\( ... \\) and \\[ ... \\] into $...$ / $$...$$,
    which is what Streamlit renders as proper formulas. Code is left untouched."""
    parts = _CODE_FENCE.split(text)
    for i in range(0, len(parts), 2):  # even items are normal text, odd items are code
        chunk = _MATH_BLOCK.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", parts[i])
        parts[i] = _MATH_INLINE.sub(lambda m: f"${m.group(1).strip()}$", chunk)
    return "".join(parts)


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(format_math(message["content"]))

        chips: list[str] = []
        for attachment in message.get("attachments", []):
            name = attachment["name"]
            if attachment["kind"] == "image":
                path = UPLOAD_DIR / attachment.get("file", "")
                if path.is_file():
                    st.image(str(path), width=260, caption=name)
                    continue
            chips.append(f'<span class="file-chip">📎 {html.escape(name)}</span>')
        if chips:
            st.markdown("".join(chips), unsafe_allow_html=True)


def get_user_input() -> tuple[str, list] | None:
    """Reads the chat box. Returns (text, files) or None when nothing was sent."""
    placeholder = "Message your personal AI chatbot..."
    try:
        value = st.chat_input(placeholder, accept_file="multiple", file_type=ACCEPTED_FILE_TYPES)
    except TypeError:  # older Streamlit without file upload in the chat box
        value = st.chat_input(placeholder)

    if value is None:
        return None
    if isinstance(value, str):
        text, files = value.strip(), []
    else:
        text = (getattr(value, "text", None) or value.get("text", "") or "").strip()
        files = list(getattr(value, "files", None) or value.get("files", []) or [])

    if not text and not files:
        return None
    return text, files


def handle_prompt(chat: dict, text: str, files: list) -> None:
    attachments, warnings = process_uploads(files) if files else ([], [])
    for warning in warnings:
        st.warning(warning)
    if not text and not attachments:
        return

    message: dict = {"role": "user", "content": text or DEFAULT_FILE_PROMPT}
    if attachments:
        message["attachments"] = attachments
    chat["messages"].append(message)
    chat["updated_at"] = now_iso()
    if chat["title"] == "New chat":
        chat["title"] = make_title(text or attachments[0]["name"])
    save_history()

    render_message(message)

    with st.chat_message("assistant"):
        try:
            model = build_model(st.session_state.api_key, streaming=True)
            answer = st.write_stream(
                stream_answer(model, build_messages(chat["messages"]))
            )
        except AuthenticationError:
            _fail(chat, "API authentication failed. Please log out and use a valid API key.")
            return
        except OpenAIError as error:
            _fail(
                chat,
                "The API request failed. Check your API access, billing, "
                "model availability and connection.",
                detail=str(error),
            )
            return
        except Exception as error:  # noqa: BLE001 - last-resort UI guard
            _fail(chat, "Something went wrong. Please try again.", detail=str(error))
            return

    if not isinstance(answer, str):
        answer = "".join(str(part) for part in answer)

    chat["messages"].append({"role": "assistant", "content": answer})
    chat["updated_at"] = now_iso()
    save_history()
    # Rerun so the sidebar shows the new chat entry immediately.
    st.rerun()


def _fail(chat: dict, message: str, detail: str = "") -> None:
    """Show an error and drop the unanswered user message so history stays consistent."""
    st.error(message)
    if detail:
        st.caption(detail[:250])
    if chat["messages"] and chat["messages"][-1]["role"] == "user":
        remove_attachment_files(chat["messages"].pop())
        if not chat["messages"]:
            chat["title"] = "New chat"
        save_history()


def render_chat_page() -> None:
    chat = get_active_chat()
    if chat is None:
        create_chat()
        chat = get_active_chat()

    render_sidebar()

    st.markdown(
        '<div class="topline"><div class="title">'
        '<span class="dot">●</span> Your Personal <span>AI Chatbot</span></div></div>',
        unsafe_allow_html=True,
    )

    if chat is None:  # defensive; create_chat() always provides one
        return

    # The chat box is read first (it stays pinned to the bottom of the page) so the
    # welcome block already knows whether a message was just sent and can fade away.
    user_input = get_user_input()

    if not chat["messages"]:
        render_welcome(vanishing=user_input is not None)

    for message in chat["messages"]:
        render_message(message)

    if user_input is not None:
        handle_prompt(chat, *user_input)


# ---------------- ENTRY POINT ----------------

if st.session_state.verified:
    render_chat_page()
else:
    render_login_page()
    st.markdown(f'<div class="app-footer">{APP_TITLE.upper()}</div>', unsafe_allow_html=True)