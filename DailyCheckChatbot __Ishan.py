"""
Assignment 02 — Daily Check-in Chatbot

This Streamlit app follows the assignment flow as directly as possible:
1) Collecting depression level from 1 to 10.
2) Giving deterministic range-based feedback in Python.
3) Asking for an OPTIONAL short note explaining the level.
4) Using an LLM for generating supportive response to the note and then supporting multi-turn chat.
5) Allowing export and clearing of chat history.

Important implementation choices safety:
- Step 2 is handled with fixed Python logic instead of the LLM so the required feedback is stable.
- The level parser is intentionally regex-based and strict so mixed strings like "null 7" are rejected.
- The LLM is used only for the note response and later free chat with a bounded support-oriented prompt.
- Conversation state is stored in st.session_state.
- The app never hardcodes an OpenAI API key.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import streamlit as st
from openai import OpenAI

MODEL = "gpt-4o-mini"
# A lightweight model is sufficient here because the app keeps the most critical decision logic in deterministic Python rather than relying on the LLM.
# -----------------------------
# UI text constants
# -----------------------------
# Keeping user-facing strings in one place makes the UI easier to maintain and helps preserve the same wording across resets and future edits.
APP_TITLE = "🧠 Daily Depression Check-in by Ishan"
INITIAL_LEVEL_PROMPT = "Please enter your depression level from 1 to 10 (e.g., 6, log 6, 7/10)."
INVALID_LEVEL_PROMPT = "Please enter a valid number from 1 to 10, for example 6, level 7, or 7/10."
OPTIONAL_NOTE_PROMPT = "Please write a short note explaining your level (optional, type skip if you want to move on)."
SKIP_NOTE_RESPONSE = "No problem. You can still chat with me about how you are feeling today."
LLM_ERROR_MESSAGE = "I had trouble generating a response just now. Please try sending that again."
MAX_HISTORY_MESSAGES = 8

st.set_page_config(page_title="Daily Check-in Chatbot")
st.title(APP_TITLE)
# Only short history window is kept so later chat stays coherent without sending an unnecessarily long transcript back to the model.
# -----------------------------
# Safety / behavior constraints
# -----------------------------
# The coping list is fixed on purpose. Limiting suggestions helps reduce model improvisation and keeps Step 4 closer to the focus on supportive,lower-risk responses.
SAFE_SUGGESTIONS = [
    "Take three slow breaths and loosen your shoulders.",
    "Do a quick grounding check: name five things you see, four you feel, three you hear.",
    "Choose one tiny task you can finish in ten minutes.",
    "Take a short walk or stretch for a few minutes.",
    "Drink water and take a short pause away from the screen.",
    "Write one sentence about what feels hardest right now.",
    "Reach out to one trusted person for a brief check-in.",
]

# Crisis-language detection is intentionally simple and conservative. If any of these phrases appear, the app returns a fixed safety-oriented message instead of sending the text to the model.
CRISIS_PATTERNS = [
    r"\b(suicide|suicidal)\b",
    r"\b(kill myself|end my life|take my life)\b",
    r"\b(self[- ]?harm|cut myself)\b",
    r"\b(i don'?t want to live|don'?t want to be here)\b",
    r"\b(overdose)\b",
]

CRISIS_MESSAGE = (
    "I’m really glad you said that. I can’t provide crisis help, but I can help you get support right now.\n\n"
    "If you feel unsafe or at risk of harming yourself, please contact local emergency services immediately. "
    "If you are in the U.S. or Canada, you can call or text 988. If you are elsewhere, contact your local crisis line or go to the nearest emergency department.\n\n"
    "If you are not in immediate danger, please reach out to a trusted person and let them know you need support today."
)

# This prompt is used only for Step 4 (note response) and later multi-turn chat.
# The range-based feedback in Step 2 remains deterministic Python logic.
SYSTEM_PROMPT = f"""
You are a supportive, non-judgmental daily check-in coach for a classroom prototype.
Your job is to respond supportively to the user's note and follow-up messages.

Rules:
- Do NOT diagnose, label, or claim to be a therapist.
- Do NOT give medical advice, medication advice, or treatment claims.
- Do NOT fabricate facts.
- Keep responses clear, warm, and concise.
- Reflect what the user said in one sentence.
- If you suggest coping ideas, choose at most 2 or 3 from this list and do not invent new ones:
{json.dumps(SAFE_SUGGESTIONS, ensure_ascii=False)}
- End with one gentle follow-up question.
"""

# -----------------------------
# Regex patterns for level input
# -----------------------------
# Th design choice prevents false positives such as:
#   - "null 7"
#   - "today is 7"
#   - "7 maybe 8"
# while still accepting the required formats of assignment.
LEVEL_PATTERNS = (
    re.compile(r"^\s*(10|[1-9])\s*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*log\s*(10|[1-9])\s*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:level|lvl)\s*(?:is|=|:)?\s*(10|[1-9])\s*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(10|[1-9])\s*/\s*10\s*[.!?]?\s*$", re.IGNORECASE),
    re.compile(r"^\s*(10|[1-9])\s+out\s+of\s+10\s*[.!?]?\s*$", re.IGNORECASE),
)


# -----------------------------
# Helpers
# -----------------------------
def now_iso() -> str:
    """Return the current UTC timestamp for exported chat history."""
    return datetime.now(timezone.utc).isoformat()


# This check is called before any model request so risky language gets a fixed,safety-oriented answer instead of an improvised model response.
def contains_crisis_language(text: str) -> bool:
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in CRISIS_PATTERNS)


# The parser intentionally accepts only the supported input shapes listed in the
# assignment examples. Using regex.fullmatch-style logic keeps the behavior easy
# to explain and makes the rejected/accepted boundary very clear.
def parse_level(text: str) -> Optional[int]:
    """Parse a 1–10 check-in level from the allowed input formats.

    Accepted examples:
      - "6"
      - "log 6"
      - "level 7" / "lvl7"
      - "7/10"
      - "7 out of 10"

    Rejected examples:
      - "null 7"
      - "-5"
      - "15"
      - "7 or 8"
    """
    if not text:
        return None

    for pattern in LEVEL_PATTERNS:
        match = pattern.match(text)
        if match:
            return int(match.group(1))
    return None


# Step 2 is deliberately deterministic. This would avoid hallucinations and ensures
# the required range-based logic is always applied the same way.
def range_feedback(level: int) -> str:
    """Return the fixed supportive message for the logged level."""
    if 1 <= level <= 3:
        return (
            f"You’re doing a good job checking in. **{level}/10** is relatively low today. "
            "Try a tiny maintenance step: take three slow breaths and pick one small thing you want to keep going."
        )
    if 4 <= level <= 6:
        return (
            f"Thanks for logging **{level}/10**. A moderate level can feel heavy or distracting. "
            "Try a quick reset: name five things you see, four you feel, three you hear, two you smell, and one you taste."
        )
    if 7 <= level <= 8:
        return (
            f"That sounds tough. **{level}/10**  is in the high range today. "
            "Reducing the load can help. Try two minutes of slow breathing, then choose one tiny task you can finish in ten minutes."
        )
    if 9 <= level <= 10:
        return (
            f"I’m sorry you’re feeling this intensely. **{level}/10** sounds very high right now. "
            "Focus on immediate calming, such as slow breathing or moving to a quieter place. "
            "If you feel unsafe or at risk of harming yourself, please contact local emergency services or a crisis hotline in your area."
        )
    raise ValueError("level must be between 1 and 10")


# Exporting to JSON satisfies the "export chat history" requirement
# while preserving role labels, timestamps, the logged level and the note.
def export_payload() -> str:
    payload = {
        "exported_at": now_iso(),
        "level": st.session_state.get("level"),
        "note": st.session_state.get("note"),
        "messages": st.session_state.get("messages", []),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# The model sees the recorded level and note as explicit system context.
# For later free chat, we also pass a short slice of recent conversation history
# so the bot could stay coherent without needing the entire transcript every time.
def build_messages(user_text: str, include_history: bool = True) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    level = st.session_state.get("level")
    note = st.session_state.get("note")

    if level is not None:
        messages.append({"role": "system", "content": f"Today's logged level: {level}/10."})
    if note:
        messages.append({"role": "system", "content": f"User note: {note}"})

    if include_history:
        for message in st.session_state.get("messages", [])[-MAX_HISTORY_MESSAGES:]:
            if message.get("role") in ("user", "assistant"):
                messages.append({"role": message["role"], "content": message["content"]})

    messages.append({"role": "user", "content": user_text})
    return messages


# Streaming keeps the interaction feeling like a chat app. The function yields
# chunks so Streamlit could display the response progressively.
def stream_llm(client: OpenAI, messages: List[Dict[str, str]]) -> Iterable[str]:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        # Low temperature and a modest token cap help keep answers short, steady and less likely to drift away from the bounded support role.
        max_tokens=220,
        stream=True,
    )
    for event in stream:
        try:
            delta = event.choices[0].delta
            if delta and delta.content:
                yield delta.content
        except Exception:
            # If one streaming event is malformed, skip it and keep going.
            continue


# Wrapping the LLM call in helper lets the app recover gracefully from API or
# network issues instead of crashing in the middle of the conversation.
def write_llm_reply(client: OpenAI, user_text: str, include_history: bool) -> str:
    try:
        with st.chat_message("assistant"):
            response_text = st.write_stream(
                stream_llm(client, build_messages(user_text, include_history=include_history))
            )
        if isinstance(response_text, str) and response_text.strip():
            return response_text
        return LLM_ERROR_MESSAGE
    except Exception:
        with st.chat_message("assistant"):
            st.write(LLM_ERROR_MESSAGE)
        return LLM_ERROR_MESSAGE


# Reset clears only the conversation specific state. The API key remains so the
# user would be able to start a fresh check-in without re-entering credentials.
def reset_conversation() -> None:
    st.session_state.stage = "level"
    st.session_state.level = None
    st.session_state.note = None
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": INITIAL_LEVEL_PROMPT,
        }
    ]


# Setting default values once keeps the rest of the script simple. Every later
# section could assume these keys already exist in st.session_state.
def initialize_session_state() -> None:
    if "stage" not in st.session_state:
        st.session_state.stage = "level"
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": INITIAL_LEVEL_PROMPT}]
    if "level" not in st.session_state:
        st.session_state.level = None
    if "note" not in st.session_state:
        st.session_state.note = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "api_key_confirmed" not in st.session_state:
        st.session_state.api_key_confirmed = False


# Rendering chat history before handling new input makes the app feel like persistent conversation rather than a single turn form.
def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


# -----------------------------
# Session state setup
# -----------------------------
initialize_session_state()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.markdown("## :material/settings: Settings")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", value=st.session_state.api_key)

if st.sidebar.button(":material/key: Change API Key", use_container_width=True):
    st.session_state.api_key = api_key_input.strip()
    st.session_state.api_key_confirmed = bool(st.session_state.api_key)
# The key is saved only after the user clicks the confirmation button, which makes the runtime secret handling explicit and easy to inspect.
st.sidebar.markdown("## Tools")
st.sidebar.download_button(
    label=":material/download: Export chat history",
    data=export_payload(),
    file_name="daily_checkin_history.json",
    mime="application/json",
    use_container_width=True,
)

if st.sidebar.button(":material/delete: Clear chat history", use_container_width=True):
    reset_conversation()
    st.rerun()

# If the user has not confirmed a key yet, we stop before creating the client.
# This prevents accidental calls with an empty API key.
if not st.session_state.api_key_confirmed:
    st.stop()
# Clearing chat resets the staged conversation but it intentionally does not erase the confirmed API key so the user could restart quickly.
client = OpenAI(api_key=st.session_state.api_key)

# Show the full conversation so far before reading the next user input.
render_history()
# Stop here until a non-empty key is confirmed, so the OpenAI client below is never created with empty credentials by mistake.
# -----------------------------
# Main flow
# -----------------------------
# The app uses simple three stage state machine:
#   level to note to chat
# This makes the assignment flow explicit and easy to debug.
if st.session_state.stage == "level":
    user_input = st.chat_input("Enter level...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        level = parse_level(user_input)

        if level is None:
            st.session_state.messages.append({"role": "assistant", "content": INVALID_LEVEL_PROMPT})
            st.rerun()

        st.session_state.level = level
        st.session_state.messages.append({"role": "assistant", "content": range_feedback(level)})
        st.session_state.messages.append({"role": "assistant", "content": OPTIONAL_NOTE_PROMPT})
        st.session_state.stage = "note"
        st.rerun()

elif st.session_state.stage == "note":
    note_input = st.chat_input("Write your note...")
    if note_input:
        st.session_state.messages.append({"role": "user", "content": note_input})
# "skip" preserves the optional-note path and moves the user straight into free chat without forcing extra input.
        if note_input.strip().lower() == "skip":
            st.session_state.note = ""
            st.session_state.messages.append({"role": "assistant", "content": SKIP_NOTE_RESPONSE})
            st.session_state.stage = "chat"
            st.rerun()

        st.session_state.note = note_input.strip()

        if contains_crisis_language(note_input):
            st.session_state.messages.append({"role": "assistant", "content": CRISIS_MESSAGE})
            st.session_state.stage = "chat"
            st.rerun()

        note_prompt = (
            f"The user logged a depression level of {st.session_state.level}/10.\n"
            f"Their note is: {st.session_state.note}\n\n"
            "Respond supportively. First reflect what they said in one sentence. "
            "Then offer 2 or 3 coping ideas chosen only from the allowed list. "
            "Then ask one gentle follow-up question."
        )
#The first LLM note response uses structured instructions plus the stored level and note but skips prior chat history for keeping it clean.
        response = write_llm_reply(client, note_prompt, include_history=False)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.stage = "chat"
        st.rerun()

elif st.session_state.stage == "chat":
    user_input = st.chat_input("Chat...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
#Repeat the crisis check in free chat because high risk language might appear after the initial note rather than during the note step.
        if contains_crisis_language(user_input):
            st.session_state.messages.append({"role": "assistant", "content": CRISIS_MESSAGE})
            st.rerun()

        response = write_llm_reply(client, user_input, include_history=True)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
