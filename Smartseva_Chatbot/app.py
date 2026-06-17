import streamlit as st
import json
import os
import io
import requests
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════
# VOICE IMPORTS
#
# SPEECH-TO-TEXT (cloud-compatible):
#   - streamlit_mic_recorder: captures audio in the USER'S BROWSER
#     using JavaScript (works on Streamlit Cloud — no server mic needed)
#   - speech_recognition: converts the recorded audio bytes to text
#     using Google's free Speech Recognition API (no API key needed)
#
# TEXT-TO-SPEECH (cloud-compatible):
#   - gTTS: generates MP3 via Google's TTS API, played with st.audio()
#     (works on Windows, phone browsers, and Streamlit Cloud — no OS
#     speech engine dependency like pyttsx3 had)
# ══════════════════════════════════════════════════════════════
try:
    from streamlit_mic_recorder import mic_recorder
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# CONFIG — OPENROUTER
# ══════════════════════════════════════════════════════════════
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ OPENROUTER_API_KEY not found. Add it to your .env file (local) "
              "or to Streamlit Cloud Secrets (deployed).")
    st.stop()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free model on OpenRouter — no billing required.
# You can swap this for another free model from openrouter.ai/models if needed.
OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"

# ══════════════════════════════════════════════════════════════
# LOAD JSON DATA
# ══════════════════════════════════════════════════════════════
def load_json(path, label):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"⚠️ '{path}' not found — {label} will be skipped.")
        return {}
    except json.JSONDecodeError:
        st.warning(f"⚠️ '{path}' has invalid JSON — {label} will be skipped.")
        return {}

services_data     = load_json("data/services.json",    "Services")
farmers_data      = load_json("data/farmers.json",     "Farmer Schemes")
scholarships_data = load_json("data/scholarship.json", "Scholarships")

# ══════════════════════════════════════════════════════════════
# SCORE-BASED MATCHING (prevents "card" matching multiple items)
# ══════════════════════════════════════════════════════════════
STOP_WORDS = {
    "for", "the", "and", "are", "what", "how", "do", "i", "to",
    "apply", "need", "get", "a", "an", "of", "in", "is", "tell",
    "me", "about", "can", "my", "give", "information", "details",
    "required", "documents", "eligibility", "fee", "steps", "process",
    "time", "benefits", "category", "please", "list"
}

def score_match(item_name: str, query: str) -> int:
    name_words = [
        w for w in item_name.lower().split()
        if len(w) > 2 and w not in STOP_WORDS
    ]
    if not name_words:
        return 0
    query_lower = query.lower()
    return sum(1 for w in name_words if w in query_lower)

def find_single_best_match(query: str, items: list, name_key: str = "name"):
    scores = []
    for item in items:
        name  = item.get(name_key, "")
        score = score_match(name, query)
        if score > 0:
            scores.append((score, item))

    if not scores:
        return None

    scores.sort(key=lambda x: x[0], reverse=True)
    top_score   = scores[0][0]
    top_matches = [item for s, item in scores if s == top_score]

    if len(top_matches) == 1:
        return top_matches[0]
    return None   # ambiguous tie → let AI handle it

# ══════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════
def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["document","documents","proof","papers","కావాల్సిన","what documents","needed"]):
        return "documents"
    if any(w in q for w in ["fee","cost","charge","charges","money","price","pay","రుసుము"]):
        return "fee"
    if any(w in q for w in ["step","steps","how to apply","process","procedure","application","దరఖాస్తు"]):
        return "steps"
    if any(w in q for w in ["time","processing","how long","days","duration","weeks","రోజులు"]):
        return "time"
    if any(w in q for w in ["eligible","eligibility","qualify","who can","criteria","అర్హత"]):
        return "eligibility"
    if any(w in q for w in ["benefit","benefits","amount","receive","grant","లబ్ధి"]):
        return "benefits"
    if "category" in q:
        return "category"
    return "all"

# ══════════════════════════════════════════════════════════════
# DISPLAY — SERVICE
# ══════════════════════════════════════════════════════════════
def show_service(service: dict, intent: str) -> str:
    name     = service.get("name", "")
    docs     = service.get("required_documents", [])
    steps    = service.get("steps_to_apply", [])
    fee      = service.get("fee", "Not specified")
    time_val = service.get("processing_time", "Not specified")

    st.subheader(f"🏛️ {name}")
    spoken = f"{name}. "

    if intent == "documents":
        st.markdown("##### 📄 Required Documents")
        for d in docs: st.write(f"• {d}")
        spoken += "Required documents are: " + ", ".join(docs) + "."

    elif intent == "fee":
        st.markdown("##### 💰 Fee")
        st.write(fee)
        spoken += f"The fee is {fee}."

    elif intent == "steps":
        st.markdown("##### 📋 Steps to Apply")
        for i, s in enumerate(steps, 1): st.write(f"{i}. {s}")
        spoken += "Steps to apply: " + ". ".join(steps) + "."

    elif intent == "time":
        st.markdown("##### ⏱️ Processing Time")
        st.write(time_val)
        spoken += f"Processing time is {time_val}."

    else:
        st.markdown("##### 📄 Required Documents")
        for d in docs: st.write(f"• {d}")
        st.markdown("##### 📋 Steps to Apply")
        for i, s in enumerate(steps, 1): st.write(f"{i}. {s}")
        st.markdown("##### 💰 Fee")
        st.write(fee)
        st.markdown("##### ⏱️ Processing Time")
        st.write(time_val)
        spoken += (
            f"Required documents: {', '.join(docs)}. "
            f"Steps: {'. '.join(steps)}. "
            f"Fee: {fee}. Processing time: {time_val}."
        )

    return spoken

# ══════════════════════════════════════════════════════════════
# DISPLAY — SCHEME / SCHOLARSHIP
# ══════════════════════════════════════════════════════════════
def show_scheme(item: dict, intent: str) -> str:
    name = item.get("name", "")
    cat  = item.get("category", "Not specified")
    elig = item.get("eligibility", "Not specified")
    docs = item.get("documents_required", [])
    ben  = item.get("benefits", "Not specified")

    st.subheader(f"📜 {name}")
    spoken = f"{name}. "

    if intent == "documents":
        st.markdown("##### 📄 Documents Required")
        for d in docs: st.write(f"• {d}")
        spoken += "Required documents: " + ", ".join(docs) + "."

    elif intent == "eligibility":
        st.markdown("##### ✅ Eligibility")
        st.write(elig)
        spoken += f"Eligibility: {elig}."

    elif intent == "benefits":
        st.markdown("##### 🎁 Benefits")
        st.write(ben)
        spoken += f"Benefits: {ben}."

    elif intent == "category":
        st.markdown("##### 🏷️ Category")
        st.write(cat)
        spoken += f"Category: {cat}."

    else:
        st.markdown("##### 🏷️ Category")
        st.write(cat)
        st.markdown("##### ✅ Eligibility")
        st.write(elig)
        st.markdown("##### 🎁 Benefits")
        st.write(ben)
        st.markdown("##### 📄 Documents Required")
        for d in docs: st.write(f"• {d}")
        spoken += (
            f"Category: {cat}. Eligibility: {elig}. "
            f"Benefits: {ben}. Documents: {', '.join(docs)}."
        )

    return spoken

# ══════════════════════════════════════════════════════════════
# OPENROUTER AI CALL
# ══════════════════════════════════════════════════════════════
def ask_openrouter(question: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are SmartSeva AI, an expert on Indian government public services, "
                    "AP and Telangana state schemes, scholarships, and welfare programs. "
                    "Give a clear, helpful, structured answer. "
                    "If the question is in Telugu, reply in Telugu."
                )
            },
            {"role": "user", "content": question}
        ]
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

def call_ai_safely(query: str):
    try:
        return ask_openrouter(query), None
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 429:
            return None, "quota"
        elif status == 401 or status == 403:
            return None, "key"
        elif status == 404:
            return None, "model"
        else:
            return None, f"other: {e}"
    except requests.exceptions.RequestException as e:
        return None, f"other: {e}"
    except (KeyError, IndexError) as e:
        return None, f"other: Unexpected response format ({e})"

# ══════════════════════════════════════════════════════════════
# VOICE — SPEECH TO TEXT using browser-recorded audio
# ══════════════════════════════════════════════════════════════
def transcribe_audio_bytes(audio_bytes: bytes):
    """Convert recorded browser audio (bytes) into text using Google STT."""
    recognizer = sr.Recognizer()
    try:
        audio_file = io.BytesIO(audio_bytes)
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text, None
    except sr.UnknownValueError:
        return None, "Could not understand the audio. Please speak clearly and try again."
    except sr.RequestError as e:
        return None, f"Speech recognition service error: {e}"
    except Exception as e:
        return None, f"Could not process audio: {e}"

# ══════════════════════════════════════════════════════════════
# VOICE — TEXT TO SPEECH using gTTS
# ══════════════════════════════════════════════════════════════
def speak_text(text: str):
    if not TTS_AVAILABLE:
        st.error("Install: pip install gtts")
        return
    if not text.strip():
        return
    try:
        safe_text = text[:1500]
        tts = gTTS(text=safe_text, lang="en")
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3")
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")

# ══════════════════════════════════════════════════════════════
# PAGE UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="SmartSeva AI", page_icon="🤖", layout="centered")
st.title("🤖 SmartSeva AI – Public Services Assistant")
st.caption("Instant help for government services · farmer schemes · scholarships")
st.divider()

# ── Voice controls ──
col1, col2 = st.columns(2)

with col1:
    st.markdown("**🎙️ Speak your question**")
    if SPEECH_AVAILABLE:
        audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=True,
            use_container_width=True,
            key="mic"
        )
    else:
        audio = None
        st.caption("Voice input unavailable — missing libraries.")

with col2:
    enable_tts = st.toggle("🔊 Read answer aloud",
                           value=False,
                           disabled=not TTS_AVAILABLE)

if not SPEECH_AVAILABLE:
    st.caption("💡 Voice input needs: `pip install streamlit-mic-recorder SpeechRecognition`")
if not TTS_AVAILABLE:
    st.caption("💡 Read aloud needs: `pip install gtts`")

# ── Session state for pre-filling from voice ──
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

if SPEECH_AVAILABLE and audio is not None and audio.get("bytes"):
    with st.spinner("🎙️ Transcribing your voice…"):
        text, err = transcribe_audio_bytes(audio["bytes"])
    if text:
        st.session_state["user_query"] = text
        st.success(f"🎙️ Heard: **{text}**")
    elif err:
        st.warning(err)

user_query = st.text_input(
    "Ask a question:",
    value=st.session_state["user_query"],
    placeholder="e.g. What documents are needed for Aadhaar Card?"
)

# Quick example buttons
st.caption("💡 Try asking:")
ex_cols = st.columns(3)
examples = [
    "Documents for Passport",
    "Eligibility for PM-KISAN",
    "Steps to apply for PAN Card",
    "Fee for Driving Licence",
    "Documents for Aadhaar Card",
    "Benefits of Kisan Credit Card",
]
for i, ex in enumerate(examples):
    with ex_cols[i % 3]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["user_query"] = ex
            st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# MAIN LOGIC
# ══════════════════════════════════════════════════════════════
if user_query and user_query.strip():
    query     = user_query.strip()
    intent    = detect_intent(query)
    found     = False
    tts_parts = []

    best_service = find_single_best_match(query, services_data.get("services", []))
    if best_service:
        spoken = show_service(best_service, intent)
        tts_parts.append(spoken)
        found = True

    best_scheme = find_single_best_match(query, farmers_data.get("schemes", []))
    if best_scheme:
        spoken = show_scheme(best_scheme, intent)
        tts_parts.append(spoken)
        found = True

    best_scholarship = find_single_best_match(query, scholarships_data.get("scholarships", []))
    if best_scholarship:
        spoken = show_scheme(best_scholarship, intent)
        tts_parts.append(spoken)
        found = True

    if not found:
        with st.spinner("🤖 Asking AI…"):
            answer, err = call_ai_safely(query)

        if answer:
            st.subheader("🤖 AI Answer")
            st.write(answer)
            tts_parts.append(answer)

        elif err == "quota":
            st.warning(
                "⚠️ **OpenRouter rate limit reached.**\n\n"
                "Free models have a request-per-minute limit. "
                "Wait a minute and try again, or add the missing service/scheme "
                "to your JSON files so it works without AI."
            )
        elif err == "key":
            st.error(
                "❌ Invalid OpenRouter API key. "
                "Check that `OPENROUTER_API_KEY` in your `.env` file (or Streamlit Secrets) is correct."
            )
        elif err == "model":
            st.error(
                "❌ Model not found. The free model name may have changed — "
                "check https://openrouter.ai/models for current free model IDs."
            )
        else:
            st.error(f"❌ AI error: {err}")

    if enable_tts and tts_parts:
        speak_text(" ".join(tts_parts))
