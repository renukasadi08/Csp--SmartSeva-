import streamlit as st
import json
import os
import requests
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════
# VOICE IMPORTS
# ══════════════════════════════════════════════════════════════
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# CONFIG — OpenRouter
# ══════════════════════════════════════════════════════════════
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("❌ OPENROUTER_API_KEY not found. Add it to your .env file.")
    st.stop()

# Free models available on OpenRouter — we try them in order
# If one fails (quota/unavailable), next one is tried automatically
FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-8b:free",
    "mistralai/mistral-7b-instruct:free",
]

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
# OPENROUTER AI CALL
# Uses requests library (no special SDK needed)
# Tries each free model until one works
# ══════════════════════════════════════════════════════════════
def call_openrouter(question: str):
    """
    Calls OpenRouter API with the question.
    Tries multiple free models in order until one succeeds.
    Returns (answer_text, error_message)
    """
    prompt = (
        "You are SmartSeva AI, an expert on Indian government public services, "
        "Andhra Pradesh and Telangana state schemes, scholarships, and welfare programs. "
        "Give a clear, helpful, structured answer. "
        "If the question is in Telugu, reply in Telugu.\n\n"
        f"Question: {question}"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",   # required by OpenRouter
        "X-Title": "SmartSeva AI"                  # shown in OpenRouter dashboard
    }

    for model in FREE_MODELS:
        try:
            payload = {
                "model": "openrouter/free",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            print("Status:", response.status_code)
            print(response.text)
            print("Response:", response.text)

            if response.status_code == 200:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                return answer, None

            elif response.status_code == 429:
                # This model is rate limited, try next one
                continue

            elif response.status_code == 401:
                return None, "key"

            else:
                # Some other error on this model, try next
                continue

        except requests.exceptions.Timeout:
            continue
        except Exception:
            continue

    # All models failed
    return None, "all_failed"

# ══════════════════════════════════════════════════════════════
# SMART MATCHING
# "card" is in IGNORE_WORDS so "aadhaar card" won't also match
# "PAN Card" or "Kisan Credit Card"
# ══════════════════════════════════════════════════════════════
IGNORE_WORDS = {
    "for", "the", "and", "are", "what", "how", "do", "to", "a", "an",
    "of", "in", "is", "me", "about", "can", "my", "give", "tell",
    "required", "documents", "eligibility", "fee", "steps", "process",
    "time", "benefits", "category", "please", "list", "scheme", "with",
    "get", "need", "apply", "information", "details", "card"
}

def score_match(item_name: str, query: str) -> int:
    name_words = [
        w.strip("()").lower()
        for w in item_name.split()
        if len(w.strip("()")) > 2 and w.strip("()").lower() not in IGNORE_WORDS
    ]
    if not name_words:
        return 0
    query_lower = query.lower()
    return sum(1 for w in name_words if w in query_lower)

def find_best_match(query: str, items: list, name_key: str = "name"):
    scored = [(score_match(item.get(name_key, ""), query), item)
              for item in items]
    scored = [(s, item) for s, item in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    top_score   = scored[0][0]
    top_matches = [item for s, item in scored if s == top_score]
    return top_matches[0] if len(top_matches) == 1 else None

# ══════════════════════════════════════════════════════════════
# INTENT DETECTION
# ══════════════════════════════════════════════════════════════
def detect_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["document","documents","proof","papers","needed","required","కావాల్సిన"]):
        return "documents"
    if any(w in q for w in ["fee","cost","charge","charges","money","price","pay","రుసుము"]):
        return "fee"
    if any(w in q for w in ["step","steps","how to apply","process","procedure","apply","దరఖాస్తు"]):
        return "steps"
    if any(w in q for w in ["time","processing","how long","days","duration","weeks","రోజులు"]):
        return "time"
    if any(w in q for w in ["eligible","eligibility","qualify","who can","criteria","అర్హత"]):
        return "eligibility"
    if any(w in q for w in ["benefit","benefits","amount","receive","grant","how much","లబ్ధి"]):
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
        spoken += f"The fee is: {fee}."

    elif intent == "steps":
        st.markdown("##### 📋 Steps to Apply")
        for i, s in enumerate(steps, 1): st.write(f"{i}. {s}")
        spoken += "Steps to apply: " + ". ".join(steps) + "."

    elif intent == "time":
        st.markdown("##### ⏱️ Processing Time")
        st.write(time_val)
        spoken += f"Processing time is: {time_val}."

    else:
        st.markdown("##### 📄 Required Documents")
        for d in docs: st.write(f"• {d}")
        st.markdown("##### 📋 Steps to Apply")
        for i, s in enumerate(steps, 1): st.write(f"{i}. {s}")
        st.markdown("##### 💰 Fee")
        st.write(fee)
        st.markdown("##### ⏱️ Processing Time")
        st.write(time_val)
        spoken += (f"Documents: {', '.join(docs)}. Steps: {'. '.join(steps)}. "
                   f"Fee: {fee}. Time: {time_val}.")
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
        spoken += (f"Category: {cat}. Eligibility: {elig}. "
                   f"Benefits: {ben}. Documents: {', '.join(docs)}.")
    return spoken

# ══════════════════════════════════════════════════════════════
# VOICE
# ══════════════════════════════════════════════════════════════
def listen_voice():
    if not SPEECH_AVAILABLE:
        st.error("Install: pip install SpeechRecognition pyaudio")
        return None
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Listening… speak clearly now")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("No speech detected. Please try again.")
    except sr.UnknownValueError:
        st.warning("Could not understand. Please speak slowly and clearly.")
    except Exception as e:
        st.error(f"Microphone error: {e}")
    return None

def speak_text(text: str):
    if not TTS_AVAILABLE:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        st.error(f"TTS error: {e}")

# ══════════════════════════════════════════════════════════════
# PAGE UI
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="SmartSeva AI", page_icon="🤖", layout="centered")
st.title("🤖 SmartSeva AI – Public Services Assistant")
st.caption("Instant answers for government services · farmer schemes · scholarships · AP & Telangana")
st.divider()

col1, col2 = st.columns(2)
with col1:
    voice_btn = st.button("🎙️ Speak your question",
                          disabled=not SPEECH_AVAILABLE,
                          use_container_width=True)
with col2:
    enable_tts = st.toggle("🔊 Read answer aloud",
                           value=False, disabled=not TTS_AVAILABLE)

if not SPEECH_AVAILABLE:
    st.caption("💡 Voice input: `pip install SpeechRecognition pyaudio`")
if not TTS_AVAILABLE:
    st.caption("💡 Read aloud: `pip install pyttsx3`")

if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

if voice_btn:
    heard = listen_voice()
    if heard:
        st.session_state["user_query"] = heard
        st.success(f"🎙️ Heard: **{heard}**")

user_query = st.text_input(
    "Ask a question:",
    value=st.session_state["user_query"],
    placeholder="e.g. What documents are needed for Aadhaar Card?"
)

st.caption("💡 Try asking:")
ex_cols = st.columns(3)
examples = [
    "Documents for Passport",
    "Eligibility for PM-KISAN",
    "Steps to apply for PAN Card",
    "Fee for Driving Licence",
    "Benefits of Kisan Credit Card",
    "Eligibility for National Fellowship for SC Students",
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

    # 1. Services
    best = find_best_match(query, services_data.get("services", []))
    if best:
        tts_parts.append(show_service(best, intent))
        found = True

    # 2. Farmer Schemes
    best = find_best_match(query, farmers_data.get("schemes", []))
    if best:
        tts_parts.append(show_scheme(best, intent))
        found = True

    # 3. Scholarships
    best = find_best_match(query, scholarships_data.get("scholarships", []))
    if best:
        tts_parts.append(show_scheme(best, intent))
        found = True

    # 4. Fallback to OpenRouter AI
    if not found:
        with st.spinner("🤖 Asking AI…"):
            answer, err = call_openrouter(query)

        if answer:
            st.subheader("🤖 AI Answer")
            st.write(answer)
            tts_parts.append(answer)

        elif err == "key":
            st.error(
                "❌ Invalid API key.\n\n"
                "Check that `OPENROUTER_API_KEY` in your `.env` file is correct.\n"
                "Get a key at: https://openrouter.ai/keys"
            )
        elif err == "all_failed":
            st.warning(
                "⚠️ All free AI models are currently busy or rate-limited.\n\n"
                "**Please wait 1–2 minutes and try again.**\n\n"
                "💡 Tip: Most services, schemes, and scholarships are already in "
                "the local knowledge base — just type the service name directly."
            )
        else:
            st.error(f"❌ Unexpected error: {err}")

    # 5. TTS
    if enable_tts and tts_parts:
        speak_text(" ".join(tts_parts))