import json
from datetime import datetime, timezone
from pathlib import Path


import requests
import streamlit as st
from rag import RAGRetriever
from agents import RetrievalAgent, OntologyAgent, EvaluationAgent, ExplanationAgent
import json

CONFIG_PATH = Path(__file__).with_name("config.json")
HISTORY_DIR = Path(__file__).with_name("history")
DEFAULT_CONFIG = {
    "ollama_url": "http://localhost:11434",
    "models": ["llama3", "gemma"],
    "default_model": "llama3",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_CONFIG


def ollama_generate(prompt: str, model: str, temperature: float, base_url: str) -> str:
    url = f"{base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as exc:
        return f"Error contacting Ollama: {exc}"
    except json.JSONDecodeError:
        return "Error: received an invalid response from Ollama."


def save_history(payload: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_path = HISTORY_DIR / "evaluations.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    
def load_history(limit: int = 50) -> list[dict]:
    history_path = HISTORY_DIR / "evaluations.jsonl"
    if not history_path.exists():
        return []
    lines = history_path.read_text(encoding="utf-8").splitlines()
    records = []
    for line in lines[-limit:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
    
@st.dialog("Evaluation History")
def show_history_dialog() -> None:
    history = load_history(limit=100)
    if not history:
        st.info("No history yet. Evaluate an answer to create entries.")
        return
    st.caption("Most recent 100 evaluations")
    table = [
        {
            "Time": item.get("timestamp", ""),
            "Question": item.get("question", ""),
            "Score": f"{item.get('score', 0)}/{item.get('max_score', 20)}",
        }
        for item in reversed(history)
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)


config = load_config()
models = config.get("models", DEFAULT_CONFIG["models"])
default_model = config.get("default_model", DEFAULT_CONFIG["default_model"])
ollama_url = config.get("ollama_url", DEFAULT_CONFIG["ollama_url"])
embedding_model = config.get("embedding_model", "nomic-embed-text")

st.set_page_config(page_title="Sri Guru AI", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Serif:wght@400;600&display=swap');

    :root {
        --bg: #0f1220;
        --panel: #171a2b;
        --panel-2: #1f2340;
        --accent: #f0b429;
        --accent-2: #2dd4bf;
        --text: #f6f7fb;
        --muted: #b6bdd6;
    }

    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #273065 0%, transparent 60%),
                    radial-gradient(900px 500px at 90% 0%, #1a4138 0%, transparent 55%),
                    var(--bg);
        color: var(--text);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.2px;
    }

    .sg-hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(240,180,41,0.12), rgba(45,212,191,0.12));
        border: 1px solid rgba(240, 180, 41, 0.15);
        box-shadow: 0 10px 30px rgba(7, 10, 24, 0.35);
        animation: fadeUp 0.7s ease-out;
    }

    .sg-title {
        font-size: 2.4rem;
        margin-bottom: 0.35rem;
    }

    .sg-subtitle {
        font-family: 'IBM Plex Serif', serif;
        color: var(--muted);
        font-size: 1.05rem;
        margin-top: 0.2rem;
    }

    .sg-card {
        background: linear-gradient(140deg, rgba(31,35,64,0.9), rgba(23,26,43,0.92));
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 12px 28px rgba(7, 10, 24, 0.35);
        animation: fadeUp 0.7s ease-out;
    }

    .sg-chip {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        background: rgba(240, 180, 41, 0.15);
        color: var(--accent);
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
        font-family: 'Space Grotesk', sans-serif;
    }

    .sg-score {
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0.2rem 0 0.6rem 0;
        font-family: 'Space Grotesk', sans-serif;
    }

    .sg-muted {
        color: var(--muted);
        font-family: 'IBM Plex Serif', serif;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(31,35,64,0.96), rgba(15,18,32,0.96));
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .stTextArea textarea {
        background: #14182b;
        border: 1px solid rgba(255,255,255,0.08);
        color: var(--text);
    }

    .stButton button {
        background: linear-gradient(120deg, #f0b429, #ffd166);
        color: #1a1a1a;
        border: none;
        padding: 0.6rem 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 10px 22px rgba(240, 180, 41, 0.25);
        transition: transform 0.15s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px) scale(1.01);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sg-hero">
        <div class="sg-chip">Offline Sinhala History Scorer</div>
        <div class="sg-title">Sri Guru AI</div>
        <div class="sg-subtitle">Ollama + RAG + Ontology + Agents, tuned for clear academic feedback.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        models,
        index=models.index(default_model) if default_model in models else 0,
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)
    st.caption("Make sure Ollama is running locally.")
    if st.button("View Evaluation History"):
        show_history_dialog()

    # Load questions
    with open("questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    question_options = [q["question"] for q in questions]
    selected_q_idx = st.selectbox("Select Question", range(len(question_options)), format_func=lambda i: question_options[i])
    selected_question = questions[selected_q_idx]


# --- AGENT SYSTEM ---
rag = RAGRetriever(base_url=ollama_url, embedding_model=embedding_model)
retrieval_agent = RetrievalAgent(rag)
ontology_agent = OntologyAgent()
evaluation_agent = EvaluationAgent()
explanation_agent = ExplanationAgent()

st.markdown("<div class=\"sg-card\">", unsafe_allow_html=True)
st.markdown("<div class=\"sg-chip\">Question</div>", unsafe_allow_html=True)
st.markdown(f"<h3>📝 {selected_question['question']}</h3>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style=\"height: 0.8rem;\"></div>", unsafe_allow_html=True)
answer = st.text_area("Enter your answer in Sinhala:", height=180)

if st.button("Evaluate Answer") and answer.strip():
    with st.spinner("Scoring your answer..."):
        # 1. Retrieve relevant knowledge
        retrieved = retrieval_agent.run(selected_question["question"], answer)
        # 2. Check ontology concepts
        concepts = ontology_agent.check_concepts(answer)
        # 3. Evaluate answer
        total, breakdown = evaluation_agent.evaluate(selected_question["id"], answer, retrieved, concepts)
        # 4. Generate explanation
        explanation = explanation_agent.explain(breakdown, retrieved, concepts)

        save_history(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question_id": selected_question["id"],
                "question": selected_question["question"],
                "answer": answer,
                "score": total,
                "max_score": 20,
                "breakdown": breakdown,
                "concepts": concepts,
                "retrieved_sources": [doc.get("filename") for doc in retrieved],
            }
        )

    st.markdown("<div class=\"sg-card\">", unsafe_allow_html=True)
    st.markdown("<div class=\"sg-chip\">Evaluation</div>", unsafe_allow_html=True)
    st.markdown(f"<div class=\"sg-score\">{total}/20</div>", unsafe_allow_html=True)
    st.markdown("<div class=\"sg-muted\">Breakdown</div>", unsafe_allow_html=True)
    st.table([{"Criteria": b["criteria"], "Marks": f'{b["marks"]}/{b["max"]}'} for b in breakdown])
    st.markdown("<div class=\"sg-muted\" style=\"margin-top:0.7rem;\">Explanation</div>", unsafe_allow_html=True)
    st.write(explanation)
    st.markdown("</div>", unsafe_allow_html=True)
