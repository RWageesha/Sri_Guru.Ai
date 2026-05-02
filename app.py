import json
from pathlib import Path


import requests
import streamlit as st
from rag import RAGRetriever
from agents import RetrievalAgent, OntologyAgent, EvaluationAgent, ExplanationAgent
import json

CONFIG_PATH = Path(__file__).with_name("config.json")
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


config = load_config()
models = config.get("models", DEFAULT_CONFIG["models"])
default_model = config.get("default_model", DEFAULT_CONFIG["default_model"])
ollama_url = config.get("ollama_url", DEFAULT_CONFIG["ollama_url"])

st.set_page_config(page_title="Sri Guru AI", page_icon="AI", layout="wide")


st.title("Sri Guru AI")
st.write("Sinhala History Answer Scorer (Offline, OLLAMA + RAG + Ontology + Agents)")


with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        models,
        index=models.index(default_model) if default_model in models else 0,
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.05)
    st.caption("Make sure Ollama is running locally.")

    # Load questions
    with open("questions.json", encoding="utf-8") as f:
        questions = json.load(f)
    question_options = [q["question"] for q in questions]
    selected_q_idx = st.selectbox("Select Question", range(len(question_options)), format_func=lambda i: question_options[i])
    selected_question = questions[selected_q_idx]


# --- AGENT SYSTEM ---
rag = RAGRetriever()
retrieval_agent = RetrievalAgent(rag)
ontology_agent = OntologyAgent()
evaluation_agent = EvaluationAgent()
explanation_agent = ExplanationAgent()

st.subheader("📝 Question")
st.write(selected_question["question"])

answer = st.text_area("Enter your answer in Sinhala:", height=150)

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

    st.success(f"Score: {total}/20")
    st.markdown("### Breakdown:")
    st.table([{"Criteria": b["criteria"], "Marks": f'{b["marks"]}/{b["max"]}'} for b in breakdown])
    st.markdown("### Explanation:")
    st.write(explanation)
