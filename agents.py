import json
import re

from rag import RAGRetriever

class RetrievalAgent:
    def __init__(self, rag):
        self.rag = rag
    def run(self, question, answer):
        return self.rag.retrieve(question + " " + answer)

class OntologyAgent:
    def __init__(self, ontology_path="ontology.json"):
        with open(ontology_path, encoding="utf-8") as f:
            self.ontology = json.load(f)
    def check_concepts(self, answer):
        found = []
        for entity in self.ontology["entities"]:
            if entity in answer:
                found.append(entity)
        return found

class EvaluationAgent:
    def __init__(self, questions_path="questions.json"):
        with open(questions_path, encoding="utf-8") as f:
            self.questions = json.load(f)
    def evaluate(self, qid, answer, retrieved, concepts):
        q = next(q for q in self.questions if q["id"] == qid)
        breakdown = []
        total = 0
        answer_lower = answer.lower()
        answer_tokens = self._tokenize(answer_lower)
        answer_token_count = len(answer_tokens)
        # Gather retrieved text for overlap fallback scoring
        retrieved_text = " ".join([doc["text"] for doc in retrieved]).lower()
        retrieved_tokens = self._tokenize(retrieved_text)
        length_ratio = min(1.0, answer_token_count / 100) if answer_token_count else 0.0
        length_factor = 0.6 + (0.4 * length_ratio)
        short_answer_penalty = 0.5 if answer_token_count < 25 else 1.0
        if answer_token_count < 40:
            length_cap = 0.85
        elif answer_token_count < 80:
            length_cap = 0.95
        else:
            length_cap = 1.0
        criteria_covered = 0
        for c in q["criteria"]:
            keywords = c.get("keywords", [c["name"]])
            keyword_hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
            keyword_ratio = keyword_hits / max(1, len(keywords))
            overlap = self._jaccard(answer_tokens, retrieved_tokens)
            content_ratio = ((0.75 * keyword_ratio) + (0.25 * length_ratio)) * length_factor * length_cap
            if keyword_ratio == 0 and overlap > 0.05:
                content_ratio = max(content_ratio, overlap * 0.1)
            marks = round(c["marks"] * content_ratio * short_answer_penalty)
            if keyword_hits > 0 and marks == 0:
                marks = 1
            if keyword_ratio >= 0.7:
                marks = max(marks, round(c["marks"] * 0.7))
            elif keyword_ratio >= 0.4 and answer_token_count >= 60:
                marks = max(marks, round(c["marks"] * 0.5))
            marks = max(0, min(marks, c["marks"]))
            if keyword_ratio >= 0.3 or overlap >= 0.08:
                criteria_covered += 1
            breakdown.append({"criteria": c["name"], "marks": marks, "max": c["marks"]})
            total += marks
        if criteria_covered >= 4 and answer_token_count >= 80:
            total = min(20, total + 3)
        elif criteria_covered >= 3 and answer_token_count >= 60:
            total = min(20, total + 1)
        return total, breakdown

    def _tokenize(self, text: str) -> set[str]:
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        return {w for w in words if len(w) >= 2}

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

class ExplanationAgent:
    def explain(self, breakdown, retrieved, concepts):
        explanation = ""
        for b in breakdown:
            if b["marks"] < b["max"]:
                explanation += f"{b['criteria']} සඳහා ලකුණු අඩු විය. "
        if concepts:
            explanation += "\nඔබගේ පිළිතුරේ සම්බන්ධිත සංකල්ප: " + ", ".join(concepts)
        return explanation
