import json
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
        # Improved: partial matching and use of retrieved knowledge/concepts
        q = next(q for q in self.questions if q["id"] == qid)
        breakdown = []
        total = 0
        answer_lower = answer.lower()
        # Gather all retrieved text for context scoring
        retrieved_text = " ".join([doc["text"] for doc in retrieved]).lower()
        for c in q["criteria"]:
            crit = c["name"].lower()
            # Partial match: if any word in criteria is in answer or retrieved or concepts
            crit_words = crit.split()
            found = False
            for word in crit_words:
                if word in answer_lower or word in retrieved_text:
                    found = True
                    break
            # Bonus: if ontology concept matches criteria
            concept_match = any(word in (concept.lower() for concept in concepts) for word in crit_words)
            # Assign marks: full if found, half if concept match, else 0
            if found:
                marks = c["marks"]
            elif concept_match:
                marks = max(1, c["marks"] // 2)
            else:
                marks = 0
            breakdown.append({"criteria": c["name"], "marks": marks, "max": c["marks"]})
            total += marks
        return total, breakdown

class ExplanationAgent:
    def explain(self, breakdown, retrieved, concepts):
        explanation = ""
        for b in breakdown:
            if b["marks"] < b["max"]:
                explanation += f"{b['criteria']} සඳහා ලකුණු අඩු විය. "
        if concepts:
            explanation += "\nඔබගේ පිළිතුරේ සම්බන්ධිත සංකල්ප: " + ", ".join(concepts)
        return explanation
