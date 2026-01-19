# app.py
# Streamlit POC: conversation -> reformulation + tensions + options + question (frugal)
#
# Run:
#   pip install streamlit
# Optional (LLM local via Ollama):
#   install Ollama, then: ollama pull llama3.1:8b (or any text model you have)
#   (No extra python deps needed for Ollama mode; uses stdlib urllib)
#
# Start:
#   streamlit run app.py

import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import streamlit as st


# ----------------------------
# Frugal heuristics (no LLM)
# ----------------------------

TENSION_PATTERNS: List[Tuple[str, str]] = [
    (r"\b(d[ée]lai|deadline|date|septembre|vite|rapidement|asap)\b", "Délai"),
    (r"\b(qualit[ée]|stable|robuste|bug|tests?|fiabilit[ée])\b", "Qualité"),
    (r"\b(dette|technique|refacto|refactor|mainten)\b", "Dette technique"),
    (r"\b(co[uû]t|budget|€|cher|moins cher)\b", "Coût"),
    (r"\b(ressources?|[ée]quipe|capacity|capacit[ée]|dispo|disponible)\b", "Capacité"),
    (r"\b(risque|risqu[ée]s?|danger|bloquant)\b", "Risque"),
    (r"\b(client|m[ée]tier|stakeholder|comit[ée])\b", "Pression externe"),
    (r"\b(p[ée]rim[èe]tre|scope|MVP|minimum|r[ée]duire)\b", "Périmètre"),
]

DEFAULT_OPTIONS = [
    "Réduire le périmètre (MVP) pour tenir la date",
    "Décaler l’échéance avec un plan et des jalons",
    "Livrer à la date en acceptant une dette technique explicite + plan de rattrapage",
    "Renforcer temporairement la capacité (renfort, réallocation, sous-traitance ciblée)",
    "Scinder en livraisons: une première valeur tôt, le reste ensuite",
]


def _extract_tensions(text: str) -> List[str]:
    hits = []
    for pat, label in TENSION_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(label)
    # dedupe preserve order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out


def heuristics_analyze(transcript: str) -> Dict:
    transcript = transcript.strip()
    tensions = _extract_tensions(transcript)

    # naive reformulation: pick 1-2 sentences (first + last) if available
    sentences = re.split(r"(?<=[\.\!\?])\s+", transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        reformulation = f"{sentences[0]} … {sentences[-1]}"
    elif sentences:
        reformulation = sentences[0]
    else:
        reformulation = ""

    # pick options: if "scope"/"périmètre" appears, prioritize scope reduction
    options = DEFAULT_OPTIONS.copy()
    if "Périmètre" in tensions:
        options = [
            "Réduire le périmètre (MVP) pour tenir la date",
            "Scinder en livraisons: une première valeur tôt, le reste ensuite",
            "Décaler l’échéance avec un plan et des jalons",
            "Livrer à la date en acceptant une dette technique explicite + plan de rattrapage",
            "Renforcer temporairement la capacité (renfort, réallocation, sous-traitance ciblée)",
        ]

    # question: pick top 2 tensions to turn into a decision question
    if len(tensions) >= 2:
        question = f"Quelle priorité prime: {tensions[0]} ou {tensions[1]} ?"
    elif tensions:
        question = f"Quel niveau de {tensions[0].lower()} est acceptable pour l’équipe ?"
    else:
        question = "Quel compromis veut-on formaliser (date, périmètre, qualité, coût) ?"

    return {
        "mode": "heuristics",
        "reformulation": reformulation,
        "tensions": tensions,
        "options": options[:5],
        "decision_question": question,
        "notes": [
            "Sortie basée sur heuristiques (mots-clés + templates).",
            "Utile pour un POC frugal sans modèle."
        ],
    }


# ----------------------------
# Local LLM via Ollama (optional)
# ----------------------------

@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    max_tokens: int = 450


LLM_SYSTEM = """Tu es un assistant personnel. Tu aides à la décision en reformulant et en structurant.
Contraintes:
- Ne décide pas à la place des humains.
- N’invente pas de faits non présents dans la conversation.
- Si une info manque, indique-la comme "inconnue".
- Réponds STRICTEMENT en JSON, sans texte autour.
Schéma:
{
  "reformulation": string,
  "tensions": [string],
  "options": [{"title": string, "tradeoffs": string}],
  "decision_question": string,
  "unknowns": [string]
}
"""

def ollama_analyze(transcript: str, cfg: OllamaConfig) -> Dict:
    prompt = f"""Conversation (texte brut):
\"\"\"\n{transcript}\n\"\"\"\n
Produis la sortie JSON selon le schéma."""
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
        },
    }
    url = f"{cfg.host}/api/chat"
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError) as e:
        raise RuntimeError(f"Ollama non joignable ou erreur API: {e}")

    content = data.get("message", {}).get("content", "").strip()

    # Try to parse JSON, with a small salvage if model wrapped it
    try:
        return {"mode": "ollama", **json.loads(content)}
    except json.JSONDecodeError:
        # salvage: extract first {...} block
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if m:
            return {"mode": "ollama", **json.loads(m.group(0))}
        raise RuntimeError("Réponse Ollama non-JSON. Ajuste le modèle/prompt.")


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="POC Aide à la décision (frugal)", layout="wide")
st.title("POC — Analyse de conversation → aide à la décision (frugal)")

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Choisir le moteur", ["Heuristiques (sans LLM)", "LLM local (Ollama)"], index=0)

    if mode == "LLM local (Ollama)":
        st.subheader("Ollama")
        host = st.text_input("Host", value="http://localhost:11434")
        model = st.text_input("Modèle", value="llama3.1:8b")
        temperature = st.slider("Température", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.slider("Max tokens (approx.)", 100, 1200, 450, 50)
        cfg = OllamaConfig(host=host, model=model, temperature=float(temperature), max_tokens=int(max_tokens))
    else:
        cfg = None

    st.divider()
    st.caption("Astuce POC: commence en mode heuristiques, puis active Ollama si tu veux une reformulation plus fine.")


if "transcript" not in st.session_state:
    st.session_state.transcript = ""

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Entrée conversation")
    st.session_state.transcript = st.text_area(
        "Colle ici un extrait (ou ajoute des messages au fil de l’eau)",
        value=st.session_state.transcript,
        height=300,
        placeholder="Ex: On peut livrer plus vite mais ça crée de la dette… Le client pousse pour septembre…",
    )

    with st.expander("Ajout rapide (simulateur temps réel)", expanded=False):
        msg = st.text_input("Nouveau message")
        if st.button("Ajouter au transcript"):
            if msg.strip():
                st.session_state.transcript = (st.session_state.transcript.rstrip() + "\n" + msg.strip()).strip()
                st.rerun()

    analyze = st.button("Analyser", type="primary", use_container_width=True)

with col2:
    st.subheader("Sortie aide à la décision")

    if analyze:
        transcript = st.session_state.transcript.strip()
        if not transcript:
            st.warning("Ajoute du texte dans la conversation.")
        else:
            with st.spinner("Analyse…"):
                try:
                    if mode == "Heuristiques (sans LLM)":
                        out = heuristics_analyze(transcript)
                    else:
                        out = ollama_analyze(transcript, cfg)  # type: ignore[arg-type]
                except Exception as e:
                    st.error(str(e))
                    out = None

            if out:
                st.markdown("### Reformulation")
                st.write(out.get("reformulation", ""))

                st.markdown("### Tensions détectées")
                tensions = out.get("tensions", [])
                if tensions:
                    st.write(", ".join(tensions))
                else:
                    st.write("Aucune (ou insuffisant).")

                st.markdown("### Options")
                options = out.get("options", [])
                if options and isinstance(options[0], dict):
                    for i, opt in enumerate(options, 1):
                        st.markdown(f"**{i}. {opt.get('title','')}**")
                        st.write(opt.get("tradeoffs", ""))
                else:
                    for i, opt in enumerate(options, 1):
                        st.markdown(f"**{i}. {opt}**")

                st.markdown("### Question de décision")
                st.info(out.get("decision_question", ""))

                unknowns = out.get("unknowns", [])
                if unknowns:
                    st.markdown("### Inconnues à clarifier")
                    for u in unknowns:
                        st.write(f"- {u}")

                st.markdown("### JSON (debug)")
                st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")


