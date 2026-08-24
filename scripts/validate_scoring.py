"""Step 1 validation: does cross-encoder/nli-deberta-v3-small separate
faithful responses from hallucinated ones on hand-written pairs?"""

import json

from sentence_transformers import CrossEncoder

PAIRS = [
    # --- faithful (label 0) ---
    {
        "context": "The Eiffel Tower is located in Paris and was completed in 1889.",
        "response": "The Eiffel Tower is in Paris.",
        "label": 0,
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level pressure.",
        "response": "At sea level, water boils at 100 degrees Celsius.",
        "label": 0,
    },
    {
        "context": "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.",
        "response": "Marie Curie won two Nobel Prizes.",
        "label": 0,
    },
    {
        "context": "The company reported revenue of $4.2 billion in Q3 2025, up 12% year over year.",
        "response": "The company's Q3 2025 revenue was $4.2 billion, a 12% increase over the prior year.",
        "label": 0,
    },
    {
        "context": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
        "response": "Plants use sunlight to convert carbon dioxide and water into glucose and oxygen.",
        "label": 0,
    },
    {
        "context": "The meeting is scheduled for Tuesday at 3 PM in Conference Room B.",
        "response": "The meeting takes place on Tuesday at 3 PM.",
        "label": 0,
    },
    {
        "context": "Python 3.12 was released in October 2023.",
        "response": "Python 3.12 came out in October 2023.",
        "label": 0,
    },
    {
        "context": "The patient's blood pressure was 120/80 and their heart rate was 72 beats per minute.",
        "response": "The patient had a heart rate of 72 beats per minute.",
        "label": 0,
    },
    # --- hallucinated (label 1): contradictions ---
    {
        "context": "The Eiffel Tower is located in Paris and was completed in 1889.",
        "response": "The Eiffel Tower was completed in 1923 and is located in Lyon.",
        "label": 1,
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level pressure.",
        "response": "Water boils at 50 degrees Celsius at sea level.",
        "label": 1,
    },
    {
        "context": "Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.",
        "response": "Marie Curie won three Nobel Prizes, all in Physics.",
        "label": 1,
    },
    {
        "context": "The company reported revenue of $4.2 billion in Q3 2025, up 12% year over year.",
        "response": "The company reported a loss of $1 billion in Q3 2025.",
        "label": 1,
    },
    {
        "context": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
        "response": "Photosynthesis converts oxygen and glucose into carbon dioxide using moonlight.",
        "label": 1,
    },
    {
        "context": "The meeting is scheduled for Tuesday at 3 PM in Conference Room B.",
        "response": "The meeting is scheduled for Friday at 9 AM in the main auditorium.",
        "label": 1,
    },
    # --- hallucinated (label 1): unsupported claim, not a contradiction ---
    {
        "context": "Python 3.12 was released in October 2023.",
        "response": "Python 3.12 was released in October 2023 and introduced built-in quantum computing support.",
        "label": 1,
    },
]

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


def main() -> None:
    model = CrossEncoder(MODEL_NAME)
    id2label = {v: k for k, v in model.config.label2id.items()}
    print(f"labels: {id2label}")
    ent_idx = model.config.label2id["entailment"]

    rows = []
    pairs = [(p["context"], p["response"]) for p in PAIRS]
    scores = model.predict(pairs, apply_softmax=True)
    for p, s in zip(PAIRS, scores):
        risk = 1.0 - float(s[ent_idx])
        rows.append(
            {
                "label": p["label"],
                "risk": risk,
                "contradiction": float(s[model.config.label2id["contradiction"]]),
                "entailment": float(s[ent_idx]),
                "neutral": float(s[model.config.label2id["neutral"]]),
            }
        )

    faithful = sorted(r["risk"] for r in rows if r["label"] == 0)
    halluc = sorted(r["risk"] for r in rows if r["label"] == 1)
    print("\nidx  label  risk    contra   entail   neutral")
    for i, r in enumerate(rows):
        tag = "faithful" if r["label"] == 0 else "HALLUC"
        print(
            f"{i:>3}  {tag:<8} {r['risk']:.4f}  {r['contradiction']:.4f}  {r['entailment']:.4f}  {r['neutral']:.4f}"
        )

    print(f"\nfaithful   risks: {[round(x, 4) for x in faithful]}")
    print(f"hallucinat risks: {[round(x, 4) for x in halluc]}")
    gap = min(halluc) - max(faithful)
    print(
        f"\nmax faithful risk: {max(faithful):.4f} | min hallucination risk: {min(halluc):.4f} | gap: {gap:.4f}"
    )
    print("SEPARATION:", "CLEAN" if gap > 0.1 else "OVERLAPPING - needs fixing")

    with open(
        r"C:\Users\Kunal\AppData\Local\Temp\opencode\labeled_pairs.jsonl", "w", encoding="utf-8"
    ) as f:
        for p in PAIRS:
            f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
