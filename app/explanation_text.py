def generate_explanation(prediction, confidence, text_tokens, mismatch_score, modality_scores=None):
    parts = []

    pct = round(confidence * 100)
    parts.append(f"The system predicts {prediction} with {pct}% confidence.")

    top = sorted(text_tokens, key=lambda t: abs(t["weight"]), reverse=True)[:3]
    words = [t["token"].replace("##", "") for t in top if abs(t["weight"]) > 0.2]
    if words:
        parts.append("The words that influenced this decision most were: " + ", ".join(words) + ".")

    if mismatch_score >= 0.6:
        parts.append("The caption and the image show a strong mismatch, which suggests possible manipulation or a misleading pairing.")
    elif mismatch_score >= 0.3:
        parts.append("The caption and the image are partly inconsistent.")
    else:
        parts.append("The caption and the image appear consistent with each other.")

    if modality_scores:
        strongest = max(modality_scores, key=modality_scores.get)
        parts.append(f"The {strongest} signal contributed most to this result.")

    if prediction == "UNCERTAIN":
        parts.append("The evidence is not strong enough for a reliable decision, so manual verification is recommended.")

    return " ".join(parts)