
import json
import os
import re
import tempfile

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from sqlalchemy.orm import Session

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.database import get_db
from app.models.analysis import Analysis

router = APIRouter()


@router.get("/report/{analysis_id}")
def generate_report(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    analysis = (
        db.query(Analysis)
        .filter(Analysis.id == analysis_id)
        .first()
    )

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    result = json.loads(analysis.result_json)

    fd, pdf_path = tempfile.mkstemp(
        suffix=".pdf",
        prefix="verification_report_"
    )
    os.close(fd)

    pdf = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    y = height - 60

    def write_line(label, value):
        nonlocal y

        if y < 60:
            pdf.showPage()
            y = height - 60

        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(50, y, label)

        y -= 18

        pdf.setFont("Helvetica", 10)

        # Wrap long text within the page width.
        words = str(value).split()
        line = ""

        for word in words:
            test_line = f"{line} {word}".strip()

            if pdf.stringWidth(test_line, "Helvetica", 10) > width - 100:
                pdf.drawString(50, y, line)
                y -= 15
                line = word
            else:
                line = test_line

        if line:
            pdf.drawString(50, y, line)
            y -= 25

    pdf.setTitle("Deepfake Verification Report")

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "Deepfake Verification Report")
    y -= 40

    write_line("Analysis ID:", analysis.id)
    write_line("Prediction:", result.get("prediction", "N/A"))
    write_line("Confidence:", f"{result.get('confidence', 0) * 100:.2f}%")
    write_line(
        "Manipulation Type:",
        result.get("manipulation_type", "N/A")
    )

    write_line(
        "Explanation:",
        result.get("explanation", "N/A")
    )

    evidence = result.get("evidence", {})

    write_line(
        "Mismatch Score:",
        evidence.get("mismatch_score", "N/A")
    )

    write_line("Flagged Text Tokens:", "")

    tokens = evidence.get("text_tokens", [])

    if tokens:
        for item in tokens:
            write_line(
                item.get("token", "N/A"),
                f"Weight: {item.get('weight', 0)}"
            )
    else:
        write_line("Tokens:", "No flagged tokens available.")

    heatmap_url = evidence.get("image_heatmap_url")

    write_line(
        "Heatmap URL:",
        heatmap_url if heatmap_url else "Not available"
    )

    pdf.save()

    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(analysis_id))

    return FileResponse(
    pdf_path,
    media_type="application/pdf",
    filename=f"verification_report_{safe_id}.pdf",
    background=BackgroundTask(os.remove, pdf_path)
    )