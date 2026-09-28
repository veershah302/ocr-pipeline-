from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

from .io import write_json
from .models import Block

PAGE_W, PAGE_H = letter


@dataclass(frozen=True)
class Fixture:
    name: str
    blocks: list[tuple[str, str, int | None]]
    layout: str


FIXTURES = [
    Fixture("single_column", [("heading", "Community Diabetes Prevention Initiative", 1), ("heading", "1. Project Summary", 2), ("paragraph", "This proposal tests a community health worker program to improve hemoglobin A1c screening and follow-up for adults at elevated risk of type 2 diabetes.", None), ("heading", "1.1 Need", 3), ("paragraph", "The target clinics serve rural communities where transportation barriers and food insecurity contribute to avoidable diabetes complications.", None), ("heading", "1.2 Expected Impact", 3), ("paragraph", "Within eighteen months, the project will enroll 240 participants and increase completed follow-up visits by twenty percent.", None)], "single"),
    Fixture("two_column", [("heading", "Research Strategy: Remote Blood Pressure Monitoring", 1), ("heading", "2. Significance", 2), ("paragraph", "Hypertension remains a leading modifiable cause of stroke. Home monitoring can identify sustained elevation between clinic visits.", None), ("heading", "2.1 Innovation", 3), ("paragraph", "The study combines Bluetooth cuffs, bilingual coaching, and clinician alerts in a pragmatic primary-care workflow.", None), ("heading", "2.2 Approach", 3), ("paragraph", "Participants will be randomized to usual care or remote monitoring. The primary endpoint is change in systolic blood pressure at six months.", None)], "columns"),
    Fixture("tables", [("heading", "Budget and Milestones", 1), ("heading", "3. Implementation Plan", 2), ("paragraph", "The implementation team will coordinate recruitment, data collection, and quarterly quality reviews across three clinics.", None), ("table", "Year | Personnel | Supplies | Total; Year 1 | $120,000 | $18,000 | $138,000; Year 2 | $126,000 | $12,000 | $138,000", None), ("heading", "Milestone Review", 2), ("paragraph", "Quarterly dashboards will report enrollment, missing data, adverse events, and participant retention.", None)], "table"),
    Fixture("figure_caption", [("heading", "Methods and Logic Model", 1), ("heading", "4. Intervention Workflow", 2), ("paragraph", "A nurse navigator reviews referrals, confirms eligibility, and schedules a telehealth education session within seven days.", None), ("figure", "Referral Navigator Education Follow-up", None), ("caption", "Figure 1. Participants move from referral to navigator outreach, telehealth education, and thirty-day follow-up.", None), ("paragraph", "Escalation protocols route uncontrolled symptoms to a clinician on the same business day.", None)], "figure"),
    Fixture("lists_notes", [("heading", "Data Management and Dissemination", 1), ("heading", "5. Data Safeguards", 2), ("paragraph", "The study will use a limited data set stored in an encrypted research environment.", None), ("list", "Obtain documented consent before enrollment; Assign a study identifier at intake; Review access logs each month", None), ("footnote", "1. The limited data set excludes direct identifiers and is reviewed by the institutional privacy office.", None), ("heading", "References", 1), ("reference", "Smith J, Patel R. Community navigation and chronic disease outcomes. Journal of Primary Care. 2024;18:101-110.", None)], "list"),
    Fixture("mixed_appendix", [("heading", "Appendix A: Reviewer Instructions and Site Readiness", 1), ("paragraph", "Reviewers should assess feasibility, equity, implementation readiness, and the adequacy of the proposed evaluation plan.", None), ("heading", "Site Readiness Checklist", 2), ("list", "Electronic health record reporting available; Community advisory board convened; Bilingual education materials approved", None), ("table", "Domain | Evidence; Staffing | Two navigators hired; Technology | Secure video platform validated", None), ("heading", "Reviewer Note", 2), ("paragraph", "Applicants must explain how findings will be returned to participating communities in plain language.", None), ("footnote", "2. Reviewer scores are advisory and do not replace institutional review requirements.", None)], "mixed"),
]


def _wrap(canvas: Canvas, text: str, x: float, y: float, width: float, size: int = 10) -> float:
    words, line = text.split(), ""
    canvas.setFont("Helvetica", size)
    for word in words:
        candidate = f"{line} {word}".strip()
        if canvas.stringWidth(candidate, "Helvetica", size) > width and line:
            canvas.drawString(x, y, line)
            y -= size + 4
            line = word
        else:
            line = candidate
    if line:
        canvas.drawString(x, y, line)
        y -= size + 4
    return y


def _draw_fixture(path: Path, fixture: Fixture) -> None:
    canvas = Canvas(str(path), pagesize=letter)
    canvas.setTitle(fixture.name)
    y = PAGE_H - 54
    col_x, col_width = 54, PAGE_W - 108
    for index, (kind, text, level) in enumerate(fixture.blocks):
        if fixture.layout == "columns" and index > 1:
            split = (len(fixture.blocks) + 1) // 2
            if index == split:
                col_x, y, col_width = PAGE_W / 2 + 14, PAGE_H - 125, PAGE_W / 2 - 68
            elif index == 1:
                y = PAGE_H - 105
        if kind == "heading":
            size = 16 if level == 1 else (13 if level == 2 else 11)
            canvas.setFillColor(HexColor("#143A5C"))
            canvas.setFont("Helvetica-Bold", size)
            y -= 9
            y = _wrap(canvas, text, col_x, y, col_width, size) - 5
            canvas.setFillColor(HexColor("#000000"))
        elif kind == "table":
            rows = [row.split(" | ") for row in text.split("; ")]
            row_h, widths = 20, [col_width / len(rows[0])] * len(rows[0])
            top = y
            for row_i, row in enumerate(rows):
                x = col_x
                for cell_i, cell in enumerate(row):
                    canvas.setStrokeColor(HexColor("#4B6B85"))
                    canvas.rect(x, top - row_h * (row_i + 1), widths[cell_i], row_h)
                    canvas.setFont("Helvetica-Bold" if row_i == 0 else "Helvetica", 8)
                    canvas.drawString(x + 3, top - row_h * row_i - 13, cell)
                    x += widths[cell_i]
            y -= row_h * len(rows) + 16
        elif kind == "figure":
            canvas.setStrokeColor(HexColor("#387B89"))
            canvas.rect(col_x + 36, y - 120, col_width - 72, 100)
            for offset, label in enumerate(["Referral", "Navigator", "Education", "Follow-up"]):
                x = col_x + 50 + offset * ((col_width - 150) / 3)
                canvas.circle(x, y - 70, 18)
                canvas.setFont("Helvetica", 8)
                canvas.drawCentredString(x, y - 74, label)
                if offset < 3:
                    canvas.line(x + 20, y - 70, x + 55, y - 70)
            y -= 138
        elif kind == "list":
            for item in text.split("; "):
                y = _wrap(canvas, f"• {item}", col_x + 8, y, col_width - 8) - 2
            y -= 6
        else:
            if kind in {"caption", "footnote", "reference"}:
                canvas.setFont("Helvetica-Oblique", 8)
                y = _wrap(canvas, text, col_x, y, col_width, 8) - 5
            else:
                y = _wrap(canvas, text, col_x, y, col_width, 10) - 7
        if y < 70:
            canvas.showPage()
            y = PAGE_H - 54
    canvas.save()


def _rasterize_pdf(source: Path, scanned: Path) -> None:
    source_pdf = fitz.open(source)
    output_pdf = fitz.open()
    for page in source_pdf:
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        target = output_pdf.new_page(width=page.rect.width, height=page.rect.height)
        target.insert_image(target.rect, stream=pixmap.tobytes("png"))
    scanned.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.save(scanned)


def generate_fixtures(data_dir: Path) -> list[Path]:
    source_dir, scan_dir, gt_dir = data_dir / "source", data_dir / "scanned", data_dir / "ground_truth"
    written: list[Path] = []
    for fixture in FIXTURES:
        source = source_dir / f"{fixture.name}.pdf"
        scanned = scan_dir / f"{fixture.name}.pdf"
        source.parent.mkdir(parents=True, exist_ok=True)
        _draw_fixture(source, fixture)
        _rasterize_pdf(source, scanned)
        blocks: list[Block] = []
        ancestors: list[str] = []
        for i, (kind, text, level) in enumerate(fixture.blocks):
            if kind == "heading" and level:
                ancestors = ancestors[: level - 1]
                ancestors.append(text)
            blocks.append(Block(f"{fixture.name}-{i:03d}", text, kind, 1, level, list(ancestors)))
        valid_boundaries = [block.id for block in blocks[:-1] if block.kind in {"heading", "paragraph", "table", "list", "caption", "reference"}]
        write_json(gt_dir / f"{fixture.name}.json", {"document": fixture.name, "blocks": [b.to_dict() for b in blocks], "valid_boundary_after": valid_boundaries})
        written.append(scanned)
    return written
