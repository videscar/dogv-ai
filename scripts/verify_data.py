"""
Verify DOGV pipeline:
- issues vs. weekends/holidays
- documents counts
- PDFs downloaded
- text extracted
"""

from datetime import date
from pathlib import Path

try:
    from . import _path  # type: ignore  # ensures project root is on sys.path
except ImportError:
    import _path  # type: ignore  # noqa: F401

from api.db import SessionLocal
from api.models import DogvIssue, DogvDocument
from scripts.download_pdfs import build_dest_path


PDF_BASE = Path("data/pdf_cache")

# If you stored PDFs in a different layout, adjust this function.
def pdf_path_for(doc: DogvDocument) -> Path:
    if not doc.pdf_url:
        return None
    return build_dest_path(doc.pdf_url)


def main():
    db = SessionLocal()
    try:
        issues = db.query(DogvIssue).order_by(DogvIssue.date).all()
        docs = db.query(DogvDocument).all()

        print("=== COUNTS ===")
        print(f"Issues total:    {len(issues)}")
        print(f"Documents total: {len(docs)}")

        # --- issues & weekends ---
        print("\n=== ISSUES WITH numero IS NULL ===")
        null_issues = [i for i in issues if i.numero is None]
        for i in null_issues:
            weekday = i.date.weekday()  # 0=Mon ... 6=Sun
            weekday_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][weekday]
            print(f"{i.date} ({weekday_name}) -> numero=None")

        # Optional: flag weekday non-holiday issues with numero=None as suspicious
        KNOWN_HOLIDAYS = {
            date(2024, 1, 1),
            # add more if you want
        }
        suspicious = [
            i for i in null_issues
            if i.date.weekday() < 5 and i.date not in KNOWN_HOLIDAYS
        ]
        if suspicious:
            print("\n[WARN] Weekday issues with numero=None (check if more holidays):")
            for i in suspicious:
                print(f"  - {i.date}")
        else:
            print("\nNo suspicious weekday issues with numero=None (only weekends/known holidays).")

        # --- PDFs & text ---
        print("\n=== PDF & TEXT COVERAGE ===")
        docs_with_pdf = [d for d in docs if d.pdf_url]
        print(f"Documents with pdf_url: {len(docs_with_pdf)}")

        missing_files = []
        for d in docs_with_pdf:
            path = pdf_path_for(d)
            if path is None or not path.is_file():
                missing_files.append((d.id, d.pdf_url))
        print(f"PDF files missing on disk: {len(missing_files)}")

        docs_with_text = [d for d in docs_with_pdf if getattr(d, "text", None)]
        print(f"Documents with pdf_url AND non-empty text: {len(docs_with_text)}")

        if missing_files:
            print("\nExamples of missing PDF files (up to 5):")
            for did, url in missing_files[:5]:
                print(f"  - doc_id={did}, pdf_url={url}")

        # --- sample text ---
        print("\n=== SAMPLE DOCUMENT TEXT ===")
        sample = docs_with_text[0] if docs_with_text else None
        if not sample:
            print("No document with text found – check your extract_text script.")
        else:
            print(f"Doc id={sample.id}, issue_date={sample.issue.date}, conselleria={sample.conselleria}")
            print(f"Title: {sample.title}")
            print("\nFirst 600 chars of text:\n")
            print(sample.text[:600])
    finally:
        db.close()


if __name__ == "__main__":
    main()
