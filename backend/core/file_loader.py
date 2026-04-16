import io
import csv
import os
import sqlite3
import pandas as pd


# CSV field size
csv.field_size_limit(int(1e9))


SUPPORTED_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".dat", ".tab", ".log",
    ".xlsx", ".xls", ".xlsm", ".ods",
    ".json", ".jsonl", ".ndjson",
    ".parquet", ".feather", ".arrow", ".orc",
    ".dta", ".sas7bdat", ".sav", ".xpt",
    ".xml",
    ".html", ".htm",
    ".db", ".sqlite", ".sqlite3",
    ".pkl", ".pickle",
    ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
    ".md", ".markdown", ".rtf",
}

EXTENSION_LABELS = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]


def _read_text_payload(target, filepath: str = None) -> str:
    if filepath:
        with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    if hasattr(target, "seek"):
        target.seek(0)
    raw = target.read() if hasattr(target, "read") else target
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _dataframe_from_text_document(text: str, source_name: str, segment_label: str = "line") -> pd.DataFrame:
    cleaned = (text or "").replace("\x00", "").strip()
    if not cleaned:
        raise ValueError(f"No readable text found in {source_name}.")

    blocks = [block.strip() for block in cleaned.splitlines() if block.strip()]
    if not blocks:
        blocks = [cleaned]

    return pd.DataFrame(
        {
            "source_file": source_name,
            "segment_type": segment_label,
            "segment_index": range(1, len(blocks) + 1),
            "text": blocks,
            "text_length": [len(block) for block in blocks],
        }
    )


def _load_pdf_as_dataframe(
    target,
    filepath: str = None,
    source_name: str = "document.pdf",
    pdf_mode: str = "text",
) -> pd.DataFrame:
    pdf_mode = (pdf_mode or "text").lower().strip()

    if pdf_mode == "tables":
        try:
            import pdfplumber
        except ImportError as exc:
            raise ValueError("PDF table extraction requires pdfplumber to be installed.") from exc

        table_rows = []
        with pdfplumber.open(filepath or target) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    extracted_tables = page.extract_tables() or []
                except Exception:
                    extracted_tables = []

                for table_idx, table in enumerate(extracted_tables, start=1):
                    if not table:
                        continue
                    normalized = pd.DataFrame(table)
                    normalized = normalized.dropna(how="all").dropna(axis=1, how="all")
                    if normalized.empty:
                        continue
                    normalized.insert(0, "table_index", table_idx)
                    normalized.insert(0, "page", page_idx)
                    normalized.insert(0, "source_file", source_name)
                    table_rows.append(normalized)

        if table_rows:
            return pd.concat(table_rows, ignore_index=True)
        raise ValueError("No tables detected in PDF. Try plain-text mode instead.")

    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise ValueError("PDF upload requires PyPDF2 to be installed.") from exc

    reader = PdfReader(filepath or target)
    rows = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        rows.append(
            {
                "source_file": source_name,
                "page": idx,
                "text": text,
                "text_length": len(text),
            }
        )

    if not rows:
        raise ValueError("No pages found in PDF.")

    if not any(row["text"] for row in rows):
        raise ValueError("PDF was loaded but no extractable text was found.")

    return pd.DataFrame(rows)


def _load_image_as_dataframe(target, filepath: str = None, source_name: str = "image") -> pd.DataFrame:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Image upload requires Pillow to be installed.") from exc

    image = Image.open(filepath or target)
    width, height = image.size
    payload = {
        "source_file": source_name,
        "format": image.format,
        "mode": image.mode,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 6) if height else None,
    }

    extracted_text = ""
    try:
        import pytesseract

        extracted_text = (pytesseract.image_to_string(image) or "").strip()
    except Exception:
        extracted_text = ""

    payload["ocr_text"] = extracted_text
    payload["ocr_text_length"] = len(extracted_text)
    return pd.DataFrame([payload])


def _sniff_delimiter(raw_bytes: bytes) -> str:
    # ✅ FIX 1: handle None / empty bytes
    if not raw_bytes:
        return ","

    try:
        sample = raw_bytes[:4096].decode("utf-8", errors="replace")
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;: ")
        return dialect.delimiter
    except Exception:
        return ","


def load_dataframe(
    contents: bytes = None,
    filename: str = None,
    filepath: str = None,
    pdf_mode: str = "text",
) -> pd.DataFrame:

    # ✅ FIX 2: validate inputs
    if not filepath and contents is None:
        raise ValueError("Either contents or filepath must be provided.")

    if filepath:
        if not os.path.exists(filepath):
            raise ValueError(f"File path does not exist: {filepath}")  # ✅ FIX 3
        target = filepath
        fname = os.path.basename(filepath).lower().strip()
    else:
        target = io.BytesIO(contents)
        fname = filename.lower().strip() if filename else "data.csv"

    ext = "." + fname.rsplit(".", 1)[-1] if "." in fname else ""

    # ── Delimited text ────────────────────────────────────────────────────────
    if ext in {".csv", ".tsv", ".txt", ".dat", ".tab", ".log", ""}:

        if ext in {".tsv", ".tab"}:
            return pd.read_csv(target, sep="\t", encoding_errors="replace")

        # sample for sniff
        try:
            if filepath:
                with open(filepath, "rb") as f:
                    sample_bytes = f.read(8192)
            else:
                sample_bytes = contents
        except Exception:
            sample_bytes = None  # ✅ FIX 4

        delim = _sniff_delimiter(sample_bytes)

        try:
            return pd.read_csv(target, sep=delim, encoding_errors="replace", engine="c")
        except Exception:
            if hasattr(target, "seek"):
                target.seek(0)
            return pd.read_csv(target, sep=delim, encoding_errors="replace", engine="python")

    if ext in {".md", ".markdown", ".rtf"}:
        text = _read_text_payload(target, filepath=filepath)
        return _dataframe_from_text_document(text, fname, segment_label="block")

    # ── Excel ───────────────────────────────────────────────────────────
    if ext in {".xlsx", ".xlsm"}:
        return pd.read_excel(target, engine="openpyxl")
    if ext == ".xls":
        return pd.read_excel(target, engine="xlrd")
    if ext == ".ods":
        return pd.read_excel(target, engine="odf")

    # ── JSON ───────────────────────────────────────────────────────────
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(target, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(target)
        except Exception:
            if hasattr(target, "seek"):
                target.seek(0)
            return pd.read_json(target, lines=True)

    # ── Binary ─────────────────────────────────────────────────────────
    if ext == ".parquet":
        return pd.read_parquet(target)
    if ext in {".feather", ".arrow"}:
        return pd.read_feather(target)
    if ext == ".orc":
        return pd.read_orc(target)

    # ── Stats ─────────────────────────────────────────────────────────
    if ext == ".dta":
        return pd.read_stata(target)
    if ext in {".sas7bdat", ".xpt"}:
        return pd.read_sas(target, format="sas7bdat" if ext == ".sas7bdat" else "xport")

    if ext == ".sav":
        try:
            import pyreadstat
            if filepath:
                df, _ = pyreadstat.read_sav(filepath)
            else:
                df, _ = pyreadstat.read_sav(target)
            return df
        except ImportError:
            raise ValueError("SPSS (.sav) requires pyreadstat")

    # ── XML ───────────────────────────────────────────────────────────
    if ext == ".xml":
        try:
            return pd.read_xml(target)
        except Exception as e:
            raise ValueError(f"Could not parse XML: {e}")

    # ── HTML ──────────────────────────────────────────────────────────
    if ext in {".html", ".htm"}:
        tables = pd.read_html(target)
        if not tables:
            raise ValueError("No tables found in HTML")
        return max(tables, key=len)

    if ext == ".pdf":
        return _load_pdf_as_dataframe(target, filepath=filepath, source_name=fname, pdf_mode=pdf_mode)

    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}:
        return _load_image_as_dataframe(target, filepath=filepath, source_name=fname)

    # ── SQLite ────────────────────────────────────────────────────────
    if ext in {".db", ".sqlite", ".sqlite3"}:
        db_path = filepath
        temp_used = False

        if not db_path:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(contents)
                db_path = tmp.name
                temp_used = True

        try:
            conn = sqlite3.connect(db_path)
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()

            if not tables:
                raise ValueError("No tables found in SQLite database.")

            table_name = tables[0][0]
            df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
            conn.close()
            return df

        finally:
            if temp_used and os.path.exists(db_path):  # ✅ FIX 5
                os.unlink(db_path)

    # ── Pickle ────────────────────────────────────────────────────────
    if ext in {".pkl", ".pickle"}:
        try:
            obj = pd.read_pickle(target)
        except Exception as e:
            raise ValueError(
                "Pickle could not be read as a pandas DataFrame. "
                "If this came from an AutoML export bundle, the archive may be pointing at a model artifact instead of the dataset. "
                f"Original error: {e}"
            ) from e
        if isinstance(obj, pd.DataFrame):
            return obj
        raise ValueError(
            f"Pickle contains {type(obj).__name__}, not a pandas DataFrame."
        )

    # ── Fallback ──────────────────────────────────────────────────────
    try:
        if filepath:
            with open(filepath, "rb") as f:
                sample_bytes = f.read(8192)
        else:
            sample_bytes = contents

        delim = _sniff_delimiter(sample_bytes)

        return pd.read_csv(
            target,
            sep=delim,
            encoding_errors="replace",
            engine="python"
        )

    except Exception as e:
        raise ValueError(
            f"Unsupported or unreadable file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. Error: {e}"
        )
