import io
import csv
import os
import sqlite3
import xml.etree.ElementTree as ET
from email import policy
from email.parser import BytesParser
from typing import Any

import pandas as pd

from infra.config import settings

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
    ".docx", ".eml", ".msg",
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


def _chunk_text_segments(df: pd.DataFrame, chunk_size: int = 0) -> pd.DataFrame:
    if chunk_size <= 0 or "text" not in df.columns:
        return df

    chunk_rows = []
    for row in df.to_dict(orient="records"):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        for chunk_index, start in enumerate(range(0, len(text), chunk_size), start=1):
            chunk_text = text[start : start + chunk_size].strip()
            if not chunk_text:
                continue
            chunk_row = dict(row)
            chunk_row["segment_type"] = f"{row.get('segment_type', 'segment')}_chunk"
            chunk_row["chunk_index"] = chunk_index
            chunk_row["chunk_start"] = start
            chunk_row["text"] = chunk_text
            chunk_row["text_length"] = len(chunk_text)
            chunk_rows.append(chunk_row)

    if chunk_rows:
        return pd.DataFrame(chunk_rows)
    return df


def _extract_docx_text(filepath: str) -> str:
    import zipfile

    with zipfile.ZipFile(filepath) as archive:
        try:
            xml_payload = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError("DOCX file is missing word/document.xml.") from exc

    root = ET.fromstring(xml_payload)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        joined = "".join(texts).strip()
        if joined:
            paragraphs.append(joined)
    return "\n".join(paragraphs)


def _load_docx_as_dataframe(filepath: str, source_name: str, chunk_size: int = 0) -> pd.DataFrame:
    text = _extract_docx_text(filepath)
    df = _dataframe_from_text_document(text, source_name, segment_label="paragraph")
    return _chunk_text_segments(df, chunk_size=chunk_size)


def _email_part_payload(message: Any) -> list[dict[str, Any]]:
    rows = []
    plain_parts = []
    html_parts = []
    attachments = 0

    if message.is_multipart():
        for part in message.walk():
            disposition = (part.get_content_disposition() or "").lower()
            if disposition == "attachment":
                attachments += 1
                continue
            content_type = (part.get_content_type() or "").lower()
            try:
                payload = part.get_content()
            except Exception:
                payload = ""
            if content_type == "text/plain" and payload:
                plain_parts.append(str(payload).strip())
            elif content_type == "text/html" and payload:
                html_parts.append(str(payload).strip())
    else:
        try:
            payload = message.get_content()
        except Exception:
            payload = ""
        if payload:
            plain_parts.append(str(payload).strip())

    body = "\n\n".join(part for part in plain_parts if part) or "\n\n".join(
        part for part in html_parts if part
    )
    rows.append(
        {
            "subject": str(message.get("subject") or "").strip(),
            "from": str(message.get("from") or "").strip(),
            "to": str(message.get("to") or "").strip(),
            "date": str(message.get("date") or "").strip(),
            "text": body.strip(),
            "text_length": len(body.strip()),
            "attachment_count": attachments,
        }
    )
    return rows


def _load_eml_as_dataframe(filepath: str, source_name: str, chunk_size: int = 0) -> pd.DataFrame:
    with open(filepath, "rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)
    df = pd.DataFrame(_email_part_payload(message))
    df.insert(0, "source_file", source_name)
    return _chunk_text_segments(df, chunk_size=chunk_size)


def _load_msg_as_dataframe(filepath: str, source_name: str, chunk_size: int = 0) -> pd.DataFrame:
    try:
        import extract_msg
    except ImportError as exc:
        raise ValueError("MSG upload requires the optional 'extract-msg' package.") from exc

    message = extract_msg.Message(filepath)
    text = (message.body or "").strip()
    df = pd.DataFrame(
        [
            {
                "source_file": source_name,
                "subject": str(message.subject or "").strip(),
                "from": str(message.sender or "").strip(),
                "to": str(message.to or "").strip(),
                "date": str(message.date or "").strip(),
                "text": text,
                "text_length": len(text),
                "attachment_count": len(getattr(message, "attachments", []) or []),
            }
        ]
    )
    return _chunk_text_segments(df, chunk_size=chunk_size)


def _image_embedding_features(image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    resized = rgb.resize((4, 4))
    pixels = list(resized.getdata())
    channels = list(zip(*pixels))
    features: dict[str, Any] = {}

    for idx, channel_name in enumerate(("r", "g", "b")):
        values = [float(value) for value in channels[idx]]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        features[f"{channel_name}_mean"] = round(mean, 4)
        features[f"{channel_name}_std"] = round(variance ** 0.5, 4)

    grayscale = image.convert("L").resize((4, 4))
    for idx, value in enumerate(grayscale.getdata()):
        features[f"embedding_{idx:02d}"] = round(float(value) / 255.0, 6)

    return features


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
    ocr_confidence = None
    ocr_word_count = 0
    try:
        import pytesseract

        extracted_text = (pytesseract.image_to_string(image) or "").strip()
        try:
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = []
            for raw_conf in ocr_data.get("conf", []):
                try:
                    conf = float(raw_conf)
                except Exception:
                    continue
                if conf >= 0:
                    confidences.append(conf)
            words = [str(word).strip() for word in ocr_data.get("text", []) if str(word).strip()]
            ocr_word_count = len(words)
            if confidences:
                ocr_confidence = round(sum(confidences) / len(confidences), 2)
        except Exception:
            ocr_confidence = None
    except Exception:
        extracted_text = ""

    payload["ocr_text"] = extracted_text
    payload["ocr_text_length"] = len(extracted_text)
    payload["ocr_confidence"] = ocr_confidence
    payload["ocr_word_count"] = ocr_word_count
    payload.update(_image_embedding_features(image))
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
    sheet_name: str | int | None = None,
    sqlite_table: str | None = None,
    allow_pickle: bool | None = None,
    text_chunk_size: int = 0,
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
        return _chunk_text_segments(
            _dataframe_from_text_document(text, fname, segment_label="block"),
            chunk_size=text_chunk_size,
        )

    if ext == ".docx":
        if not filepath:
            raise ValueError("DOCX upload requires a file path.")
        return _load_docx_as_dataframe(filepath, fname, chunk_size=text_chunk_size)

    if ext == ".eml":
        if not filepath:
            raise ValueError("EML upload requires a file path.")
        return _load_eml_as_dataframe(filepath, fname, chunk_size=text_chunk_size)

    if ext == ".msg":
        if not filepath:
            raise ValueError("MSG upload requires a file path.")
        return _load_msg_as_dataframe(filepath, fname, chunk_size=text_chunk_size)

    # ── Excel ───────────────────────────────────────────────────────────
    if ext in {".xlsx", ".xlsm"}:
        try:
            workbook = pd.ExcelFile(target, engine="openpyxl")
            selected_sheet = sheet_name if sheet_name not in {"", None} else workbook.sheet_names[0]
            if selected_sheet not in workbook.sheet_names:
                raise ValueError(
                    f"Excel sheet '{selected_sheet}' was not found. Available sheets: {', '.join(workbook.sheet_names)}"
                )
            return workbook.parse(selected_sheet)
        finally:
            if hasattr(target, "seek"):
                target.seek(0)
    if ext == ".xls":
        try:
            workbook = pd.ExcelFile(target, engine="xlrd")
            selected_sheet = sheet_name if sheet_name not in {"", None} else workbook.sheet_names[0]
            if selected_sheet not in workbook.sheet_names:
                raise ValueError(
                    f"Excel sheet '{selected_sheet}' was not found. Available sheets: {', '.join(workbook.sheet_names)}"
                )
            return workbook.parse(selected_sheet)
        finally:
            if hasattr(target, "seek"):
                target.seek(0)
    if ext == ".ods":
        try:
            workbook = pd.ExcelFile(target, engine="odf")
            selected_sheet = sheet_name if sheet_name not in {"", None} else workbook.sheet_names[0]
            if selected_sheet not in workbook.sheet_names:
                raise ValueError(
                    f"ODS sheet '{selected_sheet}' was not found. Available sheets: {', '.join(workbook.sheet_names)}"
                )
            return workbook.parse(selected_sheet)
        finally:
            if hasattr(target, "seek"):
                target.seek(0)

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
        pdf_df = _load_pdf_as_dataframe(
            target,
            filepath=filepath,
            source_name=fname,
            pdf_mode=pdf_mode,
        )
        return _chunk_text_segments(pdf_df, chunk_size=text_chunk_size)

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

            table_names = [row[0] for row in tables]
            table_name = sqlite_table or table_names[0]
            if table_name not in table_names:
                raise ValueError(
                    f"SQLite table '{table_name}' was not found. Available tables: {', '.join(table_names)}"
                )
            df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
            conn.close()
            return df

        finally:
            if temp_used and os.path.exists(db_path):  # ✅ FIX 5
                os.unlink(db_path)

    # ── Pickle ────────────────────────────────────────────────────────
    if ext in {".pkl", ".pickle"}:
        effective_allow_pickle = (
            settings.allow_pickle_uploads if allow_pickle is None else bool(allow_pickle)
        )
        if not effective_allow_pickle:
            raise ValueError(
                "Pickle uploads are disabled by default for safety. Set ALLOW_PICKLE_UPLOADS=true to enable them."
            )
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
