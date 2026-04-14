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
}

EXTENSION_LABELS = [ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)]


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


def load_dataframe(contents: bytes = None, filename: str = None, filepath: str = None) -> pd.DataFrame:

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
