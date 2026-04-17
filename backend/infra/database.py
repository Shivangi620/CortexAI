from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text as sql_text
from contextlib import contextmanager
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./automl_studio.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

Base = declarative_base()


@contextmanager
def get_db():
    """Context manager for safe, exception-aware DB session lifecycle."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    file_path = Column(String)
    profile_json = Column(Text)
    parent_dataset_id = Column(String, nullable=True)
    source_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class JobModel(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    dataset_id = Column(String)
    status = Column(String)
    history_json = Column(Text, default="[]")
    results_json = Column(Text, nullable=True)
    model_path = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    insights_json = Column(Text, nullable=True)
    reasoning_json = Column(Text, default="[]")
    story = Column(Text, nullable=True)
    params_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MetaLearningRecord(Base):
    __tablename__ = "meta_learning"

    id = Column(
        String,
        primary_key=True,
        index=True,
        default=lambda: __import__("uuid").uuid4().hex,
    )
    meta_features_json = Column(Text)
    best_model = Column(String)
    best_score = Column(String)
    task_type = Column(String)
    metric_name = Column(String)
    leaderboard_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(
        String,
        primary_key=True,
        index=True,
        default=lambda: __import__("uuid").uuid4().hex,
    )
    job_id = Column(String, index=True)
    dataset_id = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    metric_name = Column(String, nullable=True)
    score = Column(String, nullable=True)
    hyperparams_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    leaderboard_json = Column(Text, nullable=True)
    feature_count = Column(String, nullable=True)
    row_count = Column(String, nullable=True)
    task_type = Column(String, nullable=True)
    mode = Column(String, nullable=True)
    goal = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftCheck(Base):
    __tablename__ = "drift_checks"

    id = Column(
        String,
        primary_key=True,
        index=True,
        default=lambda: __import__("uuid").uuid4().hex,
    )
    job_id = Column(String, index=True)
    dataset_id = Column(String, nullable=True)
    uploaded_name = Column(String, nullable=True)
    status = Column(String, nullable=True)
    drift_score_pct = Column(String, nullable=True)
    report_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftSchedule(Base):
    __tablename__ = "drift_schedules"

    id = Column(
        String,
        primary_key=True,
        index=True,
        default=lambda: __import__("uuid").uuid4().hex,
    )
    job_id = Column(String, index=True)
    enabled = Column(String, nullable=True, default="true")
    frequency_days = Column(String, nullable=True, default="7")
    warning_threshold = Column(String, nullable=True, default="0.1")
    critical_threshold = Column(String, nullable=True, default="0.2")
    last_alert_status = Column(String, nullable=True)
    last_alert_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id = Column(
        String,
        primary_key=True,
        index=True,
        default=lambda: __import__("uuid").uuid4().hex,
    )
    run_id = Column(String, index=True)
    label = Column(String, nullable=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TeamNote(Base):
    __tablename__ = "team_notes"

    id = Column(
        String,
        primary_key=True,
        index=True,
        default=lambda: __import__("uuid").uuid4().hex,
    )
    entity_type = Column(String, nullable=True)
    entity_id = Column(String, index=True)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def _run_migrations():
    statements = [
        "ALTER TABLE jobs ADD COLUMN created_at DATETIME",
        "ALTER TABLE jobs ADD COLUMN story TEXT",
        "ALTER TABLE jobs ADD COLUMN params_json TEXT",
        "ALTER TABLE jobs ADD COLUMN insights_json TEXT",
        "ALTER TABLE jobs ADD COLUMN reasoning_json TEXT",
        "ALTER TABLE jobs ADD COLUMN model_path TEXT",
        "ALTER TABLE datasets ADD COLUMN created_at DATETIME",
        "ALTER TABLE datasets ADD COLUMN parent_dataset_id TEXT",
        "ALTER TABLE datasets ADD COLUMN source_type TEXT",
        "ALTER TABLE drift_schedules ADD COLUMN warning_threshold TEXT",
        "ALTER TABLE drift_schedules ADD COLUMN critical_threshold TEXT",
        "ALTER TABLE drift_schedules ADD COLUMN last_alert_status TEXT",
        "ALTER TABLE drift_schedules ADD COLUMN last_alert_summary TEXT",
        """CREATE TABLE IF NOT EXISTS meta_learning (
            id                 TEXT PRIMARY KEY,
            meta_features_json TEXT NOT NULL,
            best_model         TEXT,
            best_score         TEXT,
            task_type          TEXT,
            metric_name        TEXT,
            leaderboard_json   TEXT,
            created_at         DATETIME
        )""",
        """CREATE TABLE IF NOT EXISTS experiment_runs (
            id              TEXT PRIMARY KEY,
            job_id          TEXT,
            dataset_id      TEXT,
            model_name      TEXT,
            metric_name     TEXT,
            score           TEXT,
            hyperparams_json TEXT,
            metrics_json    TEXT,
            leaderboard_json TEXT,
            feature_count   TEXT,
            row_count       TEXT,
            task_type       TEXT,
            mode            TEXT,
            goal            TEXT,
            created_at      DATETIME
        )""",
        """CREATE TABLE IF NOT EXISTS drift_checks (
            id              TEXT PRIMARY KEY,
            job_id          TEXT,
            dataset_id      TEXT,
            uploaded_name   TEXT,
            status          TEXT,
            drift_score_pct TEXT,
            report_json     TEXT,
            created_at      DATETIME
        )""",
        """CREATE TABLE IF NOT EXISTS drift_schedules (
            id              TEXT PRIMARY KEY,
            job_id          TEXT,
            enabled         TEXT,
            frequency_days  TEXT,
            warning_threshold TEXT,
            critical_threshold TEXT,
            last_alert_status TEXT,
            last_alert_summary TEXT,
            created_at      DATETIME,
            updated_at      DATETIME
        )""",
        """CREATE TABLE IF NOT EXISTS model_registry (
            id              TEXT PRIMARY KEY,
            run_id          TEXT,
            label           TEXT,
            note            TEXT,
            created_at      DATETIME,
            updated_at      DATETIME
        )""",
        """CREATE TABLE IF NOT EXISTS team_notes (
            id              TEXT PRIMARY KEY,
            entity_type     TEXT,
            entity_id       TEXT,
            note            TEXT,
            created_at      DATETIME
        )""",
    ]

    with engine.connect() as conn:
        for stmt in statements:
            try:
                conn.execute(sql_text(stmt))
                conn.commit()
            except Exception:
                pass


_run_migrations()
