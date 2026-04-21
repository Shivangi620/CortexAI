import os
import sys

# Insert the backend directory into sys.path so tests can import from backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# 1. Setup testing engine
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 2. Override database module BEFORE importing app
import infra.database as database
database.engine = engine
database.SessionLocal = TestingSessionLocal

# 3. Now import app and other dependencies
from main import app
from infra.database import Base, get_db, db_session

from fastapi.testclient import TestClient

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    print(f"\nDEBUG: Base.metadata.tables: {list(Base.metadata.tables.keys())}")
    Base.metadata.create_all(bind=engine)
    print(f"DEBUG: Tables created in engine: {engine}")
    yield
    Base.metadata.drop_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()
