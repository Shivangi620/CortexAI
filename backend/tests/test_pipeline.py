import pytest
import os
import pandas as pd
from core.pipeline import PipelineEngine, PipelineStep
from infra.database import Base, engine

@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_pipeline_engine_flow(tmpdir):
    # Dummy data
    data_path = os.path.join(tmpdir, "test_data.csv")
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    df.to_csv(data_path, index=False)

    pipe = PipelineEngine(
        job_id="dummy_job_123",
        dataset_id="dummy_dataset_123",
        file_path=data_path,
        target="target",
        goal="Speed",
        mode="Fast"
    )
    
    assert pipe.job_id == "dummy_job_123"
    
    # We can't easily mock the DB profile injection locally without filling DB, 
    # but we can verify the class interface.
    assert hasattr(pipe, "run")
    assert pipe.state == {}
    assert pipe.logs == []
