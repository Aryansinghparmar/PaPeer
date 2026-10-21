"""Test-wide setup.

Backend modules build API clients at import time, so dummy credentials are set
before anything is imported. ``load_dotenv`` never overrides variables that are
already set, which means the real ``.env`` keys cannot leak into a test run and
no test can make a paid call by accident.
"""

import os

for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "QDRANT_API_KEY", "LANGSMITH_API_KEY"):
    os.environ[name] = "test-dummy-key"
os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LOCAL_METRICS_ENABLED"] = "false"
