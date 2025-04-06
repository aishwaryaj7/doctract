import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

if not st.secrets:
    load_dotenv()

CWD = Path().cwd()
DATA_DIR = CWD.joinpath("data")

# POSTGRES_HOST = os.getenv("POSTGRES_HOST") or st.secrets.get("POSTGRES_HOST")
# POSTGRES_PORT = os.getenv("POSTGRES_PORT") or st.secrets.get("POSTGRES_PORT")
# POSTGRES_USER = os.getenv("POSTGRES_USER") or st.secrets.get("POSTGRES_USER")
# POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD") or st.secrets.get("POSTGRES_PASSWORD")
# TARGET_DATABASE_NAME = os.getenv("VECTOR_DB_NAME") or st.secrets.get("VECTOR_DB_NAME")
# TARGET_TABLE_NAME = os.getenv("VECTOR_TABLE_NAME") or st.secrets.get("VECTOR_TABLE_NAME")
# EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION")) or st.secrets.get("EMBEDDING_DIMENSION")