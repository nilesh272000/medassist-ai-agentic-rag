from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
ROOT=Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    openai_api_key:str=""
    openai_model:str="gpt-4o-mini"
    embedding_model:str="text-embedding-3-small"
    chroma_path:str=str(ROOT/"data/chroma")
    kb_path:str=str(ROOT/"data/knowledge_base")
    top_k:int=5
    max_retries:int=1
    model_config=SettingsConfigDict(env_file=ROOT/".env",extra="ignore")
settings=Settings()
