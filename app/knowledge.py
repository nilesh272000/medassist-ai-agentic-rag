from pathlib import Path
import chromadb
from openai import OpenAI
from .config import settings
class MedicalKB:
    def __init__(self):
        Path(settings.chroma_path).mkdir(parents=True,exist_ok=True)
        self.client=chromadb.PersistentClient(path=settings.chroma_path)
        self.collection=self.client.get_or_create_collection("medical_kb")
        self.openai=OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
    def _embed(self,text):
        if not self.openai: raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.openai.embeddings.create(model=settings.embedding_model,input=text).data[0].embedding
    @staticmethod
    def chunk(text,size=900,overlap=100):
        text=" ".join(text.split()); out=[]; start=0
        while start<len(text):
            end=min(start+size,len(text)); out.append(text[start:end])
            if end>=len(text): break
            start=max(end-overlap,start+1)
        return out
    def ingest(self):
        ids=[]; docs=[]; metas=[]; vecs=[]
        for p in sorted(Path(settings.kb_path).glob("*.txt")):
            for i,c in enumerate(self.chunk(p.read_text(encoding="utf-8"))):
                ids.append(f"{p.stem}-{i}"); docs.append(c); metas.append({"source":p.stem,"file":p.name}); vecs.append(self._embed(c))
        if docs:self.collection.upsert(ids=ids,documents=docs,metadatas=metas,embeddings=vecs)
        return len(docs)
    def search(self,q):
        if self.collection.count()==0: raise RuntimeError("KB is empty. Run python ingest.py")
        r=self.collection.query(query_embeddings=[self._embed(q)],n_results=settings.top_k,include=["documents","metadatas","distances"])
        return [{"text":d,"source":r["metadatas"][0][i].get("source","unknown"),"distance":r["distances"][0][i]} for i,d in enumerate(r["documents"][0])]
kb=MedicalKB()
