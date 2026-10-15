import base64
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .agent import graph
from .knowledge import kb
from .config import settings
app=FastAPI(title="MedAssist AI",version="1.0")
app.mount("/static",StaticFiles(directory="static"),name="static")
@app.get("/",response_class=HTMLResponse)
def home(): return open("templates/index.html",encoding="utf8").read()
@app.get("/api/health")
def health(): return {"status":"ok","knowledge_chunks":kb.collection.count(),"openai_configured":bool(settings.openai_api_key)}
@app.post("/api/chat")
def chat(payload:dict):
    q=str(payload.get("question","")).strip()
    if len(q)<2: raise HTTPException(422,"Question is required")
    s=graph.invoke({"question":q,"retries":0})
    return {"answer":s.get("answer",""),"sources":s.get("sources",[]),"retries":s.get("retries",0),"route":s.get("intent","unknown")}
@app.post("/api/observe-image")
async def observe_image(file:UploadFile=File(...)):
    if not settings.openai_api_key: raise HTTPException(503,"OPENAI_API_KEY is not configured")
    data=base64.b64encode(await file.read()).decode()
    from openai import OpenAI
    r=OpenAI(api_key=settings.openai_api_key).chat.completions.create(model=settings.openai_model,temperature=0,messages=[{"role":"user","content":[{"type":"text","text":"Describe this image for educational medical context. Do not diagnose."},{"type":"image_url","image_url":{"url":f"data:{file.content_type or 'image/jpeg'};base64,{data}"}}]}])
    return {"observation":r.choices[0].message.content}
