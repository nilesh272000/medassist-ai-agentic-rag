import json
from typing import TypedDict
from openai import OpenAI
from langgraph.graph import StateGraph,END
from .config import settings
from .knowledge import kb
llm=OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
class State(TypedDict,total=False):
    question:str; query:str; candidates:list; relevant:list; evidence:str; answer:str; sources:list; retries:int; intent:str
def ask(prompt,system):
    if not llm: raise RuntimeError("OPENAI_API_KEY is not configured")
    r=llm.chat.completions.create(model=settings.openai_model,temperature=.1,messages=[{"role":"system","content":system},{"role":"user","content":prompt}])
    return (r.choices[0].message.content or "").strip()
def route(s):
    s["intent"]="chat" if s["question"].lower().strip() in {"hi","hello","hey","thanks"} else "medical"
    return s
def retrieve(s):
    s["candidates"]=kb.search(s.get("query") or s["question"]); return s
def grade(s):
    ctx="\n\n".join(f"SOURCE={d['source']}\n{d['text']}" for d in s["candidates"])
    raw=ask(f"Question: {s['question']}\nEvidence:\n{ctx}\nReturn JSON only: {{"verdict":"GOOD|WEAK","sources":[]}}","Strict evidence grader. GOOD only when evidence directly supports the question.")
    try:
        x=json.loads(raw); names={str(v).lower() for v in x.get("sources",[])}; good=str(x.get("verdict","WEAK")).upper()=="GOOD"
    except Exception:
        names=set(); good="GOOD" in raw.upper()
    s["relevant"]=[d for d in s["candidates"] if not names or d["source"].lower() in names] if good else []
    s["evidence"]="GOOD" if s["relevant"] else "WEAK"
    return s
def rewrite(s):
    s["query"]=ask("Rewrite this query for semantic retrieval. Return only the rewritten query:\n"+s["question"],"Rewrite without changing meaning or adding facts.")
    s["retries"]=s.get("retries",0)+1; return s
def answer(s):
    if s["intent"]=="chat":
        s["answer"]=ask(s["question"],"Be polite. Do not diagnose."); s["sources"]=[]; return s
    if not s.get("relevant"): return fallback(s)
    ctx="\n\n".join(f"[{d['source']}] {d['text']}" for d in s["relevant"])
    s["answer"]=ask(f"Question: {s['question']}\nEvidence:\n{ctx}\nAnswer only from evidence and cite [source].","Safety-conscious medical information assistant. Never diagnose or prescribe.")
    s["sources"]=sorted({d["source"] for d in s["relevant"]}); return s
def fallback(s):
    s["answer"]="I could not find sufficient evidence in the configured knowledge base to answer safely. Please consult a qualified healthcare professional."; s["sources"]=[]; return s
def build():
    g=StateGraph(State); [g.add_node(n,f) for n,f in [("route",route),("retrieve",retrieve),("grade",grade),("rewrite",rewrite),("answer",answer),("fallback",fallback)]]
    g.set_entry_point("route"); g.add_conditional_edges("route",lambda s:"answer" if s["intent"]=="chat" else "retrieve",{"answer":"answer","retrieve":"retrieve"})
    g.add_edge("retrieve","grade"); g.add_conditional_edges("grade",lambda s:"answer" if s["evidence"]=="GOOD" else "rewrite" if s.get("retries",0)<settings.max_retries else "fallback",{"answer":"answer","rewrite":"rewrite","fallback":"fallback"})
    g.add_edge("rewrite","retrieve"); g.add_edge("answer",END); g.add_edge("fallback",END); return g.compile()
graph=build()
