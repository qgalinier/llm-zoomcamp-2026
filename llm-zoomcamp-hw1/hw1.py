import os
import sys
import json
import re
from dotenv import load_dotenv
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from rag_helper import RAGBase
from groq import Groq

# ToyAIKit moderno
from toyaikit.tools import Tools
from toyaikit.llm import OpenAIChatCompletionsClient

load_dotenv()

# ---------- Configuración para reducir tokens en prompts ----------
REDUCED_K = 1
SNIPPET_HEAD = 400
SNIPPET_TAIL = 150
CHUNK_SIZE = 1000
CHUNK_STEP = 500
MAX_TOOL_OUTPUT_CHARS = 800
MAX_RAG_STEPS = 1

# ---------- 1) Cargar documentos (Q1) ----------
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []
for f in files:
    doc = f.parse()
    documents.append(doc)

print("Q1 - número de lesson pages:", len(documents))

# ---------- 2) Index + búsqueda (Q2) ----------
index_docs = [
    {
        "id": i,
        "filename": d["filename"],
        "content": d["content"],
    }
    for i, d in enumerate(documents)
]

index = Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)

index.fit(index_docs)

query_q2 = "How does the agentic loop keep calling the model until it stops?"
results_q2 = index.search(query_q2, num_results=5)

print("\nQ2 - primeros resultados:")
for r in results_q2:
    print(r["filename"])

# ---------- 3) RAG sobre index (Q3) ----------
class MyRAG(RAGBase):
    def __init__(self, index, client, model, reduced_k: int = REDUCED_K):
        super().__init__(index=index, client=client, model=model)
        self.index = index
        self.reduced_k = reduced_k

    def search(self, query: str, k: int = None):
        if k is None:
            k = self.reduced_k
        return self.index.search(query, num_results=k)

    def _make_snippet(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if len(text) <= SNIPPET_HEAD + SNIPPET_TAIL + 50:
            return text
        head = text[:SNIPPET_HEAD].rstrip()
        tail = text[-SNIPPET_TAIL:].lstrip()
        return f"{head}\n\n... (truncated) ...\n\n{tail}"

    def build_context(self, query_or_results, k: int = None):
        if k is None:
            k = self.reduced_k

        if isinstance(query_or_results, list):
            results = query_or_results[:k]
        else:
            results = self.search(query_or_results, k=k)

        parts = []
        for r in results:
            filename = r.get("filename", r.get("id", "unknown"))
            content = r.get("content", "")
            snippet = self._make_snippet(content)
            parts.append(f"Filename: {filename}\nContent:\n{snippet}\n")
        return "\n\n---\n\n".join(parts)

# Cliente GROQ
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

rag = MyRAG(index=index, client=client, model=model)

def extract_text_from_groq_response(resp):
    try:
        choice = None
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            choice = resp.choices[0]
        elif isinstance(resp, dict) and "choices" in resp and len(resp["choices"]) > 0:
            choice = resp["choices"][0]

        if choice is not None:
            msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            if msg is not None:
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                texts.append(item["text"])
                            elif "content" in item:
                                c = item["content"]
                                if isinstance(c, str):
                                    texts.append(c)
                                elif isinstance(c, list):
                                    for sub in c:
                                        if isinstance(sub, dict) and "text" in sub:
                                            texts.append(sub["text"])
                        elif isinstance(item, str):
                            texts.append(item)
                        else:
                            t = getattr(item, "text", None)
                            if t:
                                texts.append(t)
                    if texts:
                        return "\n".join(texts)
                elif isinstance(content, str):
                    return content
            if hasattr(choice, "text"):
                return choice.text
        if hasattr(resp, "output_text"):
            return resp.output_text
        if isinstance(resp, dict) and "output_text" in resp:
            return resp["output_text"]
    except Exception:
        pass
    return str(resp)

def run_rag_query(rag_obj: MyRAG, query: str):
    context = rag_obj.build_context(query, k=REDUCED_K)
    system_prompt = (
        "You are a concise course teaching assistant. Use the provided context to answer the question. "
        "Be brief and cite filenames when relevant."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer concisely:"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(model=rag_obj.model, messages=messages)
    text = extract_text_from_groq_response(resp)
    usage = None
    try:
        usage = getattr(resp, "usage", None) or (resp.get("usage") if isinstance(resp, dict) else None)
    except Exception:
        usage = None

    return text, usage

try:
    answer_q3, usage_q3 = run_rag_query(rag, query_q2)
except Exception as e:
    print("\nError while running RAG (Q3):")
    print(str(e))
    sys.exit(1)

print("\nQ3 - respuesta RAG:")
print(answer_q3)
print("\nQ3 - input tokens:", getattr(usage_q3, "prompt_tokens", "N/A"))

# ---------- 4) Chunking (Q4) ----------
chunks = chunk_documents(documents, size=CHUNK_SIZE, step=CHUNK_STEP)
print("\nQ4 - número de chunks:", len(chunks))

chunk_docs = [
    {
        "id": i,
        "filename": c["filename"],
        "content": c["content"],
        "start": c["start"],
    }
    for i, c in enumerate(chunks)
]

chunk_index = Index(
    text_fields=["content"],
    keyword_fields=["filename"],
)

chunk_index.fit(chunk_docs)

# ---------- 5) RAG con chunks (Q5) ----------
rag_chunks = MyRAG(index=chunk_index, client=client, model=model)

try:
    answer_q5, usage_q5 = run_rag_query(rag_chunks, query_q2)
except Exception as e:
    print("\nError while running RAG (Q5):")
    print(str(e))
    sys.exit(1)

print("\nQ5 - respuesta RAG (chunked):")
print(answer_q5)
print("\nQ5 - input tokens:", getattr(usage_q5, "prompt_tokens", "N/A"))

try:
    diff = usage_q3.prompt_tokens / usage_q5.prompt_tokens
except Exception:
    diff = "N/A"
print("Q5 - ratio Q3/Q5 (cuántas veces menos tokens):", diff)

# ---------- 6) Agentic RAG con toyaikit (Q6) ----------
search_count = 0

def search_tool(query: str):
    """
    Search the course lessons using the chunk index.
    """
    global search_count
    search_count += 1
    return chunk_index.search(query, num_results=REDUCED_K)

tools = Tools()
tools.add_tool(search_tool)  # registramos la función, nombre por defecto será 'search_tool'

llm_client = OpenAIChatCompletionsClient(
    model=model,
    client=client
)

def _truncate_tool_output(output: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if not isinstance(output, str):
        output = str(output)
    if len(output) <= max_chars:
        return output
    head = output[: max_chars // 2].rstrip()
    tail = output[- (max_chars // 2) :].lstrip()
    return f"{head}\n\n... (truncated) ...\n\n{tail}"

def _extract_query_from_text(text: str):
    """
    Busca patrones en texto libre que indiquen una búsqueda:
    - SEARCH: "..."
    - search for "..."
    - find "..."
    Devuelve el primer match o None.
    """
    if not isinstance(text, str):
        return None
    patterns = [
        r'SEARCH:\s*["“]?([^"\n“”]+)["”]?',
        r'search for\s+["“]?([^"\n“”]+)["”]?',
        r'find\s+["“]?([^"\n“”]+)["”]?',
        r'look up\s+["“]?([^"\n“”]+)["”]?',
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def agentic_rag(question: str, max_steps: int = 8):
    """
    Agent loop robusto que:
    - obliga al modelo a emitir SEARCH: \"...\" cuando quiera buscar,
    - ejecuta search_tool(...) directamente,
    - reinyecta resultados truncados,
    - y termina cuando el modelo devuelve una respuesta final.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise course teaching assistant. When you need to look up facts "
                "in the course materials, DO NOT rely on function-calling. Instead, output a single "
                "line starting with SEARCH: followed by the exact query in quotes. Example:\n\n"
                "SEARCH: \"agentic loop vs RAG\"\n\n"
                "After the tool results are provided, continue reasoning and either request another "
                "SEARCH or provide a final concise answer. Keep answers short and cite filenames when relevant."
            )
        },
        {"role": "user", "content": question}
    ]

    tool_calls_log = []
    steps = 0

    while True:
        if steps >= max_steps:
            return "Stopped after max steps without a final answer.", tool_calls_log

        response = llm_client.send_request(messages, tools=tools)
        try:
            msg = response.choices[0].message
        except Exception:
            msg = getattr(response, "message", None)

        # 0) Intentar extraer SEARCH: directamente del texto del assistant (prioritario)
        assistant_text = None
        if hasattr(msg, "content"):
            assistant_text = msg.content
        elif isinstance(msg, dict):
            assistant_text = msg.get("content")

        q_from_text = _extract_query_from_text(assistant_text or "")
        executed_any = False

        if q_from_text:
            # Ejecutar búsqueda directa
            try:
                results = search_tool(q_from_text)
                executed_any = True
                tool_calls_log.append({"source": "SEARCH_in_text", "query": q_from_text, "results_count": len(results) if isinstance(results, list) else "?"})
            except Exception as e:
                tool_calls_log.append({"source": "SEARCH_in_text", "query": q_from_text, "error": str(e)})
                results = []

            # Normalizar y truncar la salida antes de reinyectar
            if isinstance(results, (list, dict)):
                try:
                    results_str = json.dumps(results)
                except Exception:
                    results_str = str(results)
            else:
                results_str = str(results)

            truncated = _truncate_tool_output(results_str, MAX_TOOL_OUTPUT_CHARS)

            messages.append({
                "role": "tool",
                "tool_call_id": None,
                "content": truncated
            })

            # continuar al siguiente paso del loop para que el modelo procese los resultados
            steps += 1
            continue

        # 1) Si SDK provee tool_calls, intentar extraer query y ejecutar directamente
        tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else None)
        if tool_calls:
            for raw_call in tool_calls:
                # intentar extraer query desde raw_call (arguments, function.parameters, etc.)
                q = None
                # argumentos directos
                args = None
                if isinstance(raw_call, dict):
                    args = raw_call.get("arguments") or raw_call.get("params") or raw_call.get("args")
                else:
                    args = getattr(raw_call, "arguments", None) or getattr(raw_call, "params", None) or getattr(raw_call, "args", None)

                if args:
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                        except Exception:
                            parsed = args
                    else:
                        parsed = args
                    if isinstance(parsed, dict):
                        for key in ("query", "q", "text", "prompt"):
                            if key in parsed and isinstance(parsed[key], str):
                                q = parsed[key].strip()
                                break
                        if q is None:
                            # fallback: take first string value
                            for v in parsed.values():
                                if isinstance(v, str):
                                    q = v.strip()
                                    break
                    elif isinstance(parsed, str):
                        q = parsed.strip()

                # si no hay query en arguments, intentar function.parameters
                if not q:
                    func = None
                    if isinstance(raw_call, dict):
                        func = raw_call.get("function")
                    else:
                        func = getattr(raw_call, "function", None)
                    if isinstance(func, dict):
                        params = func.get("parameters") or func.get("args") or func.get("params")
                        if isinstance(params, dict):
                            for key in ("query", "q", "text", "prompt"):
                                if key in params and isinstance(params[key], str):
                                    q = params[key].strip()
                                    break
                    elif func is not None:
                        params = getattr(func, "parameters", None) or getattr(func, "args", None) or getattr(func, "params", None)
                        if isinstance(params, dict):
                            for key in ("query", "q", "text", "prompt"):
                                if key in params and isinstance(params[key], str):
                                    q = params[key].strip()
                                    break

                # si aún no hay query, intentar extraer del texto del assistant
                if not q and assistant_text:
                    q = _extract_query_from_text(assistant_text)

                if not q:
                    tool_calls_log.append({"raw_call": raw_call, "query": None, "error": "no query found"})
                    continue

                # Ejecutar la búsqueda directamente
                try:
                    results = search_tool(q)
                    executed_any = True
                    tool_calls_log.append({"raw_call": raw_call, "query": q, "results_count": len(results) if isinstance(results, list) else "?"})
                except Exception as e:
                    tool_calls_log.append({"raw_call": raw_call, "query": q, "error": str(e)})
                    results = []

                # Normalizar y truncar la salida antes de reinyectar
                if isinstance(results, (list, dict)):
                    try:
                        results_str = json.dumps(results)
                    except Exception:
                        results_str = str(results)
                else:
                    results_str = str(results)

                truncated = _truncate_tool_output(results_str, MAX_TOOL_OUTPUT_CHARS)

                messages.append({
                    "role": "tool",
                    "tool_call_id": getattr(raw_call, "id", None) or getattr(raw_call, "call_id", None) or None,
                    "content": truncated
                })

            # si ejecutamos alguna búsqueda, volvemos a iterar para que el modelo procese resultados
            if executed_any:
                steps += 1
                continue

        # 2) Si no ejecutamos ninguna búsqueda, entonces el modelo devolvió una respuesta final
        content = None
        if hasattr(msg, "content"):
            content = msg.content
        elif isinstance(msg, dict) and "content" in msg:
            content = msg["content"]
        elif hasattr(response, "output_text"):
            content = response.output_text
        else:
            content = str(response)

        if isinstance(content, str) and len(content) > 4000:
            content = content[:4000] + "\n\n... (truncated final answer) ..."

        return content, tool_calls_log

question_q6 = "How does the agentic loop work, and how is it different from plain RAG?"

try:
    answer_q6, tool_calls_q6 = agentic_rag(question_q6)
except Exception as e:
    print("\nError while running agentic RAG (Q6):")
    print(str(e))
    print("\nSi el error es por tokens, reduce REDUCED_K, SNIPPET_HEAD o MAX_TOOL_OUTPUT_CHARS.")
    sys.exit(1)

print("\nQ6 - agent answer:")
print(answer_q6)

print("\nQ6 - tool calls:")
for call in tool_calls_q6:
    # call may be dict entries in the log
    if isinstance(call, dict):
        raw = call.get("raw_call", None)
        q = call.get("query", None)
        rc = call.get("results_count", None)
        err = call.get("error", None)
        print("raw_call:", type(raw).__name__, "query:", q, "results_count:", rc, "error:", err)
    else:
        name = getattr(call, "name", None) or str(call)
        args = getattr(call, "arguments", None) or ""
        print(name, args)

print("\nQ6 - número de veces que llamó search:", search_count)
