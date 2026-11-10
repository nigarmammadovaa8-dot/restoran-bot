# -*- coding: utf-8 -*-
"""Narxana ağıllı ofisiant botu (RAG + Tools) - deploy üçün hazır versiya.

Colab versiyasından fərqlər:
- Colab-a aid kod (drive.mount, !pip, userdata) çıxarılıb
- Açarlar keys.txt-dən yox, mühit dəyişənlərindən (Secrets) oxunur
- Fayl yolları app.py-a nisbidir
- Gradio interfeysi Streamlit veb interfeysinə çevrilib
- Şəkil generasiyası default olaraq API ilə işləyir (pulsuz CPU serverdə GPU yoxdur)
"""

import os
import re
import ast
import json
import operator as op
from collections import Counter

import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 1. Ayarlar və açarlar (Secrets / Environment Variables)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_secret(name, default=None):
    """Streamlit Secrets və environment variables-dan parametr oxuyur."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


OPENROUTER_API_KEY = get_secret("OPENROUTER_API_KEY")
HF_TOKEN = get_secret("HF_TOKEN")

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY tapılmadı. Streamlit tətbiqinin Settings → Secrets "
        "bölməsində bu açarı əlavə edin."
    )
    st.stop()

# İstəsəniz bunları da Secrets/Variables vasitəsilə dəyişə bilərsiniz
CHAT_MODEL = os.environ.get("CHAT_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
IMAGE_BACKEND = get_secret("IMAGE_BACKEND", "disabled")  # disabled / api / local
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MD_PATH = os.path.join(BASE_DIR, "narxana_restoran.md")

TOP_K = 4
MAX_PAIRS = 5
MAX_CHARS = 450

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# ---------------------------------------------------------------------------
# 2. RAG bazası
# ---------------------------------------------------------------------------
if not os.path.exists(MD_PATH):
    raise FileNotFoundError(
        "narxana_restoran.md faylı app.py ilə eyni qovluqda olmalıdır."
    )

with open(MD_PATH, encoding="utf-8") as f:
    text = f.read()


def split_text(body, max_chars):
    """Uzun mətni sətir-sətir bölür. Çox uzun sətir olsa, cümlələrə bölür."""
    parts, cur = [], ""
    for line in body.split("\n"):
        pieces = [line] if len(line) <= max_chars else re.split(r"(?<=[.!?])\s+", line)
        for p in pieces:
            if cur and len(cur) + len(p) + 1 > max_chars:
                parts.append(cur.strip())
                cur = ""
            cur += p + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def build_chunks(text, max_chars=MAX_CHARS):
    chunks = []
    path = []
    body = []

    def flush():
        body_text = "\n".join(body).strip()
        if not body_text or not path:
            return
        prefix = " > ".join(path)
        is_menu = path[0] == "Menyu"

        lines = body_text.split("\n")
        q_idx = [i for i, l in enumerate(lines) if re.match(r"^\*\*.+\?\*\*$", l.strip())]
        if len(q_idx) >= 3:
            items = []
            if q_idx[0] > 0 and "\n".join(lines[:q_idx[0]]).strip():
                items.append("\n".join(lines[:q_idx[0]]).strip())
            bounds = q_idx + [len(lines)]
            for s, e in zip(bounds[:-1], bounds[1:]):
                items.append("\n".join(lines[s:e]).strip())
        else:
            items = [body_text]

        for item in items:
            limit = max_chars - len(prefix) - 1
            for part in split_text(item, limit):
                chunks.append({
                    "text": f"{prefix}\n{part}",
                    "path": prefix,
                    "is_menu": is_menu,
                })

    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            body = []
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1:
                path = []
            else:
                path = path[:level - 2] + [title]
        else:
            body.append(line)
    flush()
    return chunks


chunks = build_chunks(text)
print("Chunk sayı:", len(chunks), "| menyu:", sum(c["is_menu"] for c in chunks))

embedder = SentenceTransformer(EMBED_MODEL)


def embed_texts(texts, batch_size=32):
    """Mətn siyahısını normallaşdırılmış vektorlara çevirir."""
    return embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


embeddings = embed_texts([c["text"] for c in chunks])
is_menu_mask = np.array([c["is_menu"] for c in chunks])

STOP = {"qədər", "neçəy", "hansı", "varmı", "olar", "üçün", "necə",
        "nəyə", "nədir", "haqqı", "barə", "mənə", "göstə", "deyin",
        "menyu", "menu"}


def lower_az(s):
    """Azərbaycan hərflərini nəzərə alaraq kiçik hərfə çevirir."""
    return s.replace("İ", "i").replace("I", "ı").lower()


def norm_words(s):
    """Mətni kiçik hərfə çevirir, söz köklərini (ilk 5 hərf) qaytarır."""
    words = re.findall(r"\w+", lower_az(s))
    return {w[:5] for w in words if len(w) >= 3} - STOP


def head(c):
    """Chunk-ın ilk iki sətri (başlıq yolu + ilk məzmun / FAQ sualı)."""
    return " ".join(c["text"].split("\n")[:2])


head_words = [norm_words(head(c)) for c in chunks]

N = len(chunks)
df = Counter(w for hw in head_words for w in hw)
idf = {w: np.log(N / n) / np.log(N) for w, n in df.items()}


def search(query, mask=None, k=TOP_K, boost=0.3):
    """Vektor oxşarlığı + chunk başlığında (IDF çəkili) söz uyğunluğu."""
    q = embed_texts([query])[0]
    scores = embeddings @ q
    qw = norm_words(query)
    bonus = np.array([sum(idf[w] for w in (qw & hw)) for hw in head_words]) * boost
    scores = scores + bonus
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[i]) for i in top if scores[i] > -np.inf]


def retrieve_general(query, k=TOP_K):
    """Menyudan kənar chunk-lar arasında axtarır."""
    return search(query, mask=~is_menu_mask, k=k)


def retrieve_menu(query, k=TOP_K):
    """Yalnız menyu chunk-ları arasında axtarır."""
    return search(query, mask=is_menu_mask, k=k)


def retrieve(query, k=TOP_K):
    """Sorğuda 'menyu' və ya 'menu' sözü varsa menyu retrieverinə yönləndirir."""
    q = lower_az(query)
    if "menyu" in q or "menu" in q:
        return "menu", retrieve_menu(query, k)
    return "general", retrieve_general(query, k)


def format_context(results):
    """Tapılan chunk-ları LLM üçün kontekst mətninə çevirir."""
    return "\n\n---\n\n".join(c["text"] for _, c in results)


# ---------------------------------------------------------------------------
# 3. Tool-lar: kalkulyator və şəkil generasiyası
# ---------------------------------------------------------------------------
OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.USub: op.neg, ast.UAdd: op.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("İcazə verilməyən ifadə")


def calculator(expression: str) -> str:
    """Yalnız rəqəm və + - * / ( ) simvollarından ibarət ifadəni hesablayır."""
    try:
        expr = expression.replace(",", ".")
        result = _eval(ast.parse(expr, mode="eval").body)
        return f"{result:.2f}"
    except Exception as e:
        return f"Xəta: {e}"


CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Riyazi ifadəni hesablayır. Cəm, fərq, vurma, bölmə, faiz, "
            "10% xidmət haqqı kimi bütün hesablamalar üçün istifadə et. "
            "Yalnız rəqəmlər və + - * / ( ) işlət. Məsələn: (14 + 5) * 1.1"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Hesablanacaq ifadə, məsələn (14 + 5) * 1.1",
                }
            },
            "required": ["expression"],
        },
    },
}

IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Müştəri yeməyin və ya inqrediyentin şəklini istəyəndə şəkil yaradır. "
            "prompt_en yalnız İNGİLİSCƏ olmalıdır."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt_en": {
                    "type": "string",
                    "description": (
                        "Yeməyin/inqrediyentin ingiliscə adı, 1-4 söz. "
                        "Məsələn: chestnuts, saffron, lamb kebab. Əlavə söz yazma."
                    ),
                }
            },
            "required": ["prompt_en"],
        },
    },
}

TOOLS = [CALCULATOR_TOOL, IMAGE_TOOL]

# --- Şəkil generasiyası ---
image_cache = {}
MAX_CACHE = 50
_local_pipe = None


def _image_from_api(prompt):
    """İstəyə bağlı Hugging Face Inference API şəkil generasiyası."""
    if not HF_TOKEN:
        raise RuntimeError("Şəkil generasiyası üçün HF_TOKEN tələb olunur.")
    from huggingface_hub import InferenceClient
    hf = InferenceClient(token=HF_TOKEN)
    return hf.text_to_image(prompt, model=IMAGE_MODEL)


def _image_local(prompt):
    """Serverdə GPU varsa, modeli yerli işlədir (IMAGE_BACKEND=local)."""
    global _local_pipe
    import torch
    from diffusers import AutoPipelineForText2Image
    if _local_pipe is None:
        use_cuda = torch.cuda.is_available()
        _local_pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL,
            torch_dtype=torch.float16 if use_cuda else torch.float32,
            token=HF_TOKEN,
        ).to("cuda" if use_cuda else "cpu")
    return _local_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]


def generate_image(prompt_en: str, ctx: dict) -> str:
    """İngiliscə təsvirə görə şəkil yaradır. Şəkil ctx['image']-də saxlanılır."""
    if IMAGE_BACKEND == "disabled":
        return "Şəkil generasiyası hazırkı pulsuz deploy quruluşunda deaktivdir."
    key = prompt_en.lower().strip()
    if not key:
        return "Xəta: boş təsvir"

    if key in image_cache:
        print(f"  [cache] '{key}' keşdən götürüldü")
        ctx["image"] = image_cache[key]
        return "generated image"

    full_prompt = f"{prompt_en}, professional food photography, close-up, white background"
    try:
        img = _image_local(full_prompt) if IMAGE_BACKEND == "local" else _image_from_api(full_prompt)
    except Exception as e:
        print("  [image error]", repr(e))
        return "Xəta: şəkil yaradıla bilmədi"

    if len(image_cache) >= MAX_CACHE:
        image_cache.pop(next(iter(image_cache)))
    image_cache[key] = img
    ctx["image"] = img
    return "generated image"


def execute_tool(name, args, ctx):
    if name == "calculator":
        return calculator(args.get("expression", ""))
    if name == "generate_image":
        return generate_image(args.get("prompt_en", ""), ctx)
    return f"Naməlum tool: {name}"


# ---------------------------------------------------------------------------
# 4. LLM və dialoq məntiqi
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Sən Narxana restoranlar şəbəkəsinin nəzakətli və köməksevər ofisiantısan.

DİL VƏ FORMAT:
- Müştəri hansı dildə yazırsa, o dildə cavab ver (əsasən Azərbaycan dilində).
- Cavab qısa və aydın olsun: 2-5 cümlə. Siyahı lazımdırsa, qısa siyahı ver.
- Qiyməti həmişə AZN ilə yaz. Nəzakətli, isti ton saxla.

MƏLUMAT QAYDASI:
- Yalnız "Kontekst" bölməsində verilən məlumata əsaslan. Kontekstdə cavab yoxdursa,
  "Bu barədə dəqiq məlumatım yoxdur, qaynar xətt: *1909" de. Heç nə uydurma.
- Yemək təklif edəndə qiyməti, etiketləri ([vegan], [vegetarian] və s.) və allergenləri
  kontekstdən götür. Allergen və pəhriz suallarında ehtiyatlı ol.
- Restoranla əlaqəsi olmayan suallara nəzakətlə imtina et və restoran mövzusuna qayıt.

HESABLAMA QAYDASI:
- Heç bir hesablamanı özün etmə. Cəm, fərq, vurma, endirim, faiz və xidmət haqqı
  üçün HƏR DƏFƏ calculator tool-unu çağır və nəticəni ondan götür.
- Restoranda yeməyə 10% xidmət haqqı əlavə olunur (çatdırılma və take-away-da yoxdur).

ŞƏKİL QAYDASI:
- Müştəri yeməyin və ya inqrediyentin şəklini istəyəndə generate_image tool-unu çağır.
- prompt_en yalnız İNGİLİSCƏ və qısa olmalıdır (1-4 söz, məsələn: chestnuts).
- Şəkil yaradılandan sonra bir cümlə ilə nə yaradıldığını və qısa izahını yaz.
"""

FORCE_IMAGE = {"type": "function", "function": {"name": "generate_image"}}


def run_llm(messages, ctx, tools=None, tool_choice=None, max_rounds=4):
    """LLM-i çağırır; tool istəsə icra edir, yekun cavabı qaytarır."""
    tools = tools or TOOLS
    choice = tool_choice

    for _ in range(max_rounds):
        kwargs = dict(model=CHAT_MODEL, messages=messages, tools=tools)
        if choice:
            kwargs["tool_choice"] = choice
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            print("  [llm error]", repr(e))
            if choice:
                choice = None
                continue
            return "Üzr istəyirəm, indi cavab verə bilmirəm. Bir az sonra yenidən cəhd edin."
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(tc.function.name, args, ctx)
            print(f"  [tool] {tc.function.name}({args}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        choice = None

    return "Üzr istəyirəm, cavabı hazırlaya bilmədim."


def respond(query, history=None):
    """Bir sorğuya cavab verir.
    Qaytarır: (cavab, şəkil_rejimi_idi, şəkil_və_ya_None)"""
    history = history or []
    ctx = {"image": None}
    q = lower_az(query)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    if "generate" in q:
        messages.append({"role": "user", "content": query})
        answer = run_llm(messages, ctx, tools=[IMAGE_TOOL], tool_choice=FORCE_IMAGE)
        return answer, True, ctx["image"]

    route, res = retrieve(query)
    print(f"  [retrieval: {route}]")
    messages.append({
        "role": "user",
        "content": f"Kontekst:\n{format_context(res)}\n\nSual: {query}",
    })
    answer = run_llm(messages, ctx, tools=[CALCULATOR_TOOL])
    return answer, False, None


# ---------------------------------------------------------------------------
# 5. Streamlit veb interfeysi
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Narxana ağıllı ofisiant botu",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Narxana ağıllı ofisiant botu")
st.caption(
    "Restoran menyusu, qiymətlər və xidmətlər haqqında suallarınızı verə bilərsiniz."
)
st.info(
    "Menyu ilə bağlı sual verərkən sualınızda **menyu** sözünü işlədin. "
    "Şəkil generasiyası bu deploy quruluşunda standart olaraq deaktivdir."
)

with st.sidebar:
    st.subheader("Narxana botu")
    st.write("Məlumat bazası: `narxana_restoran.md`")
    st.write(f"Model: `{CHAT_MODEL}`")
    if st.button("Söhbəti təmizlə", use_container_width=True):
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_history" not in st.session_state:
    st.session_state.llm_history = []

# Əvvəlki mesajları göstər.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image") is not None:
            st.image(message["image"], caption="Yaradılmış şəkil", use_container_width=True)

# Yeni mesajı qəbul et və cavablandır.
if query := st.chat_input("Sualınızı yazın..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Cavab hazırlanır..."):
            answer, is_image, img = respond(query, st.session_state.llm_history)

        st.markdown(answer)
        if img is not None:
            st.image(img, caption="Yaradılmış şəkil", use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "image": img if is_image else None,
    })

    # LLM üçün rol/content formatında söhbət tarixçəsi.
    st.session_state.llm_history.extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": "generated image" if is_image else answer},
    ])
    st.session_state.llm_history = st.session_state.llm_history[-2 * MAX_PAIRS:]
