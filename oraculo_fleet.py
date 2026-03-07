import os
import glob
import hashlib
import shutil
import tempfile

import streamlit as st
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader


# ======================
# Config / Paths
# ======================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DOCS_DIR = os.path.join(APP_DIR, "base_docs")
CHROMA_DIR = os.path.join(APP_DIR, "chroma_db")
COLLECTION_NAME = "fleetpro0_kb"
TIPOS_SUPORTADOS = (".pdf", ".txt", ".csv", ".xlsx")

MEMORIA_PADRAO = ConversationBufferMemory()


CONFIG_MODELOS = {
    "Groq": {
        "modelos": ["llama-3.1-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini"],
        "chat": ChatOpenAI,
    },
}

# Suporte explícito a imagem (bloqueio sem tentativa/erro)
MODELOS_COM_IMAGEM = {
    "OpenAI": {"gpt-4o", "gpt-4o-mini"},
    "Groq": set(),
}


# ======================
# Utilitários base_docs / RAG
# ======================
def _listar_arquivos_base(pasta: str):
    arquivos = []
    for ext in TIPOS_SUPORTADOS:
        arquivos.extend(glob.glob(os.path.join(pasta, f"*{ext}")))
    return sorted([a for a in arquivos if os.path.isfile(a)])


def _hash_dos_arquivos(pasta: str) -> str:
    h = hashlib.sha256()
    for path in _listar_arquivos_base(pasta):
        stat = os.stat(path)
        h.update(path.encode("utf-8"))
        h.update(str(stat.st_mtime).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
    return h.hexdigest()


def _ler_hash_salvo():
    hash_path = os.path.join(CHROMA_DIR, "base_hash.txt")
    if os.path.exists(hash_path):
        with open(hash_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def _salvar_hash(base_hash: str):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    hash_path = os.path.join(CHROMA_DIR, "base_hash.txt")
    with open(hash_path, "w", encoding="utf-8") as f:
        f.write(base_hash)


def _apagar_indice():
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)


def carregar_documentos_fixos(pasta=BASE_DOCS_DIR):
    docs = []

    for path in glob.glob(os.path.join(pasta, "*.pdf")):
        docs.extend(PyPDFLoader(path).load())

    for path in glob.glob(os.path.join(pasta, "*.txt")):
        docs.extend(TextLoader(path, encoding="utf-8").load())

    for path in glob.glob(os.path.join(pasta, "*.csv")):
        docs.extend(CSVLoader(path, encoding="utf-8").load())

    for path in glob.glob(os.path.join(pasta, "*.xlsx")):
        docs.extend(UnstructuredExcelLoader(path).load())

    return docs


def _formatar_fonte(doc) -> str:
    md = doc.metadata or {}
    source = md.get("source") or md.get("file_path") or "fonte_desconhecida"
    source = os.path.basename(str(source))

    page = md.get("page")
    if page is not None:
        return f"{source} (pág. {page + 1})"

    sheet = md.get("sheet_name")
    if sheet:
        return f"{source} (aba: {sheet})"

    return source


def _formatar_contexto(docs) -> str:
    partes = []
    for d in docs:
        fonte = _formatar_fonte(d)
        partes.append(f"Fonte: {fonte}\nTrecho:\n{d.page_content}")
    return "\n\n---\n\n".join(partes)


def _formatar_fontes_unicas(docs) -> str:
    fontes = []
    seen = set()
    for d in docs:
        f = _formatar_fonte(d)
        if f not in seen:
            fontes.append(f"- {f}")
            seen.add(f)
    return "\n".join(fontes) if fontes else "- (sem fonte disponível)"


@st.cache_resource(show_spinner=True)
def obter_vectorstore(openai_api_key: str):
    os.makedirs(BASE_DOCS_DIR, exist_ok=True)

    arquivos = _listar_arquivos_base(BASE_DOCS_DIR)
    if not arquivos:
        raise RuntimeError(
            "Nenhum documento em base_docs/. "
            "Crie a pasta 'base_docs' ao lado do app.py e coloque PDFs/TXT/CSV/XLSX nela."
        )

    base_hash = _hash_dos_arquivos(BASE_DOCS_DIR)
    hash_antigo = _ler_hash_salvo()

    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    if (hash_antigo != base_hash) or (not os.path.isdir(CHROMA_DIR)):
        _apagar_indice()

        docs = carregar_documentos_fixos(BASE_DOCS_DIR)
        chunks = _split_docs(docs)

        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )

        _salvar_hash(base_hash)
        return vs

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vs


def _limpar_cache_vectorstore_se_base_mudou():
    os.makedirs(BASE_DOCS_DIR, exist_ok=True)
    if not _listar_arquivos_base(BASE_DOCS_DIR):
        return
    base_hash = _hash_dos_arquivos(BASE_DOCS_DIR)
    hash_antigo = _ler_hash_salvo()
    if base_hash and (hash_antigo != base_hash):
        obter_vectorstore.clear()


def responder_com_rag(pergunta: str, chat_model, retriever) -> str:
    docs = retriever.get_relevant_documents(pergunta)
    contexto = _formatar_contexto(docs)
    fontes = _formatar_fontes_unicas(docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é o FleetPro0, um especialista no tema FleetPro. "
                "Responda sempre em português.\n\n"
                "Regras:\n"
                "- Use PRIORITARIAMENTE as evidências do contexto recuperado.\n"
                "- Se não houver evidência suficiente, diga explicitamente que não encontrou na base.\n"
                "- Evite suposições.\n"
                "- Quando possível, finalize com 'Fontes' usando a lista fornecida.",
            ),
            (
                "human",
                "Pergunta: {pergunta}\n\n"
                "Contexto (trechos recuperados):\n{contexto}\n\n"
                "Responda de forma objetiva.\n\n"
                "Fontes (use esta lista ao final):\n{fontes}\n",
            ),
        ]
    )

    msg = prompt.invoke({"pergunta": pergunta, "contexto": contexto, "fontes": fontes})
    resp = chat_model.invoke(msg.messages)
    return resp.content.strip()


# ======================
# Imagem
# ======================
def suporta_imagem(provedor: str, modelo: str) -> bool:
    return modelo in MODELOS_COM_IMAGEM.get(provedor, set())


def _salvar_imagem_temp(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(uploaded_file.read())
        return temp.name


def analisar_imagem(uploaded_image, pergunta: str):
    chat = st.session_state.get("chat")
    provedor = st.session_state.get("provedor")
    modelo = st.session_state.get("modelo")

    if chat is None or not provedor or not modelo:
        st.info("Inicialize o Oráculo na barra lateral antes de analisar a imagem.")
        return

    if not suporta_imagem(provedor, modelo):
        st.warning("O provedor/modelo selecionado não suporta análise de imagem.")
        return

    if uploaded_image is None:
        st.warning("Faça upload de uma imagem antes de analisar.")
        return

    image_path = _salvar_imagem_temp(uploaded_image)

    msg = [
        {
            "type": "text",
            "text": (
                "Você é o FleetPro0, especialista no tema FleetPro. "
                "Analise a imagem enviada e responda objetivamente.\n\n"
                f"Pergunta do usuário: {pergunta}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"file://{image_path}"},
        },
    ]

    st.chat_message("human").markdown(f"**Pergunta sobre a imagem:** {pergunta}")

    with st.chat_message("ai"):
        resposta = st.write_stream(chat.stream(msg))

    memoria = st.session_state.get("memoria", MEMORIA_PADRAO)
    memoria.chat_memory.add_user_message(f"[Imagem] {pergunta}")
    memoria.chat_memory.add_ai_message(resposta)
    st.session_state["memoria"] = memoria


# ======================
# Inicialização do modelo
# ======================
def inicializar_oraculo(provedor: str, modelo: str, api_key: str):
    api_key = (api_key or "").strip()
    if not api_key:
        st.error("Informe a API key antes de inicializar.")
        return

    chat_cls = CONFIG_MODELOS[provedor]["chat"]
    chat = chat_cls(model=modelo, api_key=api_key)

    st.session_state["chat"] = chat
    st.session_state["provedor"] = provedor
    st.session_state["modelo"] = modelo
    st.session_state.setdefault("memoria", MEMORIA_PADRAO)

    st.success("Oráculo inicializado com sucesso.")


# ======================
# UI
# ======================
def pagina_chat():
    st.header("🕵️‍♂️ Oráculo FleetPro 🛠️", divider=True)

    chat_model = st.session_state.get("chat")
    memoria = st.session_state.get("memoria", MEMORIA_PADRAO)

    # Config do RAG
    usar_rag = st.session_state.get("usar_rag", False)
    k = st.session_state.get("k_rag", 4)

    # Renderiza histórico
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    input_usuario = st.chat_input("Pergunte ao FleetPro0 (com ou sem base_docs, conforme configuração)")

    if input_usuario:
        if chat_model is None:
            st.info("Clique em 'Inicializar Oráculo' na barra lateral para carregar o modelo.")
            st.stop()

        st.chat_message("human").markdown(input_usuario)

        with st.chat_message("ai"):
            try:
                if usar_rag:
                    openai_key = (st.session_state.get("api_key_OpenAI") or "").strip()
                    if not openai_key:
                        st.error("Para usar RAG com base_docs, informe a API key da OpenAI (aba Modelos).")
                        st.stop()

                    _limpar_cache_vectorstore_se_base_mudou()
                    vs = obter_vectorstore(openai_key)
                    retriever = vs.as_retriever(search_kwargs={"k": k})

                    resposta = responder_com_rag(input_usuario, chat_model, retriever)
                    st.markdown(resposta)
                else:
                    resposta = st.write_stream(chat_model.stream(input_usuario))

            except Exception as e:
                st.error("Erro ao responder.")
                st.exception(e)
                st.stop()

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state["memoria"] = memoria


def sidebar():
    tabs = st.tabs(["Imagem (análise)", "RAG (base_docs)", "Modelos"])

    # --------
    # Aba imagem
    # --------
    provedor_atual = st.session_state.get("provedor")
    modelo_atual = st.session_state.get("modelo")

    with tabs[0]:
        st.subheader("Analisar imagem", divider=True)

        uploaded_image = st.file_uploader(
            "Upload de imagem (PNG/JPG/WebP)",
            type=["png", "jpg", "jpeg", "webp"],
        )
        pergunta_img = st.text_input(
            "Pergunta sobre a imagem",
            value="O que aparece na imagem? Extraia pontos principais e qualquer texto visível.",
        )

        pode_analisar = bool(provedor_atual and modelo_atual and suporta_imagem(provedor_atual, modelo_atual))
        if provedor_atual and modelo_atual and not pode_analisar:
            st.info("Imagem: indisponível para o provedor/modelo selecionado. Use OpenAI gpt-4o ou gpt-4o-mini.")

        st.button(
            "Analisar imagem",
            use_container_width=True,
            disabled=not pode_analisar,
            on_click=analisar_imagem,
            kwargs={"uploaded_image": uploaded_image, "pergunta": pergunta_img},
        )

    # --------
    # Aba RAG
    # --------
    with tabs[1]:
        st.subheader("RAG com base_docs", divider=True)

        usar_rag = st.toggle("Usar base_docs no chat (RAG)", value=st.session_state.get("usar_rag", False))
        st.session_state["usar_rag"] = usar_rag

        k = st.slider("Trechos recuperados (k)", min_value=2, max_value=10, value=st.session_state.get("k_rag", 4))
        st.session_state["k_rag"] = k

        st.caption("Coloque PDFs/TXT/CSV/XLSX em ./base_docs. O índice é recriado automaticamente quando a base muda.")
        st.caption("Para RAG, é necessária API key da OpenAI (embeddings), mesmo que o chat esteja em Groq.")

    # --------
    # Aba Modelos
    # --------
    with tabs[2]:
        st.subheader("Seleção de modelo", divider=True)

        provedor = st.selectbox("Provedor", list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox("Modelo", CONFIG_MODELOS[provedor]["modelos"])

        api_key = st.text_input(
            f"API key ({provedor})",
            value=st.session_state.get(f"api_key_{provedor}", ""),
            type="password",
        )
        st.session_state[f"api_key_{provedor}"] = api_key

        if st.button("Inicializar Oráculo", use_container_width=True):
            inicializar_oraculo(provedor, modelo, api_key)


def main():
    pagina_chat()
    with st.sidebar:
        sidebar()


if __name__ == "__main__":
    main()
