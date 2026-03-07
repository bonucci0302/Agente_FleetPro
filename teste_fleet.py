import os
import glob
import re
import hashlib
import shutil
from typing import Optional

import streamlit as st

# Guard rails de página (evita branco sem nada)
st.set_page_config(page_title="Oráculo FleetPro", layout="wide")


def safe_run(fn):
    try:
        fn()
    except Exception as e:
        st.error("O app falhou ao inicializar. O erro também deve aparecer no terminal.")
        st.exception(e)
        st.stop()


# Dependências
try:
    import pandas as pd
except Exception as e:
    st.error("Dependência ausente: pandas")
    st.exception(e)
    st.stop()

try:
    import openpyxl  # noqa: F401
except Exception as e:
    st.error("Dependência ausente: openpyxl (necessário para ler .xlsx)")
    st.exception(e)
    st.stop()

from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
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

SHEET_FP_MATRIZ = "FP MATRIZ"
COLUNAS_BUSCA_PN = ["PN GEN", "PN ALTERNATIVE", "PN NXP", "PN FLEETPRO"]

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


# ======================
# Utilitários
# ======================
def norm_pn(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "").replace(".", "")
    return s


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)


# ======================
# Utilitários base_docs / RAG fixo
# (mantidos só se você quiser usar RAG em PDFs/TXT/CSV/XLSX;
#  a Solução A NÃO usa Chroma para o lookup da FP MATRIZ)
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


def carregar_documentos_fixos(pasta=BASE_DOCS_DIR):
    docs = []

    for path in glob.glob(os.path.join(pasta, "*.pdf")):
        docs.extend(PyPDFLoader(path).load())

    for path in glob.glob(os.path.join(pasta, "*.txt")):
        docs.extend(TextLoader(path, encoding="utf-8").load())

    for path in glob.glob(os.path.join(pasta, "*.csv")):
        docs.extend(CSVLoader(path, encoding="utf-8").load())

    # Carrega Excel Matriz_FP de base_docs se existir (para RAG; não usado no lookup por PN)
    excel_path = os.path.join(pasta, "Matriz_FP.xlsx")
    if os.path.exists(excel_path):
        docs.extend(UnstructuredExcelLoader(excel_path).load())

    return docs


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

    from langchain_openai import OpenAIEmbeddings

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


# ======================
# Excel Matriz_FP (base_docs) -> DF (lookup determinístico)
# ======================
@st.cache_data(show_spinner=True)
def carregar_df_fp_matriz(file_path: str, sheet_name: str) -> "pd.DataFrame":
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    for c in COLUNAS_BUSCA_PN:
        if c not in df.columns:
            df[c] = ""

    return df


def formatar_linha_inteira_como_lista(df: "pd.DataFrame", row_index_df: int) -> str:
    if row_index_df < 0 or row_index_df >= len(df):
        return "Não foi possível localizar a linha no DataFrame (índice fora do intervalo)."

    row = df.iloc[row_index_df]
    partes = []

    for col in df.columns:
        val = row[col]
        if pd.isna(val):
            val = ""
        partes.append(f"- {col}: {str(val).strip()}")

    return "\n".join(partes)


def procurar_pn_e_listar_linhas(df: "pd.DataFrame", pn_input: str, max_resultados: int = 50) -> str:
    pn = norm_pn(pn_input)
    if not pn:
        return "Informe um PN válido para pesquisa."

    df_busca = df[COLUNAS_BUSCA_PN].copy()
    for c in COLUNAS_BUSCA_PN:
        df_busca[c] = df_busca[c].apply(norm_pn)

    mask = False
    for c in COLUNAS_BUSCA_PN:
        mask = mask | (df_busca[c] == pn)

    encontrados = df[mask]
    if encontrados.empty:
        return (
            f"Não encontrei o PN `{pn_input}` nas colunas: PN GEN, PN ALTERNATIVE, PN NXP, PN FLEETPRO."
        )

    if len(encontrados) > max_resultados:
        encontrados = encontrados.head(max_resultados)

    partes = [f"Encontrei **{len(encontrados)}** opção(ões) para o PN `{pn_input}`:\n"]

    for i, (idx_df, _row) in enumerate(encontrados.iterrows(), start=1):
        row_index_df = df.index.get_loc(idx_df)
        partes.append(f"### Opção {i}\n")
        partes.append(formatar_linha_inteira_como_lista(df, int(row_index_df)))
        partes.append("")

    return "\n".join(partes)


# ======================
# Inicialização do modelo (chat livre opcional)
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
    st.session_state.setdefault("memoria", ConversationBufferMemory())

    st.success("Oráculo inicializado com sucesso.")


# ======================
# UI
# ======================
def pagina_chat():
    st.header("🕵️‍♂️ Oráculo FleetPro 🛠️", divider=True)

    chat_model = st.session_state.get("chat")
    memoria = st.session_state.get("memoria", ConversationBufferMemory())

    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    input_usuario = st.chat_input("Digite um PN (busca em PN GEN / PN ALTERNATIVE / PN NXP / PN FLEETPRO)")

    if input_usuario:
        st.chat_message("human").markdown(input_usuario)

        with st.chat_message("ai"):
            try:
                usar_fp_matriz = st.session_state.get("usar_fp_matriz", True)

                if usar_fp_matriz:
                    excel_path = os.path.join(BASE_DOCS_DIR, "Matriz_FP.xlsx")
                    if not os.path.exists(excel_path):
                        st.error("Matriz_FP.xlsx não encontrado em base_docs. Coloque o arquivo lá e reinicie o app.")
                        st.stop()

                    df_fp = carregar_df_fp_matriz(excel_path, SHEET_FP_MATRIZ)

                    resposta = procurar_pn_e_listar_linhas(
                        df_fp,
                        input_usuario,
                        max_resultados=st.session_state.get("max_resultados_fp", 50),
                    )
                    st.markdown(resposta)

                else:
                    if chat_model is None:
                        st.info("Clique em 'Inicializar Oráculo' na barra lateral para carregar o modelo.")
                        st.stop()

                    resposta_stream = st.write_stream(chat_model.stream(input_usuario))
                    st.markdown(resposta_stream)
                    resposta = resposta_stream

            except Exception as e:
                st.error("Erro ao responder.")
                st.exception(e)
                st.stop()

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state["memoria"] = memoria


def sidebar():
    tabs = st.tabs(["FP MATRIZ", "Modelos"])

    with tabs[0]:
        st.subheader("Lookup FP MATRIZ (direto no Excel)", divider=True)

        usar_fp_matriz = st.toggle(
            "Usar FP MATRIZ no chat (lookup por PN)",
            value=st.session_state.get("usar_fp_matriz", True),
        )
        st.session_state["usar_fp_matriz"] = usar_fp_matriz

        max_resultados_fp = st.slider(
            "Máximo de opções retornadas (se houver muitas linhas)",
            min_value=5,
            max_value=200,
            value=st.session_state.get("max_resultados_fp", 50),
        )
        st.session_state["max_resultados_fp"] = max_resultados_fp

        st.caption(
            "A busca normaliza o PN (remove espaços, hífens e pontos; deixa em maiúsculo) "
            "e compara exatamente com as colunas PN GEN, PN ALTERNATIVE, PN NXP e PN FLEETPRO. "
            "Ao encontrar, retorna a(s) linha(s) inteira(s) como lista 'Coluna: valor'."
        )

    with tabs[1]:
        st.subheader("Seleção de modelo (chat livre opcional)", divider=True)

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
    safe_run(main)