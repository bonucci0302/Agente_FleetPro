import os
import glob
import re
import hashlib
import shutil
from typing import Optional

import streamlit as st

st.set_page_config(page_title="FleetPro Expert", layout="wide")


def safe_run(fn):
    try:
        fn()
    except Exception as e:
        st.error("O app falhou ao inicializar. O erro também deve aparecer no terminal.")
        st.exception(e)
        st.stop()


# Dependências obrigatórias
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
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document

# ======================
# Config / Paths
# ======================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DOCS_DIR = os.path.join(APP_DIR, "base_docs")
CHROMA_DIR = os.path.join(APP_DIR, "chroma_db")
COLLECTION_NAME = "fleetpro_kb_v2"

# URL do site FleetPro para crawling
FLEETPRO_SITE_URL = "https://fleetpro.com.br/produtos/"
FLEETPRO_SITE_CACHE = os.path.join(APP_DIR, "fleetpro_site_cache.json")

# Arquivo Excel principal (lookup de PN)
MATRIX_EXCEL = "Matriz_FP.xlsx"
SHEET_FP_MATRIZ = "FP MATRIZ"
COLUNAS_BUSCA_PN = ["PN GEN", "PN ALTERNATIVE", "PN NXP", "PN FLEETPRO"]

# Colunas que contêm modelos de equipamentos CASE IH / NHAG (texto com nomes de modelos)
COLUNAS_MODELOS_EQUIP = [
    "TRACTOR - CASE IH", "COMBINE - CASE IH", "HEADERS - CASE IH",
    "SCH - CASE IH", "SPRAYERS - CASE IH", "PLANTERS - CASE IH",
    "OTHER MACHINES - CASE IH",
    "TRACTOR - NHAG", "COMBINE - NHAG", "HEADERS - NHAG",
    "SPRAYERS - NHAG", "PLANTERS - NHAG",
    "FORAGE, BALERS and OTHERS - NHAG", "OTHER MACHINES - NHAG",
]

# Colunas que contêm PNs originais de outras marcas
COLUNAS_MARCAS_PN = ["JOHN DEERE", "MACDON", "AGCO", "IDEAL", "MASSEY FERGUSON", "VALTRA"]

# Coluna de projeto de marketing (busca por tipo de produto/categoria)
COLUNA_MARKETING = "MARKETING PROJECT"

# Valores exatos que aparecem na coluna MARKETING PROJECT da planilha.
# Cada entrada é um valor real + sinônimos/variações que o usuário pode digitar.
# A busca é feita com str.contains() parcial, então "ROLAMENTO" bate em "ROLAMENTOS" etc.
PALAVRAS_MARKETING = [
    # ── Valores exatos da coluna ─────────────────────────────────────────────
    "BUCKET",
    "ALL MAKES",
    "UNDERCARRIAGE",
    "PERIODICAL MAINTENANCE KITS",
    "CORREIAS",
    "FILTROS",
    "ROLAMENTOS",
    "PEÇAS DE DESGASTES",
    "PINOS & BUCHAS",
    "PLÁSTICOS",
    "CILINDROS",
    "LUBRIFICANTES",
    "CORRENTES",
    "VEDAÇÃO",
    "PULVERIZAÇÃO",
    "USINADOS",
    # ── Sinônimos / variações em português ───────────────────────────────────
    "CORREIA",          # → CORREIAS
    "FILTRO",           # → FILTROS
    "ROLAMENTO",        # → ROLAMENTOS
    "PECA DE DESGASTE", # → PEÇAS DE DESGASTES
    "PEÇA DE DESGASTE",
    "DESGASTE",
    "PINO",             # → PINOS & BUCHAS
    "PINOS",
    "BUCHA",            # → PINOS & BUCHAS
    "BUCHAS",
    "PLASTICO",         # → PLÁSTICOS
    "PLASTICOS",
    "PLÁSTICO",
    "CILINDRO",         # → CILINDROS
    "LUBRIFICANTE",     # → LUBRIFICANTES
    "CORRENTE",         # → CORRENTES
    "VEDACAO",          # → VEDAÇÃO (sem acento)
    "PULVERIZACAO",     # → PULVERIZAÇÃO (sem acento)
    "USINADO",          # → USINADOS
    "KIT",              # → PERIODICAL MAINTENANCE KITS
    "KITS",
    "MANUTENCAO",       # → PERIODICAL MAINTENANCE KITS
    "MANUTENÇÃO",
    # ── Sinônimos em inglês ───────────────────────────────────────────────────
    "BELT",             # → CORREIAS
    "BELTS",
    "FILTER",           # → FILTROS
    "FILTERS",
    "BEARING",          # → ROLAMENTOS
    "BEARINGS",
    "BUSHING",          # → PINOS & BUCHAS
    "BUSHINGS",
    "PIN",              # → PINOS & BUCHAS
    "PINS",
    "CYLINDER",         # → CILINDROS
    "CYLINDERS",
    "LUBRICANT",        # → LUBRIFICANTES
    "LUBRICANTS",
    "CHAIN",            # → CORRENTES
    "CHAINS",
    "SEAL",             # → VEDAÇÃO
    "SEALS",
    "SPRAY",            # → PULVERIZAÇÃO
    "SPRAYING",
    "WEAR",             # → PEÇAS DE DESGASTES
    "WEAR PARTS",
    "MAINTENANCE",      # → PERIODICAL MAINTENANCE KITS
]

# Mapeia sinônimos/variações → valor EXATO que aparece na coluna MARKETING PROJECT.
# Se o termo detectado não está aqui, ele é usado diretamente como filtro parcial.
MAPA_SINONIMOS_MARKETING = {
    # Português singular → plural (valor da coluna)
    "CORREIA":          "CORREIAS",
    "FILTRO":           "FILTROS",
    "ROLAMENTO":        "ROLAMENTOS",
    "PECA DE DESGASTE": "PEÇAS DE DESGASTES",
    "PEÇA DE DESGASTE": "PEÇAS DE DESGASTES",
    "DESGASTE":         "PEÇAS DE DESGASTES",
    "PINO":             "PINOS & BUCHAS",
    "PINOS":            "PINOS & BUCHAS",
    "BUCHA":            "PINOS & BUCHAS",
    "BUCHAS":           "PINOS & BUCHAS",
    "PLASTICO":         "PLÁSTICOS",
    "PLASTICOS":        "PLÁSTICOS",
    "PLÁSTICO":         "PLÁSTICOS",
    "CILINDRO":         "CILINDROS",
    "LUBRIFICANTE":     "LUBRIFICANTES",
    "CORRENTE":         "CORRENTES",
    "VEDACAO":          "VEDAÇÃO",
    "PULVERIZACAO":     "PULVERIZAÇÃO",
    "USINADO":          "USINADOS",
    "KIT":              "PERIODICAL MAINTENANCE KITS",
    "KITS":             "PERIODICAL MAINTENANCE KITS",
    "MANUTENCAO":       "PERIODICAL MAINTENANCE KITS",
    "MANUTENÇÃO":       "PERIODICAL MAINTENANCE KITS",
    "MAINTENANCE":      "PERIODICAL MAINTENANCE KITS",
    # Inglês → valor da coluna
    "BELT":             "CORREIAS",
    "BELTS":            "CORREIAS",
    "FILTER":           "FILTROS",
    "FILTERS":          "FILTROS",
    "BEARING":          "ROLAMENTOS",
    "BEARINGS":         "ROLAMENTOS",
    "BUSHING":          "PINOS & BUCHAS",
    "BUSHINGS":         "PINOS & BUCHAS",
    "PIN":              "PINOS & BUCHAS",
    "PINS":             "PINOS & BUCHAS",
    "CYLINDER":         "CILINDROS",
    "CYLINDERS":        "CILINDROS",
    "LUBRICANT":        "LUBRIFICANTES",
    "LUBRICANTS":       "LUBRIFICANTES",
    "CHAIN":            "CORRENTES",
    "CHAINS":           "CORRENTES",
    "SEAL":             "VEDAÇÃO",
    "SEALS":            "VEDAÇÃO",
    "SPRAY":            "PULVERIZAÇÃO",
    "SPRAYING":         "PULVERIZAÇÃO",
    "WEAR":             "PEÇAS DE DESGASTES",
    "WEAR PARTS":       "PEÇAS DE DESGASTES",
}


# Mapa de palavras-chave para nomes de colunas (busca por fabricante/tipo)
MAPA_PALAVRAS_COLUNAS = {
    "JOHN DEERE": ["JOHN DEERE"],
    "JD": ["JOHN DEERE"],
    "DEERE": ["JOHN DEERE"],
    "MACDON": ["MACDON"],
    "AGCO": ["AGCO"],
    "IDEAL": ["IDEAL"],
    "MASSEY": ["MASSEY FERGUSON"],
    "MASSEY FERGUSON": ["MASSEY FERGUSON"],
    "MF": ["MASSEY FERGUSON"],
    "VALTRA": ["VALTRA"],
    "CASE": ["TRACTOR - CASE IH", "COMBINE - CASE IH", "HEADERS - CASE IH",
             "SCH - CASE IH", "SPRAYERS - CASE IH", "PLANTERS - CASE IH",
             "OTHER MACHINES - CASE IH"],
    "CASE IH": ["TRACTOR - CASE IH", "COMBINE - CASE IH", "HEADERS - CASE IH",
                "SCH - CASE IH", "SPRAYERS - CASE IH", "PLANTERS - CASE IH",
                "OTHER MACHINES - CASE IH"],
    "TRATOR": ["TRACTOR - CASE IH", "TRACTOR - NHAG"],
    "TRACTOR": ["TRACTOR - CASE IH", "TRACTOR - NHAG"],
    "COLHEITADEIRA": ["COMBINE - CASE IH", "COMBINE - NHAG"],
    "COMBINE": ["COMBINE - CASE IH", "COMBINE - NHAG"],
    "HEADER": ["HEADERS - CASE IH", "HEADERS - NHAG"],
    "PLATAFORMA": ["HEADERS - CASE IH", "HEADERS - NHAG"],
    "PULVERIZADOR": ["SPRAYERS - CASE IH", "SPRAYERS - NHAG"],
    "SPRAYER": ["SPRAYERS - CASE IH", "SPRAYERS - NHAG"],
    "PLANTADEIRA": ["PLANTERS - CASE IH", "PLANTERS - NHAG"],
    "PLANTER": ["PLANTERS - CASE IH", "PLANTERS - NHAG"],
    "NHAG": ["TRACTOR - NHAG", "COMBINE - NHAG", "HEADERS - NHAG",
             "SPRAYERS - NHAG", "PLANTERS - NHAG",
             "FORAGE, BALERS and OTHERS - NHAG", "OTHER MACHINES - NHAG"],
    "NEW HOLLAND": ["TRACTOR - NHAG", "COMBINE - NHAG", "HEADERS - NHAG",
                    "SPRAYERS - NHAG", "PLANTERS - NHAG",
                    "FORAGE, BALERS and OTHERS - NHAG", "OTHER MACHINES - NHAG"],
}

TIPOS_RAG = (".pdf", ".txt", ".csv")

CONFIG_MODELOS = {
    "Groq": {
        "modelos": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o"],
        "chat": ChatOpenAI,
    },
}


# ======================
# Utilitários
# ======================
def norm_pn(x) -> str:
    """Normaliza o Part Number: maiúsculo, sem espaços, hífens e pontos."""
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = re.sub(r"\s+", "", s)
    s = s.replace("-", "").replace(".", "")
    return s


_STOPWORDS_PN = {
    "ITEM", "PECA", "PEÇA", "NUMERO", "NÚMERO", "GENUINO", "GENUÍNO",
    "ORIGINAL", "SUBSTITUTO", "CODIGO", "CÓDIGO", "AJUDE", "VEJA",
    "OPCOES", "OPÇÕES", "ESSE", "ESTE", "PARA", "QUAL", "QUAIS",
    "BUSCA", "ENCONTRA", "ACHAR", "TENHO", "PRECISO", "PRODUCT",
    "NUMBER", "PART", "FIND", "SEARCH", "HELP", "THE", "AND", "PN",
    "COM", "UMA", "VER", "NAO", "NÃO", "SIM", "POR", "QUE", "MEU",
    "MIM", "ELE", "ELA", "SER", "TEM", "FOI", "TIPO", "MODELO",
}


def extrair_pns_da_mensagem(texto: str) -> list:
    candidatos_raw = re.findall(r"[A-Za-z0-9]+(?:[-\.][A-Za-z0-9]+)*", texto)
    candidatos = []
    for c in candidatos_raw:
        norm = norm_pn(c)
        if (
            len(norm) >= 4
            and re.search(r"\d", norm)
            and norm not in _STOPWORDS_PN
            and not norm.isalpha()
        ):
            candidatos.append(norm)
    return sorted(set(candidatos), key=lambda x: -len(x))


def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)


# ======================
# Web Scraping – Site FleetPro
# ======================
def _crawl_fleetpro_site(base_url: str, max_pages: int = 100) -> list:
    """
    Faz crawling do site FleetPro a partir da URL base de produtos.
    Segue links internos da mesma origem e extrai texto de cada página.
    Retorna lista de objetos Document (LangChain).
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin, urlparse
    except ImportError as e:
        st.warning(f"Dependência ausente para scraping do site: {e}. Instale com: pip install requests beautifulsoup4")
        return []

    base_domain = urlparse(base_url).netloc
    visited = set()
    queue = [base_url]
    documents = []

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; FleetProRAGBot/1.0; "
            "+https://fleetpro.com.br)"
        )
    }

    progress = st.progress(0, text="Iniciando crawling do site FleetPro...")

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        progress_pct = min(len(visited) / max_pages, 1.0)
        progress.progress(progress_pct, text=f"Crawling: {url[:80]}...")

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue

            # Só processa HTML
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove tags desnecessárias
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "noscript", "iframe", "svg", "form"]):
                tag.decompose()

            # Extrai título
            title = soup.find("title")
            title_text = title.get_text(strip=True) if title else url

            # Extrai texto principal
            main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|produto|product", re.I))
            text_source = main if main else soup.find("body")

            if not text_source:
                continue

            # Extrai parágrafos e headings
            blocos = []
            for el in text_source.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th"]):
                txt = el.get_text(separator=" ", strip=True)
                if txt and len(txt) > 20:
                    blocos.append(txt)

            texto_completo = f"# {title_text}\nFonte: {url}\n\n" + "\n\n".join(blocos)

            if len(texto_completo.strip()) > 100:
                doc = Document(
                    page_content=texto_completo,
                    metadata={"source": url, "title": title_text, "tipo": "site_fleetpro"},
                )
                documents.append(doc)

            # Descobre novos links internos
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full_url = urljoin(url, href)
                parsed = urlparse(full_url)

                # Só segue links do mesmo domínio, sem âncoras/parâmetros irrelevantes
                if (
                    parsed.netloc == base_domain
                    and full_url not in visited
                    and full_url not in queue
                    and not any(full_url.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".zip", ".xml"])
                    and "#" not in full_url
                ):
                    queue.append(full_url)

        except Exception as e:
            # Ignora erros em páginas individuais silenciosamente
            continue

    progress.empty()

    return documents


def _carregar_cache_site() -> dict:
    """Carrega cache do site (hash + documentos serializados)."""
    import json
    if os.path.exists(FLEETPRO_SITE_CACHE):
        try:
            with open(FLEETPRO_SITE_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _salvar_cache_site(dados: dict):
    """Salva cache do site."""
    import json
    try:
        with open(FLEETPRO_SITE_CACHE, "w", encoding="utf-8") as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Não foi possível salvar cache do site: {e}")


def obter_docs_do_site(forcar_recrawl: bool = False) -> list:
    """
    Retorna documentos do site FleetPro.
    Usa cache local para evitar crawling repetido.
    """
    cache = _carregar_cache_site()

    if not forcar_recrawl and cache.get("documentos"):
        docs = [
            Document(
                page_content=d["page_content"],
                metadata=d["metadata"],
            )
            for d in cache["documentos"]
        ]
        return docs

    # Faz o crawling
    with st.spinner("🌐 Coletando dados do site fleetpro.com.br..."):
        docs = _crawl_fleetpro_site(FLEETPRO_SITE_URL, max_pages=150)

    if docs:
        # Salva cache
        _salvar_cache_site({
            "url_base": FLEETPRO_SITE_URL,
            "total_paginas": len(docs),
            "documentos": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in docs
            ],
        })
        st.success(f"✅ {len(docs)} página(s) coletadas do site FleetPro.")
    else:
        st.warning("Nenhuma página coletada do site. Verifique a conectividade.")

    return docs


# ======================
# RAG – Documentos de texto + Site
# ======================
def _listar_arquivos_rag(pasta: str):
    arquivos = []
    for ext in TIPOS_RAG:
        arquivos.extend(glob.glob(os.path.join(pasta, f"*{ext}")))
    return sorted([a for a in arquivos if os.path.isfile(a)])


def _hash_dos_arquivos(lista_arquivos) -> str:
    h = hashlib.sha256()
    for path in lista_arquivos:
        stat = os.stat(path)
        h.update(path.encode("utf-8"))
        h.update(str(stat.st_mtime).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
    return h.hexdigest()


def _hash_completo(lista_arquivos: list, usar_site: bool) -> str:
    """Hash combinando arquivos locais + flag do site."""
    h = hashlib.sha256()
    for path in lista_arquivos:
        stat = os.stat(path)
        h.update(path.encode("utf-8"))
        h.update(str(stat.st_mtime).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
    h.update(b"site_on" if usar_site else b"site_off")
    # Inclui hash do cache do site se existir
    if usar_site and os.path.exists(FLEETPRO_SITE_CACHE):
        stat = os.stat(FLEETPRO_SITE_CACHE)
        h.update(str(stat.st_mtime).encode("utf-8"))
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


def _carregar_documentos_rag(pasta: str):
    docs = []

    for path in glob.glob(os.path.join(pasta, "*.pdf")):
        try:
            docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            st.warning(f"Não consegui ler {os.path.basename(path)}: {e}")

    for path in glob.glob(os.path.join(pasta, "*.txt")):
        try:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            st.warning(f"Não consegui ler {os.path.basename(path)}: {e}")

    for path in glob.glob(os.path.join(pasta, "*.csv")):
        try:
            docs.extend(CSVLoader(path, encoding="utf-8").load())
        except Exception as e:
            st.warning(f"Não consegui ler {os.path.basename(path)}: {e}")

    return docs


@st.cache_resource(show_spinner="Indexando documentos de conhecimento (RAG)...")
def obter_vectorstore(usar_site: bool = True):
    """
    Cria ou carrega o banco vetorial (Chroma) com documentos locais + site FleetPro.
    O cache só recria o índice se os arquivos ou o cache do site mudaram.
    """
    os.makedirs(BASE_DOCS_DIR, exist_ok=True)

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    arquivos_rag = _listar_arquivos_rag(BASE_DOCS_DIR)

    base_hash = _hash_completo(arquivos_rag, usar_site)
    hash_antigo = _ler_hash_salvo()

    precisa_recriar = (hash_antigo != base_hash) or (not os.path.isdir(CHROMA_DIR))

    if precisa_recriar:
        _apagar_indice()

        # Documentos locais
        docs = _carregar_documentos_rag(BASE_DOCS_DIR)

        # Documentos do site FleetPro
        if usar_site:
            docs_site = obter_docs_do_site(forcar_recrawl=False)
            docs.extend(docs_site)

        if not docs:
            return None

        chunks = _split_docs(docs)
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
        _salvar_hash(base_hash)
        return vs

    # Carrega índice existente
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vs


def buscar_no_rag(vectorstore, query: str, k: int = 4) -> str:
    if vectorstore is None:
        return ""

    try:
        resultados = vectorstore.similarity_search(query, k=k)
        if not resultados:
            return ""

        trechos = []
        for i, doc in enumerate(resultados, 1):
            fonte = doc.metadata.get("source", "documento")
            tipo = doc.metadata.get("tipo", "")
            if tipo == "site_fleetpro":
                titulo = doc.metadata.get("title", fonte)
                label = f"Site FleetPro – {titulo}"
            else:
                label = os.path.basename(fonte)
            trechos.append(f"[Trecho {i} – {label}]\n{doc.page_content.strip()}")

        return "\n\n".join(trechos)
    except Exception as e:
        return f"(Erro ao buscar no RAG: {e})"


# ======================
# Excel Matriz_FP – Lookup determinístico por PN
# ======================
@st.cache_data(show_spinner="Carregando Matriz FP...")
def carregar_df_fp_matriz(file_path: str, sheet_name: str) -> "pd.DataFrame":
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    for c in COLUNAS_BUSCA_PN:
        if c not in df.columns:
            df[c] = ""
    return df


def formatar_linha_como_lista(df: "pd.DataFrame", row_index_df: int) -> str:
    if row_index_df < 0 or row_index_df >= len(df):
        return "Índice fora do intervalo."
    row = df.iloc[row_index_df]
    partes = []
    for col in df.columns:
        val = row[col]
        if pd.isna(val):
            val = ""
        partes.append(f"- {col}: {str(val).strip()}")
    return "\n".join(partes)


def detectar_busca_por_equipamento(mensagem: str):
    msg_upper = mensagem.upper()
    chaves_ordenadas = sorted(MAPA_PALAVRAS_COLUNAS.keys(), key=lambda x: -len(x))
    colunas_encontradas = []
    for chave in chaves_ordenadas:
        if chave in msg_upper:
            for col in MAPA_PALAVRAS_COLUNAS[chave]:
                if col not in colunas_encontradas:
                    colunas_encontradas.append(col)
    return colunas_encontradas


def detectar_busca_marketing(mensagem: str):
    """
    Detecta se a mensagem é uma consulta por categoria/tipo de produto.
    Normaliza acentos para comparação e retorna o termo EXATO da coluna
    MARKETING PROJECT (ou o sinônimo detectado), ou None se não encontrar.

    Exemplos:
        "quais rolamentos FleetPro vocês têm?"    → "ROLAMENTOS"
        "preciso de correias para minha colheitadeira" → "CORREIAS"
        "tem filtro disponível?"                  → "FILTROS"
        "undercarriage disponível?"               → "UNDERCARRIAGE"
    """
    import unicodedata

    def normalizar(s: str) -> str:
        """Remove acentos e converte para maiúsculo."""
        return unicodedata.normalize("NFD", s.upper()).encode("ascii", "ignore").decode("ascii")

    msg_norm = normalizar(mensagem)

    # Ordena: mais longo primeiro (evita "BELT" antes de "BELTS", "CORREIA" antes de "CORREIAS")
    termos_ordenados = sorted(PALAVRAS_MARKETING, key=lambda x: -len(x))

    for termo in termos_ordenados:
        termo_norm = normalizar(termo)
        # Usa \b apenas para termos puramente alfanuméricos; para termos com & ou espaço, usa in
        if re.search(r"[^A-Z0-9]", termo_norm):
            # Termo contém espaço, & ou outro caractere especial → busca direta
            if termo_norm in msg_norm:
                return termo
        else:
            # Termo simples → busca como palavra inteira
            if re.search(r"\b" + re.escape(termo_norm) + r"\b", msg_norm):
                return termo

    return None


def _extrair_palavras_extras(mensagem: str, termo_mkt: str) -> list:
    """
    Extrai tokens da mensagem que não são o termo de marketing nem palavras genéricas.
    Permite refinar a busca na DESCRIPTION (ex: "rolamento 6205" → ["6205"]).
    """
    stopwords_busca = {
        "QUAIS", "QUAL", "PN", "PNS", "PARA", "NO", "NA", "OS", "AS", "DE",
        "DO", "DA", "UM", "UMA", "QUE", "SAO", "SÃO", "ME", "MOSTRE", "MOSTRA",
        "LISTA", "LISTAR", "VER", "TENHO", "PRECISO", "QUERO", "TEM", "FLEETPRO",
        "DISPONIVEIS", "DISPONÍVEIS", "VOCÊS", "VOCES", "TEMOS", "HAI", "HÁ",
        "EXISTE", "EXISTEM", "BUSCA", "BUSCAR", "ENCONTRA", "ACHAR", "THE", "AND",
        "FOR", "WITH", "HAVE", "SHOW", "FIND", "GET", "ALL",
    } | {termo_mkt}
    tokens = re.findall(r"[A-Za-z0-9]+", mensagem.upper())
    extras = [
        t for t in tokens
        if t not in stopwords_busca
        and len(t) >= 3
        and t not in {p.upper() for p in PALAVRAS_MARKETING}
    ]
    return list(dict.fromkeys(extras))


def buscar_por_marketing(
    df: "pd.DataFrame",
    termo: str,
    mensagem: str,
    max_resultados: int = 100,
) -> str:
    """
    Busca linhas onde MARKETING PROJECT contém o termo (parcial, case-insensitive).
    Resolve sinônimos para o valor exato da coluna antes de buscar.
    Aplica filtro extra na DESCRIPTION se o usuário informou termos adicionais.
    Retorna markdown formatado.
    """
    import unicodedata

    def norm_sem_acento(s: str) -> str:
        return unicodedata.normalize("NFD", s.upper()).encode("ascii", "ignore").decode("ascii")

    # Resolve sinônimo → valor exato da coluna (ex: "ROLAMENTO" → "ROLAMENTOS")
    termo_upper = termo.upper()
    termo_busca = MAPA_SINONIMOS_MARKETING.get(termo_upper, termo_upper)
    # Tenta também sem acento como fallback
    if termo_busca == termo_upper:
        termo_busca = MAPA_SINONIMOS_MARKETING.get(norm_sem_acento(termo_upper), termo_upper)

    if COLUNA_MARKETING not in df.columns:
        return (
            f"⚠️ A coluna `{COLUNA_MARKETING}` não foi encontrada na planilha. "
            "Verifique se o nome está correto."
        )

    mask_mkt = (
        df[COLUNA_MARKETING].notna()
        & df[COLUNA_MARKETING].astype(str).str.upper().str.contains(
            re.escape(termo_busca), na=False
        )
    )
    df_filtrado = df[mask_mkt].copy()

    if df_filtrado.empty:
        # Tenta busca parcial sem acento como último recurso
        mask_fallback = (
            df[COLUNA_MARKETING].notna()
            & df[COLUNA_MARKETING].astype(str).apply(norm_sem_acento).str.contains(
                re.escape(norm_sem_acento(termo_busca)), na=False
            )
        )
        df_filtrado = df[mask_fallback].copy()

    if df_filtrado.empty:
        return (
            f"Não encontrei nenhum produto na categoria **{termo_busca}** na matriz FP.\n\n"
            f"Categorias disponíveis: {', '.join(f'`{v}`' for v in sorted(set(MAPA_SINONIMOS_MARKETING.values())))}"
        )

    # Filtro secundário: termos extras na DESCRIPTION
    palavras_extra = _extrair_palavras_extras(mensagem, termo)
    aplicou_filtro_extra = False
    if palavras_extra and "DESCRIPTION" in df_filtrado.columns:
        pattern_extra = "|".join(re.escape(p) for p in palavras_extra)
        mask_desc = df_filtrado["DESCRIPTION"].astype(str).str.upper().str.contains(
            pattern_extra, na=False
        )
        if mask_desc.any():
            df_filtrado = df_filtrado[mask_desc]
            aplicou_filtro_extra = True

    truncado = False
    if len(df_filtrado) > max_resultados:
        df_filtrado = df_filtrado.head(max_resultados)
        truncado = True

    total = len(df_filtrado)
    sufixo_trunc = f" *(limitado aos primeiros {max_resultados})*" if truncado else ""
    subtitulo_extra = (
        f" — filtro adicional: `{'`, `'.join(palavras_extra)}`"
        if aplicou_filtro_extra
        else ""
    )

    linhas = [
        f"## 🔍 {COLUNA_MARKETING}: **{termo_busca}** — {total} produto(s){subtitulo_extra}{sufixo_trunc}\n"
    ]

    for _, row in df_filtrado.iterrows():
        pn_fp   = str(row.get("PN FLEETPRO", "") or "").strip()
        pn_gen  = str(row.get("PN GEN", "")  or "").strip()
        descr   = str(row.get("DESCRIPTION", "") or "").strip()
        mkt_val = str(row.get(COLUNA_MARKETING, "") or "").strip()

        partes = [f"- **{descr}**" if descr else "- *(sem descrição)*"]
        if mkt_val and mkt_val not in ("-", ""):
            partes.append(f"  Categoria: `{mkt_val}`")
        if pn_fp and pn_fp not in ("-", ""):
            partes.append(f"  PN FleetPro: `{pn_fp}`")
        if pn_gen and pn_gen not in ("-", ""):
            partes.append(f"  PN GEN: `{pn_gen}`")

        linhas.append("\n".join(partes))

    linhas.append("")
    return "\n".join(linhas)


def buscar_por_equipamento(df: "pd.DataFrame", mensagem: str, colunas_alvo: list, max_resultados: int = 100) -> str:
    import re as _re

    modelo_filtro = None
    tokens = _re.findall(r'[A-Za-z0-9]+', mensagem.upper())
    palavras_genericas = {
        'QUAIS', 'QUAL', 'PN', 'PNS', 'PARA', 'NO', 'NA', 'OS', 'AS', 'DE',
        'DO', 'DA', 'UM', 'UMA', 'QUE', 'SAO', 'SÃO', 'POSSO', 'USAR',
        'UTILIZAR', 'LISTA', 'LISTAR', 'MOSTRAR', 'MOSTRAME', 'QUERO',
        'VER', 'EQUIPAMENTO', 'MAQUINA', 'JOHN', 'DEERE', 'CASE', 'NHAG',
        'MASSEY', 'FERGUSON', 'VALTRA', 'AGCO', 'MACDON', 'IDEAL',
        'TRATOR', 'TRACTOR', 'COMBINE', 'COLHEITADEIRA', 'HEADER',
        'PLATAFORMA', 'PULVERIZADOR', 'SPRAYER', 'PLANTADEIRA', 'PLANTER',
        'NEW', 'HOLLAND', 'COMPATIVEIS', 'COMPATÍVEIS', 'IH', 'ME', 'MOSTRE',
    }
    candidatos_modelo = [t for t in tokens if t not in palavras_genericas and len(t) >= 3]
    if candidatos_modelo:
        modelo_filtro = max(candidatos_modelo, key=len)

    resultados_por_coluna = {}

    for col in colunas_alvo:
        if col not in df.columns:
            continue

        mask = df[col].notna() & ~df[col].isin(['-', '', 'NaN', 'nan'])
        df_col = df[mask].copy()

        if df_col.empty:
            continue

        if modelo_filtro and col in COLUNAS_MODELOS_EQUIP:
            mask_modelo = df_col[col].str.upper().str.contains(modelo_filtro, na=False)
            if mask_modelo.any():
                df_col = df_col[mask_modelo]

        if len(df_col) > max_resultados:
            df_col = df_col.head(max_resultados)
            truncado = True
        else:
            truncado = False

        resultados_por_coluna[col] = (df_col, truncado)

    if not resultados_por_coluna:
        marcas = ", ".join(colunas_alvo)
        return f"Não encontrei peças para as colunas: {marcas}."

    linhas_output = []
    total_itens = 0

    for col, (df_resultado, truncado) in resultados_por_coluna.items():
        e_coluna_marca = col in COLUNAS_MARCAS_PN
        sufixo = f"  *(limitado aos primeiros {max_resultados})*" if truncado else ""
        linhas_output.append(f"## 🔧 {col} — {len(df_resultado)} peça(s){sufixo}\n")

        for _, row in df_resultado.iterrows():
            pn_gen = str(row.get('PN GEN', '') or '').strip()
            pn_fp = str(row.get('PN FLEETPRO', '') or '').strip()
            descricao = str(row.get('DESCRIPTION', '') or '').strip()
            val_col = str(row.get(col, '') or '').strip()

            partes_linha = [f"- **{descricao}**"]
            if pn_fp and pn_fp not in ('-', ''):
                partes_linha.append(f"  PN FleetPro: `{pn_fp}`")
            if pn_gen and pn_gen not in ('-', ''):
                partes_linha.append(f"  PN GEN: `{pn_gen}`")
            if e_coluna_marca and val_col and val_col not in ('-', ''):
                partes_linha.append(f"  PN {col}: `{val_col}`")
            elif not e_coluna_marca and val_col and val_col not in ('-', ''):
                partes_linha.append(f"  Modelos: {val_col}")

            linhas_output.append("\n".join(partes_linha))
            total_itens += 1

        linhas_output.append("")

    header = f"Encontrei **{total_itens}** peça(s) compatível(is):\n"
    if modelo_filtro and any(col in COLUNAS_MODELOS_EQUIP for col in colunas_alvo):
        header = f"Encontrei **{total_itens}** peça(s) compatível(is) (filtro de modelo: `{modelo_filtro}`):\n"

    return header + "\n".join(linhas_output)


def _buscar_pn_no_df(df: "pd.DataFrame", pn_norm: str, max_resultados: int) -> "pd.DataFrame":
    df_busca = df[COLUNAS_BUSCA_PN].copy()
    for c in COLUNAS_BUSCA_PN:
        df_busca[c] = df_busca[c].apply(norm_pn)

    mask = False
    for c in COLUNAS_BUSCA_PN:
        mask = mask | (df_busca[c] == pn_norm)

    encontrados = df[mask]
    if len(encontrados) > max_resultados:
        encontrados = encontrados.head(max_resultados)
    return encontrados


def _formatar_resultados(df: "pd.DataFrame", encontrados: "pd.DataFrame", pn_exibido: str) -> str:
    partes = [f"Encontrei **{len(encontrados)}** opção(ões) para o PN `{pn_exibido}`:\n"]
    for i, (idx_df, _row) in enumerate(encontrados.iterrows(), start=1):
        row_index_df = df.index.get_loc(idx_df)
        partes.append(f"### Opção {i}\n")
        partes.append(formatar_linha_como_lista(df, int(row_index_df)))
        partes.append("")
    return "\n".join(partes)


def procurar_pn(df: "pd.DataFrame", mensagem_usuario: str, max_resultados: int = 50) -> str:
    if not mensagem_usuario or not mensagem_usuario.strip():
        return "Informe um PN válido para pesquisa."

    pn_direto = norm_pn(mensagem_usuario)
    if pn_direto:
        encontrados = _buscar_pn_no_df(df, pn_direto, max_resultados)
        if not encontrados.empty:
            return _formatar_resultados(df, encontrados, mensagem_usuario.strip())

    candidatos = extrair_pns_da_mensagem(mensagem_usuario)

    for candidato in candidatos:
        encontrados = _buscar_pn_no_df(df, candidato, max_resultados)
        if not encontrados.empty:
            return _formatar_resultados(df, encontrados, candidato)

    if candidatos:
        tentados = ", ".join(f"`{c}`" for c in candidatos[:5])
        return (
            f"Não encontrei nenhum dos seguintes PNs na matriz: {tentados}.\n\n"
            "Verifique se o número está correto ou tente digitar apenas o código da peça."
        )

    return ""


# ======================
# Inicializar LLM
# ======================
def inicializar_FleetPro(provedor: str, modelo: str, api_key: str):
    api_key = (api_key or "").strip()
    if not api_key:
        st.error("Informe a API key antes de inicializar.")
        return

    chat_cls = CONFIG_MODELOS[provedor]["chat"]
    chat = chat_cls(model=modelo, api_key=api_key)

    st.session_state["chat"] = chat
    st.session_state["provedor"] = provedor
    st.session_state["modelo"] = modelo
    st.session_state["api_key_openai_rag"] = api_key if provedor == "OpenAI" else st.session_state.get("api_key_openai_rag", "")
    st.session_state.setdefault("memoria", ConversationBufferMemory())

    obter_vectorstore.clear()

    st.success(f"Agente FleetPro inicializado: {provedor} / {modelo}")


# ======================
# UI – Chat principal
# ======================
def pagina_chat():
    col_logo, col_titulo = st.columns([1, 6])
    with col_logo:
        st.image("base_docs/fleetpro_logo.png", width=120)
    with col_titulo:
        st.header("🕵️‍♂️ FleetPro Expert 🛠️", divider=True)

    chat_model = st.session_state.get("chat")
    memoria: ConversationBufferMemory = st.session_state.get("memoria", ConversationBufferMemory())

    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    input_usuario = st.chat_input(
        "Digite um PN para busca, ou uma pergunta sobre produtos, objeções e recomendações..."
    )

    if not input_usuario:
        return

    # ── Captura escolha de perfil (ANTES de qualquer busca ou LLM) ───────
    if st.session_state.get("perfil_usuario") is None:
        resposta_lower = input_usuario.lower().strip()
        if any(p in resposta_lower for p in ["1", "vendedor", "vendo", "balcão", "balcao", "revend"]):
            st.session_state["perfil_usuario"] = "vendedor"
            st.chat_message("human").markdown(input_usuario)
            with st.chat_message("ai"):
                resposta = "✅ Perfeito! Modo **Vendedor de Balcão** ativado. Pode fazer sua pergunta!"
                st.markdown(resposta)
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta)
            st.session_state["memoria"] = memoria
            st.stop()
        elif any(p in resposta_lower for p in ["2", "usuario", "usuário", "uso", "minha máquina", "minha maquina", "agricultor", "operador"]):
            st.session_state["perfil_usuario"] = "usuario"
            st.chat_message("human").markdown(input_usuario)
            with st.chat_message("ai"):
                resposta = "✅ Perfeito! Modo **Usuário da Peça** ativado. Pode fazer sua pergunta!"
                st.markdown(resposta)
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta)
            st.session_state["memoria"] = memoria
            st.stop()

    st.chat_message("human").markdown(input_usuario)

    with st.chat_message("ai"):
        try:
            usar_fp_matriz = st.session_state.get("usar_fp_matriz", True)
            usar_rag = st.session_state.get("usar_rag", True)
            usar_site = st.session_state.get("usar_site_fleetpro", True)
            max_resultados = st.session_state.get("max_resultados_fp", 50)

            # ── 1. Lookup no Excel ────────────────────────────────────────────
            resultado_matriz = ""
            if usar_fp_matriz:
                excel_path = os.path.join(BASE_DOCS_DIR, MATRIX_EXCEL)
                if not os.path.exists(excel_path):
                    st.error(
                        f"{MATRIX_EXCEL} não encontrado em base_docs/. "
                        "Coloque o arquivo lá e reinicie o app."
                    )
                    st.stop()
                df_fp = carregar_df_fp_matriz(excel_path, SHEET_FP_MATRIZ)

                colunas_equip = detectar_busca_por_equipamento(input_usuario)
                termo_marketing = detectar_busca_marketing(input_usuario)

                if colunas_equip:
                    # Busca por fabricante / tipo de máquina
                    resultado_matriz = buscar_por_equipamento(df_fp, input_usuario, colunas_equip, max_resultados)
                elif termo_marketing:
                    # Busca por categoria de produto na coluna MARKETING PROJECT
                    resultado_matriz = buscar_por_marketing(df_fp, termo_marketing, input_usuario, max_resultados)
                else:
                    resultado_matriz = procurar_pn(df_fp, input_usuario, max_resultados)

            # ── 2. Busca RAG (documentos locais + site FleetPro) ──────────────
            contexto_rag = ""
            if usar_rag and chat_model is not None:
                api_key_openai = st.session_state.get("api_key_openai_rag", "")
                if api_key_openai:
                    try:
                        vs = obter_vectorstore(usar_site=usar_site)
                        contexto_rag = buscar_no_rag(vs, input_usuario)
                    except Exception as e:
                        contexto_rag = f"(Erro ao acessar RAG: {e})"

            # ── 3. Montar resposta ────────────────────────────────────────────
            if usar_fp_matriz and not usar_rag and chat_model is None:
                st.markdown(resultado_matriz)
                resposta = resultado_matriz

            elif chat_model is not None:
                blocos = []

                if resultado_matriz:
                    blocos.append(
                        "## Resultado da Busca na Matriz FP (dados do Excel)\n\n"
                        + resultado_matriz
                    )

                if contexto_rag:
                    blocos.append(
                        "## Conhecimento Adicional (guia de objeções / recomendações / site FleetPro)\n\n"
                        + contexto_rag
                    )

                if blocos:
                    contexto_completo = "\n\n---\n\n".join(blocos)
                    perfil = st.session_state.get("perfil_usuario")

                    if perfil is None:
                        st.session_state["pergunta_pendente"] = input_usuario
                        st.session_state["contexto_pendente"] = contexto_completo
                        resposta = (
                            "Antes de responder, preciso entender melhor como posso te ajudar! 😊\n\n"
                            "**Você é:**\n"
                            "- **1️⃣ Vendedor** — quero argumentos para atender meu cliente no balcão\n"
                            "- **2️⃣ Usuário da peça** — quero saber se o FleetPro é a melhor opção para minha máquina\n\n"
                            "_Digite o número ou o nome da opção._"
                        )
                        st.markdown(resposta)
                        memoria.chat_memory.add_user_message(input_usuario)
                        memoria.chat_memory.add_ai_message(resposta)
                        st.session_state["memoria"] = memoria
                        st.stop()

                    elif perfil == "vendedor":
                        prompt = (
                            f"Você é o FleetPro Expert, um assistente de vendas especializado em apoiar vendedores de balcão "
                            f"na comercialização de peças de reposição FleetPro para máquinas agrícolas.\n\n"
                            f"Seu papel é ajudar o VENDEDOR a:\n"
                            f"- Identificar rapidamente o PN FleetPro correto para o cliente\n"
                            f"- Ter argumentos técnicos e comerciais prontos para contornar objeções\n"
                            f"- Sempre oferecer a possibilidade de outros itens em conjunto com a demanda (Cross Selling)\n"
                            f"- Explicar as vantagens do FleetPro frente à peça original ou concorrente\n"
                            f"- Transmitir confiança e credibilidade na indicação do produto\n\n"
                            f"Tom das respostas:\n"
                            f"- Direto e objetivo — o vendedor está atendendo um cliente no balcão agora\n"
                            f"- Técnico quando necessário, mas sempre em linguagem acessível\n"
                            f"- Forneça argumentos prontos que o vendedor pode usar na hora\n"
                            f"- Se houver objeção de preço ou qualidade, sugira como rebater\n\n"
                            f"O vendedor perguntou: **{input_usuario}**\n\n"
                            f"Use as informações abaixo para responder com agilidade e precisão, "
                            f"priorizando o PN FleetPro e os argumentos que ajudem a fechar a venda:\n\n"
                            f"{contexto_completo}"
                        )
                        resposta_stream = st.write_stream(chat_model.stream(prompt))
                        resposta = resposta_stream

                    elif perfil == "usuario":
                        prompt = (
                            f"Você é o FleetPro Expert, um consultor especialista em peças de reposição para máquinas agrícolas.\n\n"
                            f"Você está falando DIRETAMENTE com o agricultor ou operador da máquina. Seu objetivo é:\n"
                            f"- Mostrar que o FleetPro é a melhor escolha para a máquina dele\n"
                            f"- Destacar que a qualidade é equivalente ou superior à peça original\n"
                            f"- Ressaltar economia de custo sem abrir mão de desempenho\n"
                            f"- Transmitir segurança e confiança na marca FleetPro\n"
                            f"- Incentivar o cliente a pedir o FleetPro ao seu revendedor\n\n"
                            f"Tom das respostas:\n"
                            f"- Próximo, empático e direto — fale como um especialista de confiança\n"
                            f"- Valorize a economia e a produtividade que o FleetPro proporciona\n"
                            f"- Use exemplos práticos do dia a dia do campo quando possível\n"
                            f"- Finalize sempre incentivando a buscar o FleetPro no revendedor mais próximo\n\n"
                            f"O usuário perguntou: **{input_usuario}**\n\n"
                            f"Use as informações abaixo para responder de forma clara e persuasiva, "
                            f"reforçando os benefícios do FleetPro para quem usa a peça no campo:\n\n"
                            f"{contexto_completo}"
                        )
                        resposta_stream = st.write_stream(chat_model.stream(prompt))
                        resposta = resposta_stream

                else:
                    prompt = input_usuario
                    resposta_stream = st.write_stream(chat_model.stream(prompt))
                    resposta = resposta_stream

            else:
                st.markdown(resultado_matriz)
                resposta = resultado_matriz

        except Exception as e:
            st.error("Erro ao responder.")
            st.exception(e)
            st.stop()

    memoria.chat_memory.add_user_message(input_usuario)
    memoria.chat_memory.add_ai_message(resposta)
    st.session_state["memoria"] = memoria


# ======================
# UI – Barra lateral
# ======================
def sidebar():
    st.title("⚙️ Configurações")
    st.image("base_docs/cnh_logo.png", width=180)

    # ── Seção 1: Modelo de linguagem (LLM) ──────────────────────────────────
    with st.expander("🤖 Modelo de Linguagem (LLM)", expanded=True):
        st.caption(
            "O LLM combina os resultados da busca no Excel com o conhecimento dos documentos "
            "e gera uma resposta enriquecida. Se não inicializado, o app retorna apenas os dados brutos da matriz."
        )

        provedor = st.selectbox("Provedor", list(CONFIG_MODELOS.keys()), key="sel_provedor")
        modelo = st.selectbox("Modelo", CONFIG_MODELOS[provedor]["modelos"], key="sel_modelo")

        api_key = st.text_input(
            f"API key ({provedor})",
            value=st.session_state.get(f"api_key_{provedor}", ""),
            type="password",
            key="input_api_key_llm",
        )
        st.session_state[f"api_key_{provedor}"] = api_key

        if provedor == "OpenAI" and api_key:
            st.session_state["api_key_openai_rag"] = api_key

        if st.button("🚀 Inicializar FleetPro_Expert", use_container_width=True):
            inicializar_FleetPro(provedor, modelo, api_key)

        st.divider()
        chat = st.session_state.get("chat")
        if chat:
            st.success(
                f"✅ Ativo: {st.session_state.get('provedor')} / {st.session_state.get('modelo')}"
            )
        else:
            st.warning("⚠️ Nenhum modelo inicializado. O chat usará apenas o lookup direto.")

    # ── Seção 2: FP MATRIZ ───────────────────────────────────────────────────
    with st.expander("📊 Lookup FP MATRIZ (Excel)", expanded=False):
        usar_fp_matriz = st.toggle(
            "Buscar PN na Matriz FP",
            value=st.session_state.get("usar_fp_matriz", True),
        )
        st.session_state["usar_fp_matriz"] = usar_fp_matriz

        max_resultados_fp = st.slider(
            "Máximo de resultados retornados",
            min_value=5, max_value=200,
            value=st.session_state.get("max_resultados_fp", 50),
        )
        st.session_state["max_resultados_fp"] = max_resultados_fp

        st.caption(
            "A busca normaliza o PN (remove espaços, hífens e pontos; maiúsculo) "
            "e compara com PN GEN, PN ALTERNATIVE, PN NXP e PN FLEETPRO."
        )

    # ── Seção 3: RAG / Documentos de conhecimento ────────────────────────────
    with st.expander("📂 Documentos de Conhecimento (RAG)", expanded=False):
        st.info(
            "Coloque seus documentos (.pdf, .txt, .csv) na pasta `base_docs/`. "
            "O sistema irá indexá-los e usará o conteúdo para ajudar com objeções, "
            "recomendações e dúvidas técnicas."
        )

        usar_rag = st.toggle(
            "Usar documentos de conhecimento no chat",
            value=st.session_state.get("usar_rag", True),
        )
        st.session_state["usar_rag"] = usar_rag

        st.caption("✅ Embeddings locais (sentence-transformers) — não requer API key adicional.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reindexar", use_container_width=True):
                api_key_rag = st.session_state.get("api_key_openai_rag", "")
                if not api_key_rag:
                    st.error("Informe a API key da OpenAI.")
                else:
                    obter_vectorstore.clear()
                    _apagar_indice()
                    try:
                        usar_site = st.session_state.get("usar_site_fleetpro", True)
                        vs = obter_vectorstore(usar_site=usar_site)
                        if vs:
                            st.success("Indexado com sucesso!")
                        else:
                            st.warning("Nenhum documento encontrado.")
                    except Exception as e:
                        st.error(f"Erro: {e}")

        with col2:
            if st.button("📋 Ver docs", use_container_width=True):
                arquivos = _listar_arquivos_rag(BASE_DOCS_DIR)
                if arquivos:
                    st.success(f"{len(arquivos)} arquivo(s):")
                    for a in arquivos:
                        st.caption(f"• {os.path.basename(a)}")
                else:
                    st.warning("Nenhum documento local encontrado.")

    # ── Seção 4: Site FleetPro ───────────────────────────────────────────────
    with st.expander("🌐 Site FleetPro (fleetpro.com.br)", expanded=False):
        st.info(
            f"O sistema pode indexar automaticamente o conteúdo de **{FLEETPRO_SITE_URL}** "
            "e todas as suas subpáginas de produtos, enriquecendo o RAG com informações "
            "atualizadas sobre produtos, descrições e especificações."
        )

        usar_site = st.toggle(
            "Incluir site FleetPro no RAG",
            value=st.session_state.get("usar_site_fleetpro", True),
        )
        st.session_state["usar_site_fleetpro"] = usar_site

        # Status do cache
        cache = _carregar_cache_site()
        if cache.get("total_paginas"):
            st.success(f"📦 Cache: {cache['total_paginas']} página(s) indexadas do site.")
        else:
            st.warning("Cache do site vazio. Clique em 'Crawlear Agora' para coletar.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("🕷️ Crawlear Agora", use_container_width=True, disabled=not usar_site):
                api_key_rag = st.session_state.get("api_key_openai_rag", "")
                if not api_key_rag:
                    st.error("Informe a API key da OpenAI primeiro.")
                else:
                    # Força re-crawl e reindexa tudo
                    obter_docs_do_site(forcar_recrawl=True)
                    obter_vectorstore.clear()
                    _apagar_indice()
                    try:
                        vs = obter_vectorstore(usar_site=True)
                        if vs:
                            st.success("Site indexado com sucesso!")
                        else:
                            st.warning("Nenhum conteúdo foi indexado.")
                    except Exception as e:
                        st.error(f"Erro: {e}")

        with col_s2:
            if st.button("🗑️ Limpar Cache", use_container_width=True):
                if os.path.exists(FLEETPRO_SITE_CACHE):
                    os.remove(FLEETPRO_SITE_CACHE)
                    obter_vectorstore.clear()
                    _apagar_indice()
                    st.success("Cache do site removido.")
                else:
                    st.info("Nenhum cache para remover.")

        st.caption(
            "O crawling é feito apenas uma vez e salvo em cache local. "
            "Use 'Crawlear Agora' para atualizar quando o site for modificado."
        )


# ======================
# Main
# ======================
def main():
    pagina_chat()
    with st.sidebar:
        sidebar()


if __name__ == "__main__":
    safe_run(main)