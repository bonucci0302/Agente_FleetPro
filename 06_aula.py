import tempfile

import streamlit as st
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from loaders import *

TIPOS_ARQUIVOS_VALIDOS = ["Site", "Youtube", "Pdf", "Csv", "Txt"]

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

MEMORIA_PADRAO = ConversationBufferMemory()


def _validar_entrada(tipo_arquivo, arquivo):
    # Para URL (Site/Youtube): precisa ser string não vazia
    if tipo_arquivo in ["Site", "Youtube"]:
        if not arquivo or not str(arquivo).strip():
            st.error("Informe a URL antes de inicializar.")
            return False

    # Para upload (Pdf/Csv/Txt): precisa ter arquivo
    if tipo_arquivo in ["Pdf", "Csv", "Txt"]:
        if arquivo is None:
            st.error("Faça upload do arquivo antes de inicializar.")
            return False

    return True


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    # Validações básicas
    if not api_key or not str(api_key).strip():
        st.error("Informe a API key antes de inicializar.")
        return

    if not _validar_entrada(tipo_arquivo, arquivo):
        return

    # Carrega documento (mantido como você fez, apesar de ainda não ser usado no chat)
    if tipo_arquivo == "Site":
        documento = carrega_site(arquivo)

    elif tipo_arquivo == "Youtube":
        documento = carrega_youtube(arquivo)

    elif tipo_arquivo == "Pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_pdf(nome_temp)

    elif tipo_arquivo == "Csv":
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_csv(nome_temp)

    elif tipo_arquivo == "Txt":
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write(arquivo.read())
            nome_temp = temp.name
        documento = carrega_txt(nome_temp)

    else:
        st.error("Tipo de arquivo inválido.")
        return

    # Cria o chat model
    chat = CONFIG_MODELOS[provedor]["chat"](model=modelo, api_key=api_key)

    # Salva no session_state
    st.session_state["chat"] = chat
    st.session_state["documento"] = documento  # opcional, mas útil para evoluir depois

    # Garante que existe memória no session_state
    st.session_state.setdefault("memoria", MEMORIA_PADRAO)

    st.success("Oráculo inicializado com sucesso.")


def pagina_chat():
    st.header("🕵️‍♂️Bem-vindo ao Oráculo FleetPro🛠️", divider=True)

    chat_model = st.session_state.get("chat")
    memoria = st.session_state.get("memoria", MEMORIA_PADRAO)

    # Renderiza histórico
    for mensagem in memoria.buffer_as_messages:
        chat_msg = st.chat_message(mensagem.type)
        chat_msg.markdown(mensagem.content)

    # Input do usuário
    input_usuario = st.chat_input("Fale com o oráculo FleetPro")

    if input_usuario:
        # Se usuário mandar mensagem antes de inicializar, não quebra
        if chat_model is None:
            st.info("Clique em 'Inicializar Oráculo' na barra lateral para carregar o modelo.")
            st.stop()

        chat_human = st.chat_message("human")
        chat_human.markdown(input_usuario)

        chat_ai = st.chat_message("ai")
        resposta = chat_ai.write_stream(chat_model.stream(input_usuario))

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state["memoria"] = memoria


def sidebar():
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos"])

    with tabs[0]:
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)

        arquivo = None
        if tipo_arquivo == "Site":
            arquivo = st.text_input("Digite a url do site")

        elif tipo_arquivo == "Youtube":
            arquivo = st.text_input("Digite a url do vídeo")

        elif tipo_arquivo == "Pdf":
            arquivo = st.file_uploader("Faça o upload do arquivo pdf", type=["pdf"])

        elif tipo_arquivo == "Csv":
            arquivo = st.file_uploader("Faça o upload do arquivo csv", type=["csv"])

        elif tipo_arquivo == "Txt":
            arquivo = st.file_uploader("Faça o upload do arquivo txt", type=["txt"])

    with tabs[1]:
        provedor = st.selectbox("Selecione o provedor dos modelo", CONFIG_MODELOS.keys())
        modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])

        api_key = st.text_input(
            f"Adicione a api key para o provedor {provedor}",
            value=st.session_state.get(f"api_key_{provedor}", ""),
            type="password",
        )
        st.session_state[f"api_key_{provedor}"] = api_key

    if st.button("Inicializar Oráculo", use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)


def main():
    pagina_chat()
    with st.sidebar:
        sidebar()


if __name__ == "__main__":
    main()