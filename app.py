"""Interface única: pergunta, resposta e evidências."""
from html import escape

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from core import ask, connect, path_text

EXAMPLES = [
    "Quais materiais são utilizados na obra Residencial Horizonte?",
    "Qual fornecedor fornece cimento para a obra?",
    "Quais etapas da construção podem ser afetadas pela falta de cimento?",
]


def graph_html(paths):
    graph = Network(height="420px", width="100%", directed=True, cdn_resources="in_line")
    nodes, edges = set(), set()
    for path in paths:
        for node in path["nodes"]:
            if node["id"] not in nodes:
                graph.add_node(node["id"], label=escape(node["nome"]), group=node["tipo"],
                               title=escape(node["tipo"]))
                nodes.add(node["id"])
        for edge in path["edges"]:
            key = (edge["source"], edge["target"], edge["tipo"])
            if key not in edges:
                graph.add_edge(key[0], key[1], label=key[2], title=escape(", ".join(edge["fontes"])))
                edges.add(key)
    return graph.generate_html()


def main():
    st.set_page_config(page_title="GraphRAG — Construção Civil", layout="wide")
    st.title("GraphRAG — Construção Civil")
    st.caption("MVP acadêmico • dados inteiramente fictícios")
    st.write("Exemplos:")
    for example in EXAMPLES:
        st.write(f"• {example}")
    with st.form("consulta"):
        question = st.text_input("Faça sua pergunta sobre a obra.", value=EXAMPLES[0], max_chars=500)
        submitted = st.form_submit_button("Consultar")
    if submitted:
        try:
            with st.spinner("Consultando documentos e caminhos no Neo4j…"):
                with connect() as driver:
                    result = ask(driver, question)
        except Exception as exc:
            st.error(f"Consulta não concluída: {type(exc).__name__}: {exc}")
            st.info("Verifique .env, conexão com Neo4j e execução de python ingest.py. Não houve resposta simulada.")
            return
        st.subheader("Resposta fundamentada")
        st.write(result["answer"])
        st.caption("Geração restrita: o LLM seleciona evidências; frases são construídas com os fatos dos caminhos.")
        st.info("Confiança: " + result["confidence"] + ". Indicador de rastreabilidade, não probabilidade calibrada nem garantia de verdade.")
        if result["paths"]:
            st.subheader("Subgrafo usado na resposta")
            components.html(graph_html(result["paths"]), height=440)
        with st.expander("Nós consultados e recuperação híbrida"):
            st.write(result["nodes"])
            st.write("Documentos recuperados (vetor + Fulltext, combinados por RRF):", result["retrieved"])
        with st.expander("Algoritmo: menor caminho estrutural"):
            for path in result["shortest"]:
                st.write(path_text(path))
            st.caption("Usado como contexto adicional pelo LLM. A análise de impacto segue DEPENDE_DE na direção indicada.")
        with st.expander("Consultas Cypher executadas"):
            for item in result["trace"]:
                st.code(item["cypher"], language="cypher")
                st.json(item["parametros"])
        with st.expander("Documentos utilizados como fonte"):
            for source in result["sources"]:
                st.write(source["id"])
                st.text(source["texto"])


if __name__ == "__main__":
    main()
