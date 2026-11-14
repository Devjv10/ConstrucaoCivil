"""Unitários usam dublês explícitos; integração real exige RUN_INTEGRATION=1."""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

import core
import ingest
from app import EXAMPLES, graph_html

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def sample_path():
    return {"nodes": [
        {"id": "Obra:residencial horizonte", "nome": "Residencial Horizonte", "tipo": "Obra"},
        {"id": "Material:cimento", "nome": "cimento", "tipo": "Material"}],
        "edges": [{"source": "Obra:residencial horizonte", "target": "Material:cimento",
                   "tipo": "UTILIZA", "fontes": ["01_obra.txt"]}]}


class UnitTests(unittest.TestCase):
    def test_remote_api_is_blocked(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://api.openai.com/v1"}), patch("core.urlopen") as send:
            with self.assertRaises(ValueError):
                core.api("/chat/completions", {"model": "qwen2.5:7b"})
            send.assert_not_called()

    def test_cloud_model_is_blocked(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:11434/v1"}), patch("core.urlopen") as send:
            with self.assertRaises(ValueError):
                core.api("/chat/completions", {"model": "example:cloud"})
            send.assert_not_called()

    def test_local_api_does_not_send_key(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:11434/v1", "LLM_API_KEY": "not-used"}), patch("core.urlopen") as send:
            send.return_value.__enter__.return_value.read.return_value = b'{"ok": true}'
            self.assertTrue(core.api("/embeddings", {"model": "embeddinggemma"})["ok"])
            request = send.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/v1/embeddings")
            self.assertIsNone(request.get_header("Authorization"))

    def test_canonicalization(self):
        self.assertEqual(core.entity_id("Material", "  CIMENTO Portland "), "Material:cimento")
        self.assertEqual(core.entity_id("Material", "AÇO"), "Material:aco")

    def test_extraction_and_resolution(self):
        text = "A obra Horizonte utiliza cimento Portland e cimento."
        result = {"entities": [{"label": "Obra", "name": "Horizonte"},
                                {"label": "Material", "name": "cimento Portland"},
                                {"label": "Material", "name": "cimento"}],
                  "relations": [{"source": "Horizonte", "target": "cimento Portland",
                                 "type": "UTILIZA", "quote": text}]}
        nodes, edges = ingest.validate_extraction(result, text)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(edges[0]["target"], "Material:cimento")
        bad = copy.deepcopy(result)
        bad["relations"][0]["quote"] = "Informação inventada"
        with self.assertRaises(ValueError):
            ingest.validate_extraction(bad, text)
        bad = copy.deepcopy(result)
        bad["relations"][0]["type"] = "FORNECE"
        with self.assertRaises(ValueError):
            ingest.validate_extraction(bad, text)

    def test_reject_hallucinated_entity(self):
        with self.assertRaises(ValueError):
            ingest.validate_extraction({"entities": [{"label": "Material", "name": "ouro"}],
                                        "relations": []}, "A obra utiliza cimento.")

    def test_cypher_allowlist(self):
        plan = {"intencao": "materiais", "cypher": core.QUERIES["materiais"],
                "parametros": {"obra": "Residencial Horizonte", "material": ""}}
        self.assertEqual(core.validate_plan(plan)[0], "materiais")
        for injected in ["MATCH (n) DETACH DELETE n", plan["cypher"] + " UNION MATCH (n) RETURN n",
                         "CALL dbms.components()"]:
            with self.assertRaises(ValueError):
                core.validate_plan({**plan, "cypher": injected})

    def test_values_remain_parameters(self):
        plan = {"intencao": "fornecedor", "cypher": core.QUERIES["fornecedor"],
                "parametros": {"obra": "x' RETURN 1 //", "material": "cimento"}}
        _, params = core.validate_plan(plan)
        self.assertNotIn(params["obra"], plan["cypher"])

    def test_rrf_fuses_rankings(self):
        result = core.fuse([{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "c"}])
        self.assertEqual(result[0], "b")
        self.assertEqual(set(result), {"a", "b", "c"})

    def test_grounding_and_refusal(self):
        answer, paths = core.grounded_answer("materiais", [sample_path()], {"caminhos": [0]})
        self.assertIn("utiliza cimento", answer)
        self.assertIn("01_obra.txt", answer)
        self.assertEqual(len(paths), 1)
        answer, paths = core.grounded_answer("materiais", [], {"caminhos": []})
        self.assertIn("não encontrada", answer)
        for ids in [[1], [-1], [True], ["0"]]:
            with self.assertRaises(ValueError):
                core.grounded_answer("materiais", [sample_path()], {"caminhos": ids})

    def test_impact_direction_and_wording(self):
        path = {"nodes": [{"id": "i", "nome": "Instalações", "tipo": "Etapa"},
                          {"id": "a", "nome": "Alvenaria", "tipo": "Etapa"},
                          {"id": "c", "nome": "cimento", "tipo": "Material"}],
                "edges": [{"source": "i", "target": "a", "tipo": "DEPENDE_DE", "fontes": ["04.txt"]},
                          {"source": "a", "target": "c", "tipo": "NECESSITA", "fontes": ["04.txt"]}]}
        answer, _ = core.grounded_answer("impacto", [path], {"caminhos": [0]})
        self.assertIn("Instalações pode ser afetada indiretamente", answer)
        self.assertIn("não comprovam atraso efetivo", answer)

    def test_embedding_dimension_validation(self):
        with patch("core.api", return_value={"data": [{"embedding": [1.0, 0.0]}]}), patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": "3"}):
            with self.assertRaises(ValueError):
                core.embed("cimento")

    def test_html_contains_graph_and_inline_library(self):
        html = graph_html([sample_path()])
        self.assertIn("vis.Network", html)
        self.assertIn("UTILIZA", html)
        self.assertIn("cimento", html)

    def test_interface_initial_render(self):
        app = AppTest.from_file(APP).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "GraphRAG — Construção Civil")
        self.assertEqual(app.button[0].label, "Consultar")

    def test_interface_success_with_explicit_test_double(self):
        result = {"answer": "Residencial Horizonte utiliza cimento.", "paths": [sample_path()],
                  "shortest": [], "sources": [{"id": "01.txt", "texto": "Dados fictícios"}],
                  "trace": [{"cypher": core.QUERIES["materiais"], "parametros": {}}],
                  "confidence": "Evidência documental", "nodes": ["Material:cimento"], "retrieved": ["01.txt"]}
        with patch("core.connect", return_value=MagicMock()), patch("core.ask", return_value=result):
            app = AppTest.from_file(APP).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any("utiliza cimento" in x.value for x in app.markdown))

    def test_interface_reports_connection_error(self):
        with patch("core.connect", side_effect=ValueError("Banco indisponível")):
            app = AppTest.from_file(APP).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("Banco indisponível", app.error[0].value)

    def test_out_of_scope_stops_before_database(self):
        with patch("core.llm", return_value={"intencao": "fora_do_escopo"}), patch("core.run") as run:
            answer = core.ask(None, "Qual é o preço do cimento?")
            self.assertIn("não encontrada", answer["answer"])
            run.assert_not_called()


@unittest.skipUnless(os.getenv("RUN_INTEGRATION") == "1", "Requer Neo4j e Ollama locais; RUN_INTEGRATION=1")
class IntegrationTests(unittest.TestCase):
    def test_real_ingestion(self):
        # Use banco exclusivo do projeto. Este teste grava o dataset usando Ollama local.
        ingest.main()
        with core.connect() as driver:
            counts = core.run(driver, "MATCH (n:Entidade) RETURN count(n) AS total")[0]
            self.assertEqual(counts["total"], 14)
            self.assertEqual(core.run(driver, "MATCH (d:Documento) RETURN count(d) AS total")[0]["total"], 5)
            indexes = core.run(driver, "SHOW INDEXES YIELD name, state RETURN name, state")
            states = {r["name"]: r["state"] for r in indexes}
            for name in ("entity_name", "document_text", "document_vectors"):
                self.assertEqual(states[name], "ONLINE")
            self.assertEqual(core.run(driver, "MATCH (:Entidade)-[r]->(:Entidade) RETURN count(r) AS total")[0]["total"], 30)
            source_counts = {r["source"]: r["total"] for r in core.run(driver,
                "MATCH (:Entidade)-[r]->(:Entidade) UNWIND r.fontes AS source RETURN source, count(*) AS total")}
            self.assertEqual(source_counts, {"01_obra.txt": 10, "02_fornecedores.txt": 5,
                                             "03_fundacao.txt": 7, "04_alvenaria.txt": 5, "05_acabamento.txt": 3})

    def test_three_questions_and_refusals(self):
        with core.connect() as driver:
            expected = [set(["cimento", "areia", "brita", "aço", "tijolo"]),
                        {"Construmax"}, {"Fundação", "Estrutura", "Alvenaria", "Instalações", "Acabamento"}]
            for question, names in zip(EXAMPLES, expected):
                answer = core.ask(driver, question)
                for name in names:
                    self.assertIn(name, answer["answer"])
                self.assertTrue(answer["sources"])
                self.assertTrue(answer["shortest"])
                self.assertTrue(answer["retrieved"])
                self.assertIn("vis.Network", graph_html(answer["paths"]))
            self.assertTrue(any(len(p["edges"]) >= 2 for p in answer["paths"]))
            # Busca textual real, ausência de material e ausência de dados financeiros.
            self.assertTrue(core.run(driver, core.FULLTEXT, {"text": "cimento"}))
            absent = core.ask(driver, "Qual fornecedor fornece vidro para a obra?")
            self.assertIn("não encontrada", absent["answer"])
            outside = core.ask(driver, "Quanto custa o cimento?")
            self.assertIn("não encontrada", outside["answer"])


if __name__ == "__main__":
    unittest.main()
