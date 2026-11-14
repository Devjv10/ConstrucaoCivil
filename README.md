# GraphRAG — Construção Civil

MVP acadêmico para consultar materiais, fornecedores e dependências de etapas do **Residencial Horizonte**, usando somente dados fictícios. Referência principal: especificação do professor Ricardo Roberto de Lima. São três arquivos de aplicação, quatro dependências diretas e uma tela.

## Executar gratuitamente no Mac

A IA agora roda **somente no Ollama local**, sem chave, conta ou créditos. O código rejeita endpoints externos e modelos com nome `cloud`. O servidor deve ser iniciado com `OLLAMA_NO_CLOUD=1`. Neo4j, Streamlit e a lógica GraphRAG permanecem os mesmos.

Configuração verificada: Apple M4, 16 GiB de RAM, macOS 26.6.2 e aproximadamente 67 GiB livres antes dos downloads. Modelos escolhidos: **qwen2.5:7b** (quantizado Q4_K_M, cerca de 4,7 GB, suporte a português/JSON) e **embeddinggemma** (300M, cerca de 622 MB, multilíngue, 768 dimensões). O modelo de chat é maior que variantes de 3B para favorecer extração estruturada, mas cabe na memória disponível. Ollama instalado pelo Homebrew. Não foi adicionada biblioteca Python.

### Iniciar no dia da apresentação

1. No Neo4j Desktop, inicie a instância `projeto-final` e espere **RUNNING**. A instalação atual usa Neo4j 2026.10.2; não é preciso recriar o banco nem instalar plugins.
2. Abra um terminal e mantenha o Ollama rodando:

```bash
OLLAMA_NO_CLOUD=1 OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=1 ollama serve
```

Se informar que a porta 11434 já está ocupada, o Ollama pode já estar aberto: confira `ollama list`. Não inicie uma segunda cópia.

3. Em outro terminal, abra a pasta do projeto e a interface:

```bash
cd caminho/para/ConstrucaoCivil
source .venv/bin/activate
streamlit run app.py
```

Abra `http://127.0.0.1:8501`. Depois de uma alteração no `.env`, reinicie o Streamlit. Para encerrar um servidor, pressione **Ctrl+C** no terminal correspondente. Os dados do Neo4j permanecem gravados.

### Preparação em outro Mac (uma vez)

Requer Python 3.11+ (ambiente do projeto: 3.14), Neo4j e Ollama. Com Homebrew instalado:

```bash
git clone https://github.com/Devjv10/ConstrucaoCivil.git
cd ConstrucaoCivil
brew install ollama
# Inicie o servidor com o comando acima; em outro terminal:
ollama pull qwen2.5:7b
ollama pull embeddinggemma
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Copie somente se ainda não tiver .env:
cp .env.example .env
```

No `.env`, configure a senha do Neo4j e mantenha:

```dotenv
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen2.5:7b
EMBEDDING_MODEL=embeddinggemma
EMBEDDING_DIMENSIONS=768
```

Não há `LLM_API_KEY` necessária. Use um banco exclusivo do projeto. Para carregar ou atualizar os cinco documentos:

```bash
python ingest.py
```

Não repita a ingestão durante a apresentação. Ela faz duas chamadas de extração por documento (entidades, depois relações restritas a esses nomes; uma correção adicional de relações se necessário) e cinco chamadas de embeddings locais; cada consulta atendida usa duas chamadas de chat e uma de embedding. O carregamento inicial do modelo pode demorar mais; aguarde. O limite é 180 segundos por chamada. Feche aplicativos pesados se necessário.

Na adaptação, apenas o índice `document_vectors` foi recriado (1536 → 768 dimensões). Nenhum nó, relação ou outro índice foi apagado nessa migração. Se trocar o modelo/dimensão novamente, reingira os textos e ajuste o índice; não misture embeddings de modelos diferentes.

O download inicial exige internet; depois, banco, modelos e aplicação executam localmente. Não há cobrança por consultas. Os custos existentes de energia e internet continuam sendo do próprio computador. Referências: [modelo Qwen2.5](https://ollama.com/library/qwen2.5:7b), [EmbeddingGemma](https://ollama.com/library/embeddinggemma), [API local compatível](https://docs.ollama.com/api/openai-compatibility) e [desativar nuvem](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features).

## Como funciona

```text
5 TXT → LLM identifica entidades → LLM extrai relações entre esses nomes
      → validação e normalização → Neo4j
Pergunta → LLM emite Cypher permitido + parâmetros
         → embedding + Fulltext → fusão por posição (RRF)
         → documentos → entidades → travessia Cypher
         → shortestPath como contexto adicional
         → LLM seleciona caminhos → frases restritas + fontes + grafo
```

Cada TXT é pequeno e funciona como uma unidade de recuperação. Vetores usam similaridade de cosseno. A fusão soma `1/(60 + posição)` em cada ranking e seleciona até cinco documentos. Seus nós mencionados são as sementes da travessia. Como a base tem apenas cinco documentos, a recuperação pode incluir a base inteira; não se alega ganho de precisão ou desempenho nessa escala.

O Ollama recebe um JSON Schema que restringe a saída ao par intenção/Cypher de uma das três consultas parametrizadas do catálogo. Os parâmetros são gerados antes do Cypher, e obra/material obrigatórios não podem ficar vazios. Qualquer alteração no texto Cypher é rejeitada. Somente as consultas permitidas chegam à execução; nomes não são concatenados ao Cypher. O modo de leitura do driver é adicional, não substitui permissões do banco. A ingestão precisa de escrita.

**Geração restrita:** o LLM seleciona os identificadores dos caminhos; o código valida esses identificadores e verbaliza seus fatos com templates. Isso limita a liberdade de linguagem e evita novos nomes/relações na resposta. A extração e a escolha de intenção ainda podem errar; validação de trecho literal não prova correção semântica. Não se afirma eliminar alucinações.

KAG é usado aqui como inspiração para schema, validação e regras de dependência. Este MVP não reproduz integralmente os frameworks KAG/OpenSPG ou Microsoft GraphRAG.

## Modelo e requisitos obrigatórios

14 entidades de domínio: 1 Obra, 5 Materiais, 3 Fornecedores e 5 Etapas. Mais 5 Documentos. `Entidade` é um rótulo comum auxiliar para unicidade. Há 30 relações de domínio, além de `MENCIONA`; são extraídas dos textos, não inseridas por um cadastro manual.

| Requisito do professor | Implementação |
|---|---|
| Text-to-Graph / NER e RE | `ingest.py`: LLM extrai listas, trecho literal e extremos são validados |
| Resolução de entidades | `core.py`: acentos, caixa, espaços e aliases explícitos |
| LPG | Obra–UTILIZA→Material; Fornecedor–FORNECE→Material; Obra–POSSUI_ETAPA→Etapa; Etapa–NECESSITA→Material; Etapa–DEPENDE_DE→Etapa; Documento–MENCIONA→Entidade |
| Propriedades e constraints | IDs únicos; nomes; documentos com texto, hash, embedding e modelo; relações com lista de fontes |
| Índices | RANGE em nome, FULLTEXT em texto, VECTOR em embedding; unicidade também cria índices |
| Busca híbrida | Cosseno + Fulltext + travessia parametrizada, com fusão RRF |
| Multi-hop | Até quatro dependências entre etapas, seguidas de NECESSITA |
| Algoritmo de grafo | `shortestPath`, sem instalar GDS, passado ao LLM como contexto |
| Cypher via LLM | Emissão restrita a três formatos exatos, mais parâmetros |
| Explicabilidade | Grafo interativo, caminhos citados, nós consultados, Cypher e fontes |
| Confiança | Indicador qualitativo de cobertura documental; não é probabilidade |
| Entrega e defesa | Este README, testes, `RELATORIO.tex` e roteiro de oito minutos |

O professor menciona BTree: Neo4j 5 usa **RANGE** para essa finalidade. Essa atualização está documentada na [migração oficial](https://neo4j.com/docs/upgrade-migration-guide/current/version-5/migration/planning/).

## Testes e situação da validação

```bash
python -m unittest discover -s tests -v
# Com Neo4j e Ollama locais iniciados; grava dados, sem cobrança:
RUN_INTEGRATION=1 python -m unittest discover -s tests -v
```

Validação final: **19 testes aprovados com `RUN_INTEGRATION=1`**, sem testes ignorados (194,5 segundos nesta execução). Sem essa variável, são 17 testes locais e dois testes integrados ignorados; eles precisam de Neo4j e Ollama ativos. Os testes locais verificam normalização, validação de extração, rejeição de Cypher não permitido, RRF, fundamentação, recusa, dimensões de embedding, geração do HTML e interface Streamlit (inicialização, sucesso com dublê explícito e erro). Dublês existem apenas nos testes, não na aplicação. As quatro versões diretas instaladas foram fixadas em `requirements.txt`.

Os testes de integração verificam ingestão, contagens, fontes de cada relação, índices, as três perguntas, menor caminho e perguntas sem resposta. A execução real com Ollama já carregou 14 entidades, 5 documentos e 30 relações de domínio. As três perguntas responderam corretamente com fontes; tempos observados numa execução: 14,3 s, 9,8 s e 14,5 s, respectivamente. São medições pontuais, não um benchmark. AppTest e HTML gerado não substituem a inspeção visual no navegador antes da defesa.

## Demonstração e limites

1. “Quais materiais são utilizados na obra Residencial Horizonte?” — espera-se cimento, areia, brita, aço e tijolo.
2. “Qual fornecedor fornece cimento para a obra?” — espera-se Construmax.
3. “Quais etapas da construção podem ser afetadas pela falta de cimento?” — diretamente Fundação, Estrutura, Alvenaria e Acabamento; indiretamente Instalações.

Esses resultados foram observados com os modelos locais e o banco real. Perguntas sobre estoque, preços e prazos devem ser recusadas. O roteiro de apresentação da equipe é mantido apenas localmente e não faz parte deste repositório.

O catálogo cobre apenas esses três tipos de consulta. Não há raciocínio aberto, cálculo de cronograma, treinamento de modelo ou benchmark comparativo. Documentos modificados são reprocessados ao repetir ingestão; IDs e relações usam MERGE. Remover/renomear documentos, trocar dimensão de embedding ou reutilizar banco de outro projeto exige preparar novamente um banco exclusivo; não há gestão de migrações neste MVP.

Repositório: [Devjv10/ConstrucaoCivil](https://github.com/Devjv10/ConstrucaoCivil). Para entregar: preencha autores/instituição e revise os resultados em `RELATORIO.tex`, compile em um projeto IEEE no Overleaf (não compilado localmente) e garanta acesso do professor ao repositório. O PDF do artigo permanece uma tarefa de entrega.

## Ordem de estudo e créditos

1. `data/`: fatos fictícios e sentido das dependências.
2. `ingest.py`: extração, validação, índices e gravação.
3. `core.py`: catálogo Cypher, recuperação e resposta.
4. `app.py`: apresentação dos resultados.

Implementação desenvolvida com assistência de IA; a equipe deve revisar, compreender e declarar esse uso conforme as regras da disciplina. Bibliotecas utilizadas: [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/), [Streamlit](https://docs.streamlit.io/), [PyVis](https://pyvis.readthedocs.io/) e [python-dotenv](https://github.com/theskumar/python-dotenv). API HTTP via biblioteca padrão do Python.

Referências técnicas: [índices vetoriais](https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/vector-indexes/), [shortestPath em Neo4j 5](https://neo4j.com/docs/cypher-manual/5/patterns/shortest-paths/), [GraphRAG, Edge et al.](https://arxiv.org/abs/2404.16130) e [KAG, Liang et al.](https://arxiv.org/abs/2409.13731).
