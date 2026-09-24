"""
Protocolo de avaliação do GAM (Yan et al., 2025, arXiv:2511.18423), fixado.

Tudo o que o artigo e o código oficial determinam está aqui, com a origem de
cada valor. O artigo é curto nos detalhes; o que ele não diz foi lido do código
de avaliação publicado pelos autores:

    https://github.com/VectorSpaceLab/general-agentic-memory
    commit 565db2cc2518d377e44389b82aecf3cc129d5fe5, pasta research/
    (eval/hotpotqa_test.py, eval/ruler_test.py, eval/narrativeqa_test.py,
     scripts/eval_*.sh, scripts/download_data.sh, download_data/*.py)

O que é do PROTOCOLO (igual para qualquer método, inclusive o nosso):

* dados, recortes, número de amostras e ordem;
* páginas de 2.048 tokens do tokenizador do BGE-M3, sem sobreposição, com o
  cabeçalho "[Session i]";
* até 5 páginas recuperadas (o artigo: "top-5 retrieved segments" no RAG e
  "maximum number of retrieved pages to 5" no GAM);
* o prompt do modelo que responde ("working generator"), temperatura 0,3 e
  teto de 256 tokens, uma única mensagem de usuário sem system prompt;
* as métricas: F1 do LongBench (máximo sobre as respostas de referência) e,
  no RULER, a acurácia por contenção de todas as saídas esperadas.

O que é do MÉTODO do GAM e por isso NÃO entra aqui: o memorizador, os prompts
de pesquisa, os system prompts específicos por tarefa do RULER, a profundidade
de reflexão. O nosso método substitui essa parte; o resto é idêntico.
"""

from __future__ import annotations

from dataclasses import dataclass

GAM_PAPER = "arXiv:2511.18423v1 (General Agentic Memory Via Deep Research)"
GAM_REPO = "https://github.com/VectorSpaceLab/general-agentic-memory"
GAM_COMMIT = "565db2cc2518d377e44389b82aecf3cc129d5fe5"

# -- páginas (eval/*_test.py: --max-tokens 2048 --embedding-model-path BAAI/bge-m3)
PAGE_TOKENS = 2048
PAGE_TOKENIZER = "BAAI/bge-m3"
TOP_K = 5

# -- modelo que responde (working generator em eval/*_test.py)
READER_TEMPERATURE = 0.3
READER_MAX_TOKENS = 256
# O gerador do GAM não envia seed. Enviamos 42 para que duas execuções com o
# mesmo contexto deem a mesma resposta; é o único parâmetro acrescentado, e a
# temperatura continua 0,3. `--reader-seed none` reproduz o GAM literalmente.
READER_SEED = 42


@dataclass(frozen=True)
class RemoteFile:
    """Arquivo de um repositório de dataset do Hugging Face, fixado por commit."""

    repo: str
    revision: str
    path: str
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


# -- HotpotQA: conjunto do MemAgent (scripts/download_data.sh, eval_hotpotqa.sh)
# O script de avaliação do GAM usa eval_400, eval_1600 e eval_3200 (56K, 224K e
# 448K tokens). O download_data.sh baixa eval_6400 no lugar do eval_3200; é um
# descuido do repositório: o eval_hotpotqa.sh e o artigo usam 3200 (448K).
HOTPOT_REPO = "BytedTsinghua-SIA/hotpotqa"
HOTPOT_REVISION = "27275ff4fee67ac0acb6478e405e7ac07efbdc1a"
HOTPOT_SPLITS = {"56k": "eval_400.json", "224k": "eval_1600.json", "448k": "eval_3200.json"}
HOTPOT_FILES = {
    "eval_400.json": RemoteFile(HOTPOT_REPO, HOTPOT_REVISION, "eval_400.json", 29652378,
                                "da11b9e4fa986476e0d29cffc4e079b0bf60b761e40dbe18fea182cd663dd9bb"),
    "eval_1600.json": RemoteFile(HOTPOT_REPO, HOTPOT_REVISION, "eval_1600.json", 118487493,
                                 "1e103e0dfc05ad5326f6e105d22e047b70dc23149943395f9895cc73c29c681b"),
    "eval_3200.json": RemoteFile(HOTPOT_REPO, HOTPOT_REVISION, "eval_3200.json", 236959262,
                                 "1a1c25b721ce303356aab899005fc17d8a90b8bb3624e9fbf4b97305dae948fe"),
}
HOTPOT_SAMPLES = 128  # cada arquivo; todas as amostras são avaliadas

# -- RULER 128K (download_data/download_ruler.py, scripts/eval_ruler.sh)
RULER_REPO = "lighteval/RULER-131072-Qwen2.5-Instruct"
RULER_REVISION = "5bdc2f0e2a6e2dc79abbc65378b504d400397415"
_RULER_SHA = {
    "cwe": (85438632, "98e3b7d9c20f107c9fe94998c78b96ff1ba7d98fd4042ca649bb165f28faa5ec"),
    "fwe": (41712112, "039a8be5242bef5224877e2e7dd5276507145842caee1d3480ac62c1716a824d"),
    "niah_multikey_1": (177992614, "8c28d97f38c943cbd0e37852af25eca730ef815a13a594a57298dd88c2478034"),
    "niah_multikey_2": (65931758, "2ed5b3df91208f62c83a4de582a495f521223fb63684e6ea51f88ec28c2e2321"),
    "niah_multikey_3": (62999368, "666bce3bafd860fac1dae267c22d1d57674d8fcb7c583709c342f14f656c7431"),
    "niah_multiquery": (178034178, "cc6db6b95ccfcc813416136ad4862b85b088cd0a7b07cc1295b8e61379c9f069"),
    "niah_multivalue": (177999974, "a30f971e31c76f99417c17a0cddc89e97724c5345dd1ec4693ac1340c80bdd3f"),
    "niah_single_1": (11940953, "9186b5ebef3d20e085d49299c88170ca7454261a1428c9e4b937e53cb77f7284"),
    "niah_single_2": (177911012, "d55ab89c5fce8697912cc65612c1e858f8d2d2ac90985621df38ce2f9398f9ab"),
    "niah_single_3": (177952106, "9978650cf88518205c8c592452f62645cb29902b64f24ab82d1a0d4d027f0769"),
    "qa_1": (183241397, "68aeab3cca9f3524b3493dbeedb3c81244aa8ff51ae75c5b0b2f943d505181cb"),
    "qa_2": (163038403, "4e3ec049810107660a91e0d726e2dfd80050e0ba2fda2f91cf2ff48140f65eaf"),
    "vt": (12025512, "a1f0059a511faa7f3e9b3234fc4e8f33a0c8315ace52011598e5d5bfb690dc59"),
}
RULER_FILES = {task: RemoteFile(RULER_REPO, RULER_REVISION, f"data/{task}-00000-of-00001.parquet",
                                size, digest)
               for task, (size, digest) in _RULER_SHA.items()}
# Ordem de eval_ruler.sh.
RULER_TASKS = ("qa_1", "qa_2", "vt", "niah_single_1", "niah_single_2", "niah_single_3",
               "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
               "niah_multivalue", "cwe", "fwe")
# As quatro colunas da Tabela 1(b). O artigo não lista a composição; é a da
# RULER original (recuperação = as oito NIAH, rastreamento = VT, agregação =
# CWE e FWE, QA = qa_1 e qa_2). Os números publicados são consistentes com
# 500 amostras por tarefa: todas as colunas Retri. são múltiplos de 1/4000
# (arredondados), MT de 1/500 e AGG./QA de 1/1000.
RULER_GROUPS = {
    "Retri.": ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
               "niah_multikey_2", "niah_multikey_3", "niah_multiquery", "niah_multivalue"),
    "MT": ("vt",),
    "AGG.": ("cwe", "fwe"),
    "QA": ("qa_1", "qa_2"),
}
RULER_SAMPLES = 500  # por tarefa; eval_ruler.sh avalia todas

# -- NarrativeQA (download_data/download_narrativeqa.py, scripts/eval_narrativeqa.sh)
# load_dataset("deepmind/narrativeqa") salvo em test.parquet e relido na mesma
# ordem (os oito shards em sequência); random.seed(42); random.shuffle; as 300
# primeiras. O artigo: "randomly sample a subset of 300 questions ... average
# token length is 87K". Com o tokenizador do GPT-4o a média dá 86K.
NARRATIVEQA_REPO = "deepmind/narrativeqa"
NARRATIVEQA_REVISION = "2e643e7363944af1c33a652d1c87320d0871c4e4"
_NQA_SHA = (
    (8559262, "638fe830caab599fd40333b5c6783a83e12e431de35c9dde730d329596456f66"),
    (44507679, "51acc457d81b38e4cfb048436a239fe415aafa9c559577456720774c0cc5b407"),
    (101411128, "83f7672070ca24edb5a2844271cfbb18f351fc1fd84be945411a6814a13eb2e3"),
    (221739348, "2b305c1ac27799d710172855265ba665b8fc258430a97a7994d976414db03021"),
    (60842742, "c69949be3fe1d57f1171f929f18646eb99410784162b15632e778085a15abb79"),
    (121182898, "38b87f8c0a4f0dc139808693bd019e2d0eb3a193f0b52fd52977b20b61f793c4"),
    (243201775, "d0c894370b0cd1462582e89d39924a4e66db5caf69fa8d9ecc75e80284e7091a"),
    (58520238, "c9eddfda2441b23a7828886c3dad94986e693b989bf663a324c69c87870f2c3b"),
)
NARRATIVEQA_FILES = tuple(
    RemoteFile(NARRATIVEQA_REPO, NARRATIVEQA_REVISION, f"data/test-{i:05d}-of-00008.parquet",
               size, digest) for i, (size, digest) in enumerate(_NQA_SHA))
NARRATIVEQA_TEST_SIZE = 10557
NARRATIVEQA_SEED = 42
NARRATIVEQA_SAMPLES = 300
# random.seed(42); random.shuffle(list(range(10557)))[:5]; conferido com os dados.
NARRATIVEQA_FIRST_INDICES = (418, 5545, 6257, 5098, 2465)

BENCHMARKS = ("hotpotqa", "ruler", "narrativeqa")
EXPECTED = {("hotpotqa", split): HOTPOT_SAMPLES for split in HOTPOT_SPLITS}
EXPECTED.update({("ruler", task): RULER_SAMPLES for task in RULER_TASKS})
EXPECTED[("narrativeqa", "test")] = NARRATIVEQA_SAMPLES

# -- Tabela 1(b) do artigo, para comparação no relatório. Colunas:
# HotpotQA 56K, 224K, 448K (F1); RULER Retri., MT, AGG., QA (acurácia);
# NarrativeQA (F1). Linhas copiadas do PDF.
TABLE_COLUMNS = ("HotpotQA 56K", "HotpotQA 224K", "HotpotQA 448K",
                 "RULER Retri.", "RULER MT", "RULER AGG.", "RULER QA", "NarrativeQA")
PAPER_TABLE_1B = {
    "GPT-4o-mini": {
        "LONG-LLM": (56.56, 54.29, 53.92, 80.30, 60.60, 36.70, 61.60, 31.26),
        "RAG": (52.71, 51.84, 54.01, 94.25, 0.00, 35.50, 55.90, 25.00),
        "A-MEM": (33.90, 30.22, 31.37, 44.23, 0.00, 29.20, 46.50, 27.07),
        "MEM0": (32.58, 31.74, 27.41, 46.83, 53.80, 34.10, 51.70, 29.16),
        "MEMORYOS": (26.47, 23.10, 24.16, 63.10, 2.40, 35.60, 36.90, 26.70),
        "LIGHTMEM": (40.93, 35.28, 30.02, 27.63, 36.20, 34.00, 52.60, 17.51),
        "GAM": (63.22, 64.56, 59.81, 97.70, 93.20, 42.50, 72.50, 36.86),
    },
    "Qwen2.5-14B": {
        "LONG-LLM": (49.75, 46.82, 43.17, 70.85, 80.00, 15.40, 45.60, 29.69),
        "RAG": (51.81, 46.72, 48.36, 92.78, 0.00, 24.70, 47.80, 18.29),
        "A-MEM": (27.04, 25.65, 22.92, 39.73, 0.00, 25.80, 40.20, 25.18),
        "MEM0": (30.12, 32.44, 26.55, 43.03, 41.20, 31.50, 46.10, 27.80),
        "MEMORYOS": (24.58, 30.25, 23.13, 54.58, 3.00, 5.20, 34.60, 23.45),
        "LIGHTMEM": (37.30, 27.72, 28.25, 27.53, 17.40, 25.60, 53.00, 16.57),
        "GAM": (64.07, 55.99, 57.87, 93.43, 90.20, 36.10, 74.50, 34.77),
    },
}


def column_of(benchmark: str, split: str) -> str | None:
    """Coluna da Tabela 1(b) à qual um recorte contribui."""
    if benchmark == "hotpotqa":
        return {"56k": "HotpotQA 56K", "224k": "HotpotQA 224K", "448k": "HotpotQA 448K"}.get(split)
    if benchmark == "narrativeqa":
        return "NarrativeQA"
    for group, tasks in RULER_GROUPS.items():
        if split in tasks:
            return f"RULER {group}"
    return None
