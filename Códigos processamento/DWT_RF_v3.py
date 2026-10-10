# =============================================================================
# CLASSIFICADOR DE DISTÚRBIOS DE QUALIDADE DE ENERGIA
# =============================================================================
#
# EXPERIMENTO:
#
#     SIGNAL_FEATURE
#          |
#          v
#     SINAL BRUTO
#          |
#          v
#     DWT - TRANSFORMADA WAVELET DISCRETA
#          |
#          v
#     EXTRAÇÃO DE CARACTERÍSTICAS
#          |
#          v
#     STRATIFIED K-FOLD
#          |
#          +-------------------------------+
#          |                               |
#          v                               v
#     CONJUNTO TREINO                CONJUNTO TESTE
#          |                               |
#          v                               |
#     FILTRO CORRELAÇÃO                    |
#          |                               |
#          v                               |
#     FISHER DISCRIMINANT RATIO            |
#          |                               |
#          v                               |
#     14 CARACTERÍSTICAS                   |
#          |                               |
#          v                               |
#     STANDARD SCALER                      |
#          |                               |
#          v                               |
#     RANDOM FOREST                        |
#          |                               |
#          +---------------+---------------+
#                          |
#                          v
#                    PREDIÇÃO OOF
#                          |
#                          v
#                    MÉTRICAS FINAIS
#
# ---------------------------------------------------------------------------
# IMPORTANTE:
#
# Este código utiliza EXCLUSIVAMENTE:
#
#     Signal_Feature/signal_*.csv
#
# Não utiliza:
#
#     Dataset_Train
#     Dataset_train_class
#     index_total
#
# Portanto, a seleção de características é refeita dentro de cada fold.
#
# =============================================================================


# =============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from codecarbon import EmissionsTracker


warnings.filterwarnings("ignore")


# =============================================================================
# 2. CONFIGURAÇÕES GERAIS
# =============================================================================


# -----------------------------------------------------------------------------
# PASTA DO DATASET
# -----------------------------------------------------------------------------
#
# ATENÇÃO:
#
# Estamos apontando especificamente para:
#
#     Signal_Feature
#
# e NÃO para Dataset_Train.
#
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PASTA_DATASET = (
    BASE_DIR
    / "Banco_de_Dados"
    / "CSV"
    / "Signal_Feature"
)


# -----------------------------------------------------------------------------
# PASTA DE RESULTADOS
# -----------------------------------------------------------------------------

PASTA_RESULTADOS = (
    Path(__file__).resolve().parent
    / "Resultados"
    / "Resultados Testes"
    / "Teste 5 - DWT & RF"
)


# =============================================================================
# 3. PASTAS DE SAÍDA
# =============================================================================

PASTA_CODECARBON = os.path.join(
    PASTA_RESULTADOS,
    "CodeCarbon"
)

PASTA_MATRIZ = os.path.join(
    PASTA_RESULTADOS,
    "Matriz_Confusao"
)

PASTA_ROC = os.path.join(
    PASTA_RESULTADOS,
    "ROC"
)

PASTA_PR = os.path.join(
    PASTA_RESULTADOS,
    "Precision_Recall"
)

PASTA_METRICAS = os.path.join(
    PASTA_RESULTADOS,
    "Metricas"
)

PASTA_RELATORIO = os.path.join(
    PASTA_RESULTADOS,
    "Relatorio"
)

PASTA_WAVELET = os.path.join(
    PASTA_RESULTADOS,
    "Caracteristicas_Wavelet"
)

PASTA_DOMINIO_WAVELET = os.path.join(
    PASTA_RESULTADOS,
    "Dominio_Wavelet"
)

PASTA_AUDITORIA = os.path.join(
    PASTA_RESULTADOS,
    "Auditoria"
)


PASTAS = [
    PASTA_RESULTADOS,
    PASTA_CODECARBON,
    PASTA_MATRIZ,
    PASTA_ROC,
    PASTA_PR,
    PASTA_METRICAS,
    PASTA_RELATORIO,
    PASTA_WAVELET,
    PASTA_DOMINIO_WAVELET,
    PASTA_AUDITORIA
]


for pasta in PASTAS:

    os.makedirs(
        pasta,
        exist_ok=True
    )


# =============================================================================
# 4. CONFIGURAÇÃO DO DATASET
# =============================================================================

CLASSES = [
    "capc",
    "har",
    "normal",
    "notch",
    "sag",
    "spike",
    "swell"
]


# -----------------------------------------------------------------------------
# FREQUÊNCIA DE AMOSTRAGEM
# -----------------------------------------------------------------------------

FS = 15360


# -----------------------------------------------------------------------------
# FREQUÊNCIA FUNDAMENTAL
# -----------------------------------------------------------------------------

FUNDAMENTAL = 60


# -----------------------------------------------------------------------------
# CONFIGURAÇÃO WAVELET
# -----------------------------------------------------------------------------

WAVELET = "db4"

NIVEL_WAVELET = 6


# -----------------------------------------------------------------------------
# SELEÇÃO DE CARACTERÍSTICAS
# -----------------------------------------------------------------------------

N_FEATURES_FISHER = 14

LIMITE_CORRELACAO = 0.95


# -----------------------------------------------------------------------------
# RANDOM FOREST
# -----------------------------------------------------------------------------

N_ESTIMATORS = 100

RANDOM_STATE = 42

N_JOBS = 1


# -----------------------------------------------------------------------------
# STRATIFIED K-FOLD
# -----------------------------------------------------------------------------

N_SPLITS = 5


# -----------------------------------------------------------------------------
# BOOTSTRAP
# -----------------------------------------------------------------------------

N_BOOTSTRAP = 1000


# =============================================================================
# 5. FUNÇÃO DE AUDITORIA DO DATASET
# =============================================================================

def auditar_origem_dataset():

    print()
    print("=" * 80)
    print("AUDITORIA DA ORIGEM DOS DADOS")
    print("=" * 80)

    caminho_normalizado = os.path.normpath(
        PASTA_DATASET
    )

    print()
    print(
        "Pasta utilizada:"
    )

    print(
        caminho_normalizado
    )

    # -------------------------------------------------------------------------
    # Verifica se Signal_Feature está no caminho
    # -------------------------------------------------------------------------

    if "Signal_Feature" not in caminho_normalizado:

        raise RuntimeError(
            "\nERRO DE AUDITORIA:\n"
            "O caminho do dataset não aponta para Signal_Feature."
        )

    # -------------------------------------------------------------------------
    # Verifica se Dataset_Train NÃO está no caminho
    # -------------------------------------------------------------------------

    if "Dataset_Train" in caminho_normalizado:

        raise RuntimeError(
            "\nERRO DE AUDITORIA:\n"
            "O código está apontando para Dataset_Train."
        )

    print()
    print(
        "[OK] Signal_Feature está sendo utilizado."
    )

    print(
        "[OK] Dataset_Train não está sendo utilizado."
    )

    print(
        "[OK] Dataset_train_class não está sendo utilizado."
    )

    print(
        "[OK] index_total não está sendo utilizado."
    )

    print()


# =============================================================================
# 6. CARREGAMENTO DOS SINAIS
# =============================================================================

def carregar_dataset():

    print()
    print("=" * 80)
    print("CARREGANDO DATASET - SIGNAL_FEATURE")
    print("=" * 80)
    print()

    sinais = []

    labels = []

    identificadores = []

    for classe in CLASSES:

        arquivo = os.path.join(
            PASTA_DATASET,
            f"signal_{classe}.csv"
        )

        if not os.path.exists(arquivo):

            raise FileNotFoundError(
                f"\nArquivo não encontrado:\n{arquivo}\n"
            )

        # ---------------------------------------------------------------------
        # Leitura
        # ---------------------------------------------------------------------

        dados = pd.read_csv(
            arquivo,
            header=None
        )

        dados = dados.values.astype(
            np.float64
        )

        print(
            f"{classe:<10} -> "
            f"{os.path.basename(arquivo):<30} "
            f"shape = {dados.shape}"
        )

        # ---------------------------------------------------------------------
        # Verificação do número de amostras
        # ---------------------------------------------------------------------

        if dados.shape[1] != 2560:

            raise ValueError(
                f"\nO arquivo {arquivo} possui "
                f"{dados.shape[1]} amostras por sinal. "
                f"Esperado: 2560."
            )

        # ---------------------------------------------------------------------
        # Adiciona os sinais
        # ---------------------------------------------------------------------

        sinais.append(
            dados
        )

        labels.extend(
            [classe] * dados.shape[0]
        )

        # ---------------------------------------------------------------------
        # Identificador único de cada sinal
        #
        # Esse identificador NÃO é utilizado como característica.
        #
        # Ele serve apenas para auditoria.
        # ---------------------------------------------------------------------

        for i in range(
            dados.shape[0]
        ):

            identificadores.append(
                f"{classe}_{i:03d}"
            )

    # -------------------------------------------------------------------------
    # Junta tudo
    # -------------------------------------------------------------------------

    X = np.vstack(
        sinais
    )

    y = np.array(
        labels
    )

    ids = np.array(
        identificadores
    )

    # -------------------------------------------------------------------------
    # Informações
    # -------------------------------------------------------------------------

    print()

    print(
        f"Número total de sinais: "
        f"{X.shape[0]}"
    )

    print(
        f"Número de amostras por sinal: "
        f"{X.shape[1]}"
    )

    print(
        f"Número de classes: "
        f"{len(np.unique(y))}"
    )

    print()

    print(
        "Distribuição das classes:"
    )

    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    return (
        X,
        y,
        ids
    )


# =============================================================================
# 7. EXTRAÇÃO DE CARACTERÍSTICAS WAVELET
# =============================================================================

def extrair_caracteristicas_wavelet(
    sinal
):

    # -------------------------------------------------------------------------
    # DWT
    # -------------------------------------------------------------------------

    coeficientes = pywt.wavedec(
        sinal,
        WAVELET,
        level=NIVEL_WAVELET
    )

    # -------------------------------------------------------------------------
    # Energia total
    # -------------------------------------------------------------------------

    energia_total = sum(
        np.sum(
            coef ** 2
        )
        for coef in coeficientes
    )

    caracteristicas = []

    nomes = []

    # -------------------------------------------------------------------------
    # Nomes dos níveis
    # -------------------------------------------------------------------------

    nomes_niveis = [
        f"A{NIVEL_WAVELET}"
    ]

    for nivel in range(
        NIVEL_WAVELET,
        0,
        -1
    ):

        nomes_niveis.append(
            f"D{nivel}"
        )

    # -------------------------------------------------------------------------
    # Extração
    # -------------------------------------------------------------------------

    for nome_nivel, coef in zip(
        nomes_niveis,
        coeficientes
    ):

        coef = np.asarray(
            coef
        )

        # Energia
        energia = np.sum(
            coef ** 2
        )

        # Energia relativa
        if energia_total > 0:

            energia_relativa = (
                energia /
                energia_total
            )

        else:

            energia_relativa = 0.0

        # Média
        media = np.mean(
            coef
        )

        # Desvio padrão
        desvio = np.std(
            coef
        )

        # RMS
        rms = np.sqrt(
            np.mean(
                coef ** 2
            )
        )

        # Máximo absoluto
        maximo = np.max(
            np.abs(coef)
        )

        # Mínimo absoluto
        minimo = np.min(
            np.abs(coef)
        )

        # ---------------------------------------------------------------------
        # Entropia
        # ---------------------------------------------------------------------

        energia_coef = (
            coef ** 2
        )

        soma_energia = np.sum(
            energia_coef
        )

        if soma_energia > 0:

            probabilidade = (
                energia_coef /
                soma_energia
            )

            probabilidade = (
                probabilidade[
                    probabilidade > 0
                ]
            )

            entropia = -np.sum(
                probabilidade *
                np.log2(
                    probabilidade
                )
            )

        else:

            entropia = 0.0

        # ---------------------------------------------------------------------
        # Valores
        # ---------------------------------------------------------------------

        valores = [

            energia,

            energia_relativa,

            media,

            desvio,

            rms,

            maximo,

            minimo,

            entropia
        ]

        # ---------------------------------------------------------------------
        # Nomes
        # ---------------------------------------------------------------------

        nomes_caracteristicas = [

            f"{nome_nivel}_energia",

            f"{nome_nivel}_energia_relativa",

            f"{nome_nivel}_media",

            f"{nome_nivel}_desvio",

            f"{nome_nivel}_RMS",

            f"{nome_nivel}_max",

            f"{nome_nivel}_min",

            f"{nome_nivel}_entropia"
        ]

        caracteristicas.extend(
            valores
        )

        nomes.extend(
            nomes_caracteristicas
        )

    return (
        np.array(
            caracteristicas
        ),
        nomes
    )


# =============================================================================
# 7B. [NOVO] REPRESENTAÇÃO BRUTA DO SINAL NO DOMÍNIO WAVELET
# =============================================================================
#
# A função "extrair_caracteristicas_wavelet" (acima) calcula ESTATÍSTICAS
# (energia, média, RMS, entropia, etc.) a partir dos coeficientes da DWT.
#
# A função abaixo é diferente: ela NÃO resume nada. Ela devolve os
# PRÓPRIOS coeficientes da DWT, nível a nível.
#
# Ou seja, é a representação do sinal no domínio da transformada wavelet
# (equivalente ao que o espectro de Fourier representa no domínio da
# frequência), e não características derivadas dela.
#
# =============================================================================

def extrair_representacao_wavelet(
    sinal
):

    # -------------------------------------------------------------------------
    # DWT
    # -------------------------------------------------------------------------

    coeficientes = pywt.wavedec(
        sinal,
        WAVELET,
        level=NIVEL_WAVELET
    )

    # -------------------------------------------------------------------------
    # Nomes dos níveis (A6, D6, D5, ..., D1)
    # -------------------------------------------------------------------------

    nomes_niveis = [
        f"A{NIVEL_WAVELET}"
    ]

    for nivel in range(
        NIVEL_WAVELET,
        0,
        -1
    ):

        nomes_niveis.append(
            f"D{nivel}"
        )

    # -------------------------------------------------------------------------
    # Um vetor de coeficientes por nível, sem nenhum resumo estatístico
    # -------------------------------------------------------------------------

    representacao = {
        nome_nivel: np.asarray(coef)
        for nome_nivel, coef in zip(
            nomes_niveis,
            coeficientes
        )
    }

    return (
        representacao,
        nomes_niveis
    )


def construir_matriz_dominio_wavelet(
    X,
    y,
    ids
):

    print()
    print("=" * 80)
    print("[NOVO] REPRESENTAÇÃO BRUTA NO DOMÍNIO WAVELET")
    print("=" * 80)
    print()

    inicio = time.perf_counter()

    coeficientes_por_nivel = None

    nomes_niveis_ref = None

    for i, sinal in enumerate(
        X
    ):

        representacao, nomes_niveis = (
            extrair_representacao_wavelet(
                sinal
            )
        )

        if coeficientes_por_nivel is None:

            nomes_niveis_ref = nomes_niveis

            coeficientes_por_nivel = {
                nivel: [] for nivel in nomes_niveis_ref
            }

        for nivel in nomes_niveis_ref:

            coeficientes_por_nivel[nivel].append(
                representacao[nivel]
            )

        if (
            (i + 1) % 100 == 0
            or
            i == len(X) - 1
        ):

            print(
                f"Processados: "
                f"{i + 1}/{len(X)}"
            )

    # -------------------------------------------------------------------------
    # Monta um bloco de colunas por nível
    #
    # (dentro de um mesmo nível, todos os sinais têm o mesmo número de
    # coeficientes, pois o comprimento depende apenas do tamanho do sinal,
    # da wavelet e do nível — nunca do conteúdo do sinal)
    # -------------------------------------------------------------------------

    blocos = []

    total_coeficientes = 0

    for nivel in nomes_niveis_ref:

        matriz_nivel = np.array(
            coeficientes_por_nivel[nivel]
        )

        total_coeficientes += matriz_nivel.shape[1]

        colunas_nivel = [
            f"{nivel}_coef_{indice:04d}"
            for indice in range(
                matriz_nivel.shape[1]
            )
        ]

        blocos.append(
            pd.DataFrame(
                matriz_nivel,
                columns=colunas_nivel
            )
        )

    df_dominio_wavelet = pd.concat(
        blocos,
        axis=1
    )

    df_dominio_wavelet["classe"] = y

    df_dominio_wavelet["ID_sinal"] = ids

    tempo = (
        time.perf_counter()
        -
        inicio
    )

    print()

    print(
        f"Coeficientes por sinal (todos os níveis): "
        f"{total_coeficientes}"
    )

    print(
        f"Tempo de extração: "
        f"{tempo:.2f} segundos"
    )

    return df_dominio_wavelet


# =============================================================================
# 8. CONSTRUÇÃO DA MATRIZ WAVELET
# =============================================================================

def construir_matriz_wavelet(
    X
):

    print()
    print("=" * 80)
    print("ETAPA 1 - TRANSFORMADA WAVELET")
    print("=" * 80)

    print()
    print(
        f"Wavelet utilizada      : "
        f"{WAVELET}"
    )

    print(
        f"Nível de decomposição  : "
        f"{NIVEL_WAVELET}"
    )

    inicio = time.perf_counter()

    matriz = []

    nomes = None

    for i, sinal in enumerate(
        X
    ):

        caracteristicas, nomes_temp = (
            extrair_caracteristicas_wavelet(
                sinal
            )
        )

        matriz.append(
            caracteristicas
        )

        if nomes is None:

            nomes = nomes_temp

        if (
            (i + 1) % 100 == 0
            or
            i == len(X) - 1
        ):

            print(
                f"Processados: "
                f"{i + 1}/{len(X)}"
            )

    X_wavelet = np.array(
        matriz
    )

    tempo = (
        time.perf_counter()
        -
        inicio
    )

    print()

    print(
        f"Características Wavelet: "
        f"{X_wavelet.shape[1]}"
    )

    print(
        f"Tempo de extração: "
        f"{tempo:.2f} segundos"
    )

    return (
        X_wavelet,
        nomes
    )


# =============================================================================
# 9. FILTRO DE CORRELAÇÃO
# =============================================================================

def filtro_correlacao(
    X_train,
    nomes_features
):

    print()
    print(
        "ETAPA 2 - FILTRO DE CORRELAÇÃO"
    )

    df = pd.DataFrame(
        X_train,
        columns=nomes_features
    )

    matriz_corr = (
        df.corr(
            method="pearson"
        ).abs()
    )

    superior = matriz_corr.where(
        np.triu(
            np.ones(
                matriz_corr.shape
            ),
            k=1
        ).astype(bool)
    )

    remover = [

        coluna

        for coluna in superior.columns

        if any(
            superior[coluna]
            >
            LIMITE_CORRELACAO
        )
    ]

    manter = [

        coluna

        for coluna in df.columns

        if coluna not in remover
    ]

    indices_mantidos = [

        nomes_features.index(
            nome
        )

        for nome in manter
    ]

    print(
        f"Características antes: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Características depois: "
        f"{len(manter)}"
    )

    print(
        f"Características removidas: "
        f"{len(remover)}"
    )

    return (
        indices_mantidos,
        manter
    )


# =============================================================================
# 10. FISHER DISCRIMINANT RATIO
# =============================================================================

def fisher_score(
    X,
    y
):

    classes = np.unique(
        y
    )

    scores = []

    media_global = np.mean(
        X,
        axis=0
    )

    for j in range(
        X.shape[1]
    ):

        numerador = 0.0

        denominador = 0.0

        for classe in classes:

            valores = X[
                y == classe,
                j
            ]

            if len(valores) == 0:

                continue

            media_classe = np.mean(
                valores
            )

            variancia_classe = np.var(
                valores
            )

            n = len(
                valores
            )

            numerador += (

                n *

                (
                    media_classe
                    -
                    media_global[j]
                ) ** 2
            )

            denominador += (

                n *

                variancia_classe
            )

        if denominador > 0:

            score = (
                numerador /
                denominador
            )

        else:

            score = 0.0

        scores.append(
            score
        )

    return np.array(
        scores
    )


# =============================================================================
# 11. SELEÇÃO PELO FISHER
# =============================================================================

def selecionar_fisher(
    X_train,
    y_train,
    nomes_features
):

    print()
    print(
        "ETAPA 3 - FISHER DISCRIMINANT RATIO"
    )

    scores = fisher_score(
        X_train,
        y_train
    )

    indices_ordenados = (
        np.argsort(
            scores
        )[::-1]
    )

    n_selecionar = min(
        N_FEATURES_FISHER,
        X_train.shape[1]
    )

    indices_selecionados = (
        indices_ordenados[
            :n_selecionar
        ]
    )

    nomes_selecionados = [

        nomes_features[i]

        for i in indices_selecionados
    ]

    scores_selecionados = [

        scores[i]

        for i in indices_selecionados
    ]

    print(
        f"Características selecionadas: "
        f"{n_selecionar}"
    )

    print()

    print(
        "Características Wavelet selecionadas:"
    )

    for nome, score in zip(
        nomes_selecionados,
        scores_selecionados
    ):

        print(
            f"{nome:<35} "
            f"FDR = {score:.6e}"
        )

    return (
        indices_selecionados,
        nomes_selecionados,
        scores_selecionados
    )


# =============================================================================
# 12. AUDITORIA DA SEPARAÇÃO DOS FOLDS
# =============================================================================

def auditar_fold(
    train_idx,
    test_idx,
    ids,
    fold
):

    # -------------------------------------------------------------------------
    # Verifica interseção dos índices
    # -------------------------------------------------------------------------

    intersecao_indices = np.intersect1d(
        train_idx,
        test_idx
    )

    if len(
        intersecao_indices
    ) > 0:

        raise RuntimeError(
            f"\nVAZAMENTO DETECTADO NO FOLD {fold}!\n"
            f"Existem índices simultaneamente em treino e teste."
        )

    # -------------------------------------------------------------------------
    # Verifica interseção dos IDs
    # -------------------------------------------------------------------------

    ids_train = set(
        ids[train_idx]
    )

    ids_test = set(
        ids[test_idx]
    )

    intersecao_ids = (
        ids_train &
        ids_test
    )

    if len(
        intersecao_ids
    ) > 0:

        raise RuntimeError(
            f"\nVAZAMENTO DETECTADO NO FOLD {fold}!\n"
            f"Existem sinais simultaneamente em treino e teste."
        )

    print()
    print(
        f"[AUDITORIA FOLD {fold}]"
    )

    print(
        f"Treino: {len(train_idx)} sinais"
    )

    print(
        f"Teste : {len(test_idx)} sinais"
    )

    print(
        "[OK] Nenhum sinal aparece simultaneamente "
        "em treino e teste."
    )


# =============================================================================
# 13. BOOTSTRAP
# =============================================================================

def bootstrap_intervalo(
    y_true,
    y_pred,
    metrica,
    n_bootstrap=1000
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    valores = []

    n = len(
        y_true
    )

    for _ in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            n
        )

        yt = y_true[
            indices
        ]

        yp = y_pred[
            indices
        ]

        try:

            valor = metrica(
                yt,
                yp
            )

            valores.append(
                valor
            )

        except Exception:

            continue

    if len(
        valores
    ) == 0:

        return (
            np.nan,
            np.nan
        )

    inferior = np.percentile(
        valores,
        2.5
    )

    superior = np.percentile(
        valores,
        97.5
    )

    return (
        inferior,
        superior
    )


# =============================================================================
# 14. PIPELINE PRINCIPAL
# =============================================================================

def executar_pipeline():

    inicio_total = time.perf_counter()

    # =========================================================================
    # AUDITORIA DE ORIGEM
    # =========================================================================

    auditar_origem_dataset()

    # =========================================================================
    # CARREGAMENTO
    # =========================================================================

    X_sinais, y, ids = (
        carregar_dataset()
    )

    # =========================================================================
    # TRANSFORMAÇÃO WAVELET
    # =========================================================================
    #
    # A DWT é uma transformação determinística.
    #
    # Ela NÃO aprende parâmetros usando y.
    #
    # Portanto, realizar a DWT antes do K-Fold NÃO constitui vazamento.
    #
    # A seleção de características, entretanto, é feita depois da divisão.
    #
    # =========================================================================

    X_wavelet, nomes_wavelet = (
        construir_matriz_wavelet(
            X_sinais
        )
    )

    # =========================================================================
    # SALVA MATRIZ WAVELET
    # =========================================================================

    df_wavelet = pd.DataFrame(
        X_wavelet,
        columns=nomes_wavelet
    )

    df_wavelet["classe"] = y

    df_wavelet["ID_sinal"] = ids

    arquivo_wavelet = os.path.join(
        PASTA_WAVELET,
        "caracteristicas_wavelet.csv"
    )

    df_wavelet.to_csv(
        arquivo_wavelet,
        index=False
    )

    # =========================================================================
    # [NOVO] SALVA A REPRESENTAÇÃO BRUTA NO DOMÍNIO WAVELET
    # =========================================================================
    #
    # Diferente do bloco acima (que salva as 56 características estatísticas),
    # aqui são salvos os coeficientes da DWT propriamente ditos, nível a
    # nível, para cada sinal — ou seja, os dados de processamento do sinal
    # no domínio da transformada, antes de qualquer extração de
    # característica.
    #
    # =========================================================================

    df_dominio_wavelet = (
        construir_matriz_dominio_wavelet(
            X_sinais,
            y,
            ids
        )
    )

    arquivo_dominio_wavelet = os.path.join(
        PASTA_DOMINIO_WAVELET,
        "representacao_dominio_wavelet.csv"
    )

    df_dominio_wavelet.to_csv(
        arquivo_dominio_wavelet,
        index=False
    )

    print()

    print(
        f"Representação no domínio wavelet salva em: "
        f"{arquivo_dominio_wavelet}"
    )

    # =========================================================================
    # STRATIFIED K-FOLD
    # =========================================================================

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados_folds = []

    caracteristicas_folds = []

    y_true_global = []

    y_pred_global = []

    y_prob_global = []

    classes = np.unique(
        y
    )

    # =========================================================================
    # LOOP DOS FOLDS
    # =========================================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(

        skf.split(
            X_wavelet,
            y
        ),

        start=1
    ):

        print()
        print("=" * 80)
        print(
            f"FOLD {fold}/{N_SPLITS}"
        )
        print("=" * 80)

        inicio_fold = (
            time.perf_counter()
        )

        # =====================================================================
        # AUDITORIA
        # =====================================================================

        auditar_fold(
            train_idx,
            test_idx,
            ids,
            fold
        )

        # =====================================================================
        # SEPARAÇÃO
        # =====================================================================

        X_train = X_wavelet[
            train_idx
        ]

        X_test = X_wavelet[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # =====================================================================
        # CORRELAÇÃO
        # =====================================================================
        #
        # IMPORTANTE:
        #
        # O filtro usa SOMENTE X_train.
        #
        # X_test não participa da escolha.
        #
        # =====================================================================

        (
            indices_corr,
            nomes_corr
        ) = filtro_correlacao(

            X_train,
            nomes_wavelet
        )

        X_train_corr = (
            X_train[
                :,
                indices_corr
            ]
        )

        X_test_corr = (
            X_test[
                :,
                indices_corr
            ]
        )

        # =====================================================================
        # FISHER
        # =====================================================================
        #
        # O Fisher usa SOMENTE:
        #
        #     X_train
        #     y_train
        #
        # =====================================================================

        (
            indices_fisher,
            nomes_fisher,
            scores_fisher
        ) = selecionar_fisher(

            X_train_corr,
            y_train,
            nomes_corr
        )

        X_train_selected = (
            X_train_corr[
                :,
                indices_fisher
            ]
        )

        X_test_selected = (
            X_test_corr[
                :,
                indices_fisher
            ]
        )

        # =====================================================================
        # NORMALIZAÇÃO
        # =====================================================================
        #
        # fit_transform SOMENTE no treino.
        #
        # O teste utiliza apenas transform().
        #
        # =====================================================================

        print()
        print(
            "ETAPA 4 - NORMALIZAÇÃO"
        )

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train_selected
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test_selected
            )
        )

        # =====================================================================
        # RANDOM FOREST
        # =====================================================================

        print()
        print(
            "ETAPA 5 - TREINAMENTO DO RANDOM FOREST"
        )

        modelo = RandomForestClassifier(

            n_estimators=N_ESTIMATORS,

            random_state=RANDOM_STATE,

            n_jobs=N_JOBS,

            class_weight=None
        )

        modelo.fit(
            X_train_scaled,
            y_train
        )

        # =====================================================================
        # PREDIÇÃO
        # =====================================================================

        print()
        print(
            "Classificando conjunto de validação..."
        )

        y_pred = modelo.predict(
            X_test_scaled
        )

        y_prob = modelo.predict_proba(
            X_test_scaled
        )

        # =====================================================================
        # MÉTRICAS
        # =====================================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        kappa = cohen_kappa_score(
            y_test,
            y_pred
        )

        # ---------------------------------------------------------------------
        # ROC-AUC
        # ---------------------------------------------------------------------

        y_test_bin = label_binarize(
            y_test,
            classes=classes
        )

        roc_auc = roc_auc_score(

            y_test_bin,

            y_prob,

            average="macro",

            multi_class="ovr"
        )

        # ---------------------------------------------------------------------
        # LOG LOSS
        # ---------------------------------------------------------------------

        logloss = log_loss(
            y_test,
            y_prob,
            labels=classes
        )

        # ---------------------------------------------------------------------
        # TEMPO
        # ---------------------------------------------------------------------

        tempo_fold = (
            time.perf_counter()
            -
            inicio_fold
        )

        # =====================================================================
        # EXIBIÇÃO
        # =====================================================================

        print()

        print(
            f"Accuracy          : "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision Macro   : "
            f"{precision:.4f}"
        )

        print(
            f"Recall Macro      : "
            f"{recall:.4f}"
        )

        print(
            f"F1 Macro          : "
            f"{f1:.4f}"
        )

        print(
            f"MCC               : "
            f"{mcc:.4f}"
        )

        print(
            f"Cohen's Kappa     : "
            f"{kappa:.4f}"
        )

        print(
            f"ROC-AUC Macro     : "
            f"{roc_auc:.4f}"
        )

        print(
            f"Log Loss          : "
            f"{logloss:.4f}"
        )

        print(
            f"Tempo do fold     : "
            f"{tempo_fold:.2f} s"
        )

        # =====================================================================
        # RESULTADOS DO FOLD
        # =====================================================================

        resultados_folds.append({

            "Fold":
                fold,

            "Accuracy":
                accuracy,

            "Precision_macro":
                precision,

            "Recall_macro":
                recall,

            "F1_macro":
                f1,

            "MCC":
                mcc,

            "Cohen_Kappa":
                kappa,

            "ROC_AUC_macro":
                roc_auc,

            "Log_Loss":
                logloss,

            "Features_Wavelet":
                X_wavelet.shape[1],

            "Features_apos_correlacao":
                X_train_corr.shape[1],

            "Features_Fisher":
                X_train_selected.shape[1],

            "N_treino":
                len(train_idx),

            "N_teste":
                len(test_idx),

            "Tempo_s":
                tempo_fold
        })

        # =====================================================================
        # CARACTERÍSTICAS SELECIONADAS
        # =====================================================================

        for ordem, (
            nome,
            score
        ) in enumerate(

            zip(
                nomes_fisher,
                scores_fisher
            ),

            start=1
        ):

            caracteristicas_folds.append({

                "Fold":
                    fold,

                "Ordem":
                    ordem,

                "Caracteristica":
                    nome,

                "Fisher_Score":
                    score
            })

        # =====================================================================
        # PREDIÇÕES OOF
        # =====================================================================
        #
        # Cada amostra aparece exatamente uma vez como teste.
        #
        # =====================================================================

        y_true_global.extend(
            y_test
        )

        y_pred_global.extend(
            y_pred
        )

        y_prob_global.append(
            y_prob
        )

    # =========================================================================
    # CONVERSÃO
    # =========================================================================

    y_true_global = np.array(
        y_true_global
    )

    y_pred_global = np.array(
        y_pred_global
    )

    y_prob_global = np.vstack(
        y_prob_global
    )

    # =========================================================================
    # AUDITORIA DAS PREDIÇÕES OOF
    # =========================================================================

    if len(
        y_true_global
    ) != len(y):

        raise RuntimeError(
            "\nERRO NA AUDITORIA OOF:\n"
            "O número de predições OOF não corresponde "
            "ao número total de sinais."
        )

    print()
    print("=" * 80)
    print("AUDITORIA DAS PREDIÇÕES OOF")
    print("=" * 80)

    print()

    print(
        f"Sinais originais : {len(y)}"
    )

    print(
        f"Predições OOF    : "
        f"{len(y_true_global)}"
    )

    print()

    print(
        "[OK] Cada amostra possui uma predição OOF."
    )

    # =========================================================================
    # MÉTRICAS GLOBAIS
    # =========================================================================

    accuracy_global = accuracy_score(
        y_true_global,
        y_pred_global
    )

    precision_global = precision_score(
        y_true_global,
        y_pred_global,
        average="macro",
        zero_division=0
    )

    recall_global = recall_score(
        y_true_global,
        y_pred_global,
        average="macro",
        zero_division=0
    )

    f1_global = f1_score(
        y_true_global,
        y_pred_global,
        average="macro",
        zero_division=0
    )

    mcc_global = matthews_corrcoef(
        y_true_global,
        y_pred_global
    )

    kappa_global = cohen_kappa_score(
        y_true_global,
        y_pred_global
    )

    y_global_bin = label_binarize(
        y_true_global,
        classes=classes
    )

    roc_auc_global = roc_auc_score(

        y_global_bin,

        y_prob_global,

        average="macro",

        multi_class="ovr"
    )

    logloss_global = log_loss(

        y_true_global,

        y_prob_global,

        labels=classes
    )

    # =========================================================================
    # MATRIZ DE CONFUSÃO
    # =========================================================================

    cm = confusion_matrix(

        y_true_global,

        y_pred_global,

        labels=classes
    )

    # =========================================================================
    # RELATÓRIO POR CLASSE
    # =========================================================================

    relatorio_classes = classification_report(

        y_true_global,

        y_pred_global,

        labels=classes,

        target_names=classes,

        output_dict=True,

        zero_division=0
    )

    df_classes = pd.DataFrame(
        relatorio_classes
    ).transpose()

    arquivo_classes = os.path.join(

        PASTA_METRICAS,

        "metricas_por_classe_Wavelet_RF.csv"
    )

    df_classes.to_csv(
        arquivo_classes
    )


    def salvar_matriz_confusao_fft_rf(
        cm,
        classes,
        caminho,
        titulo
    ):

        fig, ax = plt.subplots(
            figsize=(9, 8)
        )

        maior_valor = np.max(cm)

        ax.imshow(
            np.zeros_like(cm),
            cmap="Blues",
            vmin=0,
            vmax=maior_valor
        )

        for i in range(len(classes)):

            for j in range(len(classes)):

                valor = cm[i, j]

                if i == j:

                    intensidade = (
                        valor / maior_valor
                        if maior_valor > 0
                        else 0
                    )

                    cor = plt.cm.Blues(
                        0.15 + 0.75 * intensidade
                    )

                else:

                    max_erro = np.max(
                        cm - np.diag(np.diag(cm))
                    )

                    intensidade = (
                        valor / max_erro
                        if max_erro > 0
                        else 0
                    )

                    cor = plt.cm.Oranges(
                        0.05 + 0.80 * intensidade
                    )

                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor=cor,
                        edgecolor="white",
                        linewidth=1
                    )
                )

                cor_texto = (
                    "white"
                    if intensidade > 0.55
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    str(valor),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=cor_texto
                )

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Classe Predita")
        ax.set_ylabel("Classe Real")
        ax.set_title(titulo)

        plt.tight_layout()

        plt.savefig(
            caminho,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =========================================================================
    # MATRIZ DE CONFUSÃO
    # =========================================================================

    arquivo_cm = os.path.join(

        PASTA_MATRIZ,

        "matriz_confusao_Wavelet_RandomForest.png"
    )

    salvar_matriz_confusao_fft_rf(
        cm,
        classes,
        arquivo_cm,
        "Matriz de Confusão - Wavelet + Random Forest"
    )

    # =========================================================================
    # CURVA ROC
    # =========================================================================

    plt.figure(
        figsize=(8, 7)
    )

    for i, classe in enumerate(
        classes
    ):

        fpr, tpr, _ = roc_curve(

            y_global_bin[:, i],

            y_prob_global[:, i]
        )

        roc_valor = auc(
            fpr,
            tpr
        )

        plt.plot(

            fpr,

            tpr,

            label=(
                f"{classe} "
                f"(AUC = {roc_valor:.3f})"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel(
        "Taxa de Falsos Positivos"
    )

    plt.ylabel(
        "Taxa de Verdadeiros Positivos"
    )

    plt.title(
        "Curva ROC - Wavelet + Random Forest"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    arquivo_roc = os.path.join(

        PASTA_ROC,

        "curva_ROC_Wavelet_RandomForest.png"
    )

    plt.savefig(

        arquivo_roc,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    # =========================================================================
    # PRECISION-RECALL
    # =========================================================================

    plt.figure(
        figsize=(8, 7)
    )

    for i, classe in enumerate(
        classes
    ):

        precision_curve, recall_curve, _ = (
            precision_recall_curve(

                y_global_bin[:, i],

                y_prob_global[:, i]
            )
        )

        ap = average_precision_score(

            y_global_bin[:, i],

            y_prob_global[:, i]
        )

        plt.plot(

            recall_curve,

            precision_curve,

            label=(
                f"{classe} "
                f"(AP = {ap:.3f})"
            )
        )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Curva Precision-Recall - Wavelet + Random Forest"
    )

    plt.legend(
        loc="lower left"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    arquivo_pr = os.path.join(

        PASTA_PR,

        "curva_Precision_Recall_Wavelet_RandomForest.png"
    )

    plt.savefig(

        arquivo_pr,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

    # =========================================================================
    # MÉTRICAS POR FOLD
    # =========================================================================

    df_folds = pd.DataFrame(
        resultados_folds
    )

    arquivo_folds = os.path.join(

        PASTA_METRICAS,

        "metricas_por_fold_Wavelet_RandomForest.csv"
    )

    df_folds.to_csv(

        arquivo_folds,

        index=False
    )

    # =========================================================================
    # MÉTRICAS FINAIS
    # =========================================================================

    df_finais = pd.DataFrame([
        {
            "Accuracy": accuracy_global,
            "Precision_macro": precision_global,
            "Recall_macro": recall_global,
            "F1_macro": f1_global,
            "MCC": mcc_global,
            "Cohen_Kappa": kappa_global,
            "ROC_AUC_macro": roc_auc_global,
            "Log_Loss": logloss_global
        }
    ])

    arquivo_finais = os.path.join(

        PASTA_METRICAS,

        "metricas_finais_Wavelet_RandomForest.csv"
    )

    df_finais.to_csv(

        arquivo_finais,

        index=False
    )

    # =========================================================================
    # BOOTSTRAP
    # =========================================================================

    print()
    print(
        "Calculando intervalos de confiança..."
    )

    metricas_bootstrap = {

        "Accuracy":
            accuracy_score,

        "Precision Macro":
            lambda yt, yp:
                precision_score(

                    yt,

                    yp,

                    average="macro",

                    zero_division=0
                ),

        "Recall Macro":
            lambda yt, yp:
                recall_score(

                    yt,

                    yp,

                    average="macro",

                    zero_division=0
                ),

        "F1 Macro":
            lambda yt, yp:
                f1_score(

                    yt,

                    yp,

                    average="macro",

                    zero_division=0
                ),

        "MCC":
            matthews_corrcoef,

        "Cohen's Kappa":
            cohen_kappa_score
    }

    resultados_bootstrap = []

    for nome, funcao in (
        metricas_bootstrap.items()
    ):

        inferior, superior = (
            bootstrap_intervalo(

                y_true_global,

                y_pred_global,

                funcao,

                N_BOOTSTRAP
            )
        )

        resultados_bootstrap.append({

            "Metrica":
                nome,

            "IC_95_inferior":
                inferior,

            "IC_95_superior":
                superior
        })

    df_bootstrap = pd.DataFrame(
        resultados_bootstrap
    )

    arquivo_bootstrap = os.path.join(

        PASTA_METRICAS,

        "intervalos_confianca_bootstrap_Wavelet_RF.csv"
    )

    df_bootstrap.to_csv(

        arquivo_bootstrap,

        index=False
    )

    # =========================================================================
    # CARACTERÍSTICAS SELECIONADAS
    # =========================================================================

    df_caracteristicas = pd.DataFrame(
        caracteristicas_folds
    )

    arquivo_caracteristicas = os.path.join(

        PASTA_METRICAS,

        "caracteristicas_Wavelet_selecionadas_por_fold.csv"
    )

    df_caracteristicas.to_csv(

        arquivo_caracteristicas,

        index=False
    )

    # =========================================================================
    # TEMPO
    # =========================================================================

    tempo_total = (

        time.perf_counter()

        -

        inicio_total
    )

    # =========================================================================
    # RELATÓRIO DE AUDITORIA
    # =========================================================================

    arquivo_auditoria = os.path.join(

        PASTA_AUDITORIA,

        "auditoria_vazamento_dados.txt"
    )

    with open(

        arquivo_auditoria,

        "w",

        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "=" * 80 + "\n"
        )

        arquivo.write(
            "AUDITORIA DE VAZAMENTO DE DADOS\n"
        )

        arquivo.write(
            "=" * 80 + "\n\n"
        )

        arquivo.write(
            "ORIGEM DOS DADOS\n"
        )

        arquivo.write(
            f"Pasta utilizada:\n"
            f"{PASTA_DATASET}\n\n"
        )

        arquivo.write(
            "Signal_Feature utilizado: SIM\n"
        )

        arquivo.write(
            "Dataset_Train utilizado: NAO\n"
        )

        arquivo.write(
            "Dataset_train_class utilizado: NAO\n"
        )

        arquivo.write(
            "index_total utilizado: NAO\n\n"
        )

        arquivo.write(
            "TRANSFORMACOES\n"
        )

        arquivo.write(
            "DWT: deterministica e sem ajuste de parametros\n"
        )

        arquivo.write(
            "Filtro de correlacao: ajustado somente no treino\n"
        )

        arquivo.write(
            "Fisher: calculado somente no treino\n"
        )

        arquivo.write(
            "StandardScaler: fit somente no treino\n"
        )

        arquivo.write(
            "Random Forest: treinado somente no treino\n\n"
        )

        arquivo.write(
            "VALIDACAO\n"
        )

        arquivo.write(
            f"Numero de folds: {N_SPLITS}\n"
        )

        arquivo.write(
            f"Random state: {RANDOM_STATE}\n\n"
        )

        arquivo.write(
            "PREDICOES OOF\n"
        )

        arquivo.write(
            f"Sinais totais: {len(y)}\n"
        )

        arquivo.write(
            f"Predicoes OOF: {len(y_true_global)}\n\n"
        )

        arquivo.write(
            "RESULTADO DA AUDITORIA\n"
        )

        arquivo.write(
            "Nenhuma intersecao entre treino e teste foi detectada.\n"
        )

        arquivo.write(
            "As etapas supervisionadas foram ajustadas somente "
            "com os dados de treinamento de cada fold.\n"
        )

    # =========================================================================
    # RELATÓRIO PRINCIPAL
    # =========================================================================

    arquivo_relatorio = os.path.join(

        PASTA_RELATORIO,

        "relatorio_Wavelet_RandomForest.txt"
    )

    with open(

        arquivo_relatorio,

        "w",

        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "=" * 80 + "\n"
        )

        arquivo.write(
            "RELATORIO - WAVELET + RANDOM FOREST\n"
        )

        arquivo.write(
            "=" * 80 + "\n\n"
        )

        arquivo.write(
            "ORIGEM DOS DADOS\n"
        )

        arquivo.write(
            "Signal_Feature: UTILIZADO\n"
        )

        arquivo.write(
            "Dataset_Train: NAO UTILIZADO\n"
        )

        arquivo.write(
            "Dataset_train_class: NAO UTILIZADO\n"
        )

        arquivo.write(
            "index_total: NAO UTILIZADO\n\n"
        )

        arquivo.write(
            "CONFIGURACOES\n"
        )

        arquivo.write(
            f"Wavelet: {WAVELET}\n"
        )

        arquivo.write(
            f"Nivel Wavelet: {NIVEL_WAVELET}\n"
        )

        arquivo.write(
            f"Frequencia amostragem: {FS} Hz\n"
        )

        arquivo.write(
            f"Frequencia fundamental: {FUNDAMENTAL} Hz\n"
        )

        arquivo.write(
            f"Limite correlacao: "
            f"{LIMITE_CORRELACAO}\n"
        )

        arquivo.write(
            f"Caracteristicas Fisher: "
            f"{N_FEATURES_FISHER}\n"
        )

        arquivo.write(
            f"Random Forest estimators: "
            f"{N_ESTIMATORS}\n"
        )

        arquivo.write(
            f"K-Fold: {N_SPLITS}\n"
        )

        arquivo.write(
            f"Random state: {RANDOM_STATE}\n\n"
        )

        arquivo.write(
            "METRICAS FINAIS\n"
        )

        arquivo.write(
            f"Accuracy: "
            f"{accuracy_global:.6f}\n"
        )

        arquivo.write(
            f"Precision Macro: "
            f"{precision_global:.6f}\n"
        )

        arquivo.write(
            f"Recall Macro: "
            f"{recall_global:.6f}\n"
        )

        arquivo.write(
            f"F1 Macro: "
            f"{f1_global:.6f}\n"
        )

        arquivo.write(
            f"MCC: "
            f"{mcc_global:.6f}\n"
        )

        arquivo.write(
            f"Cohen's Kappa: "
            f"{kappa_global:.6f}\n"
        )

        arquivo.write(
            f"ROC-AUC Macro: "
            f"{roc_auc_global:.6f}\n"
        )

        arquivo.write(
            f"Log Loss: "
            f"{logloss_global:.6f}\n"
        )

        arquivo.write(
            f"\nTempo total: "
            f"{tempo_total:.2f} segundos\n"
        )

        arquivo.write(
            "\nMATRIZ DE CONFUSAO\n\n"
        )

        arquivo.write(
            str(cm)
        )

        arquivo.write(
            "\n\nRELATORIO POR CLASSE\n\n"
        )

        arquivo.write(

            classification_report(

                y_true_global,

                y_pred_global,

                labels=classes,

                target_names=classes,

                zero_division=0
            )
        )

    # =========================================================================
    # EXIBIÇÃO FINAL
    # =========================================================================

    print()
    print("=" * 80)
    print("RESULTADOS FINAIS")
    print("=" * 80)

    print()

    print(
        f"Accuracy          : "
        f"{accuracy_global:.4f} "
        f"({accuracy_global * 100:.2f}%)"
    )

    print(
        f"Precision Macro   : "
        f"{precision_global:.4f} "
        f"({precision_global * 100:.2f}%)"
    )

    print(
        f"Recall Macro      : "
        f"{recall_global:.4f} "
        f"({recall_global * 100:.2f}%)"
    )

    print(
        f"F1 Macro          : "
        f"{f1_global:.4f} "
        f"({f1_global * 100:.2f}%)"
    )

    print(
        f"MCC               : "
        f"{mcc_global:.4f}"
    )

    print(
        f"Cohen's Kappa     : "
        f"{kappa_global:.4f}"
    )

    print(
        f"ROC-AUC Macro     : "
        f"{roc_auc_global:.4f}"
    )

    print(
        f"Log Loss          : "
        f"{logloss_global:.4f}"
    )

    print()

    print(
        "Matriz de confusão:"
    )

    print(
        cm
    )

    print()

    print(
        "Auditoria de vazamento:"
    )

    print(
        arquivo_auditoria
    )

    print()

    print(
        f"Tempo total: "
        f"{tempo_total:.2f} segundos"
    )

    print(
        f"Tempo total: "
        f"{tempo_total / 60:.2f} minutos"
    )

    print()
    print("=" * 80)

    return {

        "accuracy":
            accuracy_global,

        "precision_macro":
            precision_global,

        "recall_macro":
            recall_global,

        "f1_macro":
            f1_global,

        "mcc":
            mcc_global,

        "kappa":
            kappa_global,

        "roc_auc":
            roc_auc_global,

        "log_loss":
            logloss_global,

        "confusion_matrix":
            cm,

        "tempo":
            tempo_total,

        "arquivo_auditoria":
            arquivo_auditoria
    }



# =============================================================================
# 15. ANÁLISE APROFUNDADA DE SEPARABILIDADE E AUDITORIA DO RESULTADO
# =============================================================================
# IMPORTANTE: toda a análise abaixo é EXPLORATÓRIA. Ela não alimenta o modelo
# principal e, portanto, não altera as métricas OOF do pipeline original.

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, permutation_test_score
from scipy.spatial.distance import mahalanobis

PASTA_ANALISE = os.path.join(PASTA_RESULTADOS, 'Analise_Separabilidade')
PASTA_SINAIS = os.path.join(PASTA_ANALISE, '01_Sinais')
PASTA_COEF = os.path.join(PASTA_ANALISE, '02_Wavelet')
PASTA_FEATURES = os.path.join(PASTA_ANALISE, '03_Distribuicoes')
PASTA_PROJECOES = os.path.join(PASTA_ANALISE, '04_Projecoes')
PASTA_DIST = os.path.join(PASTA_ANALISE, '05_Distancias')
PASTA_VALIDACAO = os.path.join(PASTA_ANALISE, '06_Validacao')
for _p in [PASTA_ANALISE, PASTA_SINAIS, PASTA_COEF, PASTA_FEATURES, PASTA_PROJECOES, PASTA_DIST, PASTA_VALIDACAO]:
    os.makedirs(_p, exist_ok=True)


def _salvar(fig, caminho):
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotar_sinais(X, y):
    t = np.arange(X.shape[1]) / FS
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 14), sharex=True)
    for ax, classe in zip(axes, CLASSES):
        sinais = X[y == classe]
        ax.plot(t, sinais[0], alpha=0.85, label='Sinal representativo')
        ax.plot(t, sinais.mean(axis=0), linewidth=1.5, label='Média da classe')
        ax.set_ylabel(classe)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Tempo (s)')
    _salvar(fig, os.path.join(PASTA_SINAIS, 'sinais_representativos_e_media.png'))

    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 14), sharex=True)
    for ax, classe in zip(axes, CLASSES):
        sinais = X[y == classe]
        media, desvio = sinais.mean(axis=0), sinais.std(axis=0)
        ax.plot(t, media, label='Média')
        ax.fill_between(t, media-desvio, media+desvio, alpha=0.25, label='± 1 desvio padrão')
        ax.set_ylabel(classe); ax.grid(alpha=0.25); ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Tempo (s)')
    _salvar(fig, os.path.join(PASTA_SINAIS, 'media_desvio_por_classe.png'))


def plotar_wavelet(X, y):
    fig, axes = plt.subplots(len(CLASSES), NIVEL_WAVELET + 1, figsize=(18, 14))
    for i, classe in enumerate(CLASSES):
        sinal = X[y == classe][0]
        coefs = pywt.wavedec(sinal, WAVELET, level=NIVEL_WAVELET)
        nomes = [f'A{NIVEL_WAVELET}'] + [f'D{n}' for n in range(NIVEL_WAVELET, 0, -1)]
        for j, (coef, nome) in enumerate(zip(coefs, nomes)):
            axes[i, j].plot(coef, linewidth=0.7)
            if i == 0: axes[i, j].set_title(nome)
            if j == 0: axes[i, j].set_ylabel(classe)
            axes[i, j].grid(alpha=0.2)
    _salvar(fig, os.path.join(PASTA_COEF, 'coeficientes_wavelet_representativos.png'))


def auditoria_similaridade(X, y, ids):
    # -------------------------------------------------------------------------
    # 1) Duplicatas exatas: contabiliza grupos e linhas duplicadas extras
    # -------------------------------------------------------------------------
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    grupos_dup = int(np.sum(counts > 1))
    linhas_dup_extras = int(np.sum(counts[counts > 1] - 1))

    grupos_mistos = 0
    pares_duplicados = []
    for g in np.where(counts > 1)[0]:
        idx = np.where(inverse == g)[0]
        classes = np.unique(y[idx])
        if len(classes) > 1:
            grupos_mistos += 1
        for a in idx:
            for b in idx:
                if a < b:
                    pares_duplicados.append({
                        'id_1': ids[a], 'classe_1': y[a],
                        'id_2': ids[b], 'classe_2': y[b],
                        'mesma_classe': bool(y[a] == y[b])
                    })
    pd.DataFrame(pares_duplicados).to_csv(
        os.path.join(PASTA_DIST, 'pares_duplicados_exatos.csv'), index=False
    )

    # -------------------------------------------------------------------------
    # 2) Vizinho mais próximo no sinal bruto padronizado por posição temporal
    # -------------------------------------------------------------------------
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(Xs)
    dist, ind = nn.kneighbors(Xs)
    nn_mesma = (y[ind[:, 1]] == y)

    df = pd.DataFrame({
        'id': ids,
        'classe': y,
        'vizinho_mais_proximo': ids[ind[:, 1]],
        'classe_vizinho': y[ind[:, 1]],
        'distancia': dist[:, 1],
        'mesma_classe': nn_mesma
    })
    df.to_csv(os.path.join(PASTA_DIST, 'vizinho_mais_proximo_sinal_bruto.csv'), index=False)

    resumo = {
        'grupos_duplicados_exatos': grupos_dup,
        'linhas_duplicadas_extras': linhas_dup_extras,
        'grupos_duplicados_com_classes_diferentes': grupos_mistos,
        'pureza_1NN_sinal_bruto': float(nn_mesma.mean()),
        'distancia_1NN_min': float(dist[:, 1].min()),
        'distancia_1NN_mediana': float(np.median(dist[:, 1])),
        'distancia_1NN_media': float(np.mean(dist[:, 1]))
    }
    pd.DataFrame([resumo]).to_csv(
        os.path.join(PASTA_DIST, 'auditoria_similaridade_sinais.csv'), index=False
    )
    return resumo


def selecionar_features_exploratorio(Xw, y, nomes):
    # Somente para visualização global. NÃO é usado para calcular a métrica OOF.
    idx_corr, _ = filtro_correlacao(Xw, nomes)
    idx_corr = np.asarray(idx_corr, dtype=int)
    Xc = Xw[:, idx_corr]
    scores = fisher_score(Xc, y)
    ordem = np.argsort(scores)[::-1][:N_FEATURES_FISHER]
    idx_final = np.asarray(idx_corr)[ordem]
    return Xw[:, idx_final], [nomes[i] for i in idx_final], idx_final


def plotar_distribuicoes(X, y, nomes, sufixo):
    df = pd.DataFrame(X, columns=nomes); df['classe'] = y
    for nome in nomes:
        fig, ax = plt.subplots(figsize=(9, 5))
        grupos = [df.loc[df['classe']==c, nome].values for c in CLASSES]
        ax.boxplot(grupos, tick_labels=CLASSES, showfliers=True)
        ax.set_title(f'Distribuição por classe: {nome} ({sufixo})')
        ax.set_ylabel(nome); ax.grid(axis='y', alpha=0.25)
        _salvar(fig, os.path.join(PASTA_FEATURES, f'boxplot_{sufixo}_{nome}.png'))


def plotar_projecoes(X, y, titulo, sufixo):
    Xs = StandardScaler().fit_transform(X)
    resultados = {}
    # PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(Xs)
    resultados['PCA_var_exp_PC1_PC2'] = float(pca.explained_variance_ratio_[:2].sum())
    for metodo, dados in [('PCA', Z)]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for c in CLASSES:
            m = y == c; ax.scatter(dados[m,0], dados[m,1], s=25, alpha=0.75, label=c)
        ax.set_title(f'{metodo} - {titulo}'); ax.set_xlabel(f'{metodo} 1'); ax.set_ylabel(f'{metodo} 2'); ax.legend(); ax.grid(alpha=0.2)
        _salvar(fig, os.path.join(PASTA_PROJECOES, f'{metodo}_{sufixo}.png'))
    # LDA supervisionada: máximo de 6 componentes para 7 classes
    lda = LinearDiscriminantAnalysis(n_components=2)
    Z = lda.fit_transform(Xs, y)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in CLASSES:
        m = y == c; ax.scatter(Z[m,0], Z[m,1], s=25, alpha=0.75, label=c)
    ax.set_title(f'LDA - {titulo} (exploratório)'); ax.set_xlabel('LD1'); ax.set_ylabel('LD2'); ax.legend(); ax.grid(alpha=0.2)
    _salvar(fig, os.path.join(PASTA_PROJECOES, f'LDA_{sufixo}.png'))
    # t-SNE apenas exploratório
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, (len(X)-1)//3)), init='pca', learning_rate='auto', random_state=RANDOM_STATE)
    Z = tsne.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in CLASSES:
        m = y == c; ax.scatter(Z[m,0], Z[m,1], s=25, alpha=0.75, label=c)
    ax.set_title(f't-SNE - {titulo} (exploratório)'); ax.set_xlabel('Dimensão 1'); ax.set_ylabel('Dimensão 2'); ax.legend(); ax.grid(alpha=0.2)
    _salvar(fig, os.path.join(PASTA_PROJECOES, f'tSNE_{sufixo}.png'))
    return resultados


def metricas_separabilidade(X, y, sufixo):
    Xs = StandardScaler().fit_transform(X)
    D = pairwise_distances(Xs, metric='euclidean')
    mesma = y[:,None] == y[None,:]
    diag = np.eye(len(y), dtype=bool)
    intra = D[mesma & ~diag]
    inter = D[~mesma]
    sil_global = float(silhouette_score(Xs, y, metric='euclidean'))
    sil = silhouette_samples(Xs, y, metric='euclidean')
    linhas = [{'escopo':'global','silhouette':sil_global,'distancia_intra_media':float(intra.mean()),'distancia_inter_media':float(inter.mean()),'razao_inter_intra':float(inter.mean()/intra.mean())}]
    for c in CLASSES:
        m=y==c; linhas.append({'escopo':c,'silhouette':float(sil[m].mean()),'distancia_intra_media':np.nan,'distancia_inter_media':np.nan,'razao_inter_intra':np.nan})
    pd.DataFrame(linhas).to_csv(os.path.join(PASTA_DIST, f'metricas_separabilidade_{sufixo}.csv'), index=False)
    fig, ax = plt.subplots(figsize=(9,5)); ax.hist(intra, bins=50, alpha=0.6, label='Intra-classe'); ax.hist(inter, bins=50, alpha=0.6, label='Interclasse'); ax.set_title(f'Distâncias intra vs interclasse - {sufixo}'); ax.set_xlabel('Distância euclidiana padronizada'); ax.legend(); ax.grid(alpha=0.2)
    _salvar(fig, os.path.join(PASTA_DIST, f'distancias_{sufixo}.png'))
    # Pureza dos vizinhos
    purezas=[]
    for k in [1,3,5,10]:
        nn=NearestNeighbors(n_neighbors=k+1).fit(Xs); _, ind=nn.kneighbors(Xs)
        purezas.append({'k':k,'pureza_media_mesma_classe':float(np.mean([(y[ind[i,1:]]==y[i]).mean() for i in range(len(y))]))})
    pd.DataFrame(purezas).to_csv(os.path.join(PASTA_DIST, f'pureza_vizinhos_{sufixo}.csv'), index=False)
    return linhas[0]


def validar_classificadores(Xw, y):
    # Comparação independente usando TODAS as 56 features; nenhum método abaixo
    # usa o resultado da seleção global exploratória.
    cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    modelos={
        'RandomForest': RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS),
        'SVM_linear': Pipeline([('scaler',StandardScaler()),('clf',SVC(kernel='linear', C=1))]),
        'SVM_RBF': Pipeline([('scaler',StandardScaler()),('clf',SVC(kernel='rbf', C=1, gamma='scale'))]),
        'kNN': Pipeline([('scaler',StandardScaler()),('clf',KNeighborsClassifier(n_neighbors=5))]),
        'LogReg': Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=5000))])
    }
    resultados=[]
    for nome, modelo in modelos.items():
        pred=cross_val_predict(modelo, Xw, y, cv=cv, n_jobs=N_JOBS)
        resultados.append({'modelo':nome,'accuracy':accuracy_score(y,pred),'f1_macro':f1_score(y,pred,average='macro'),'mcc':matthews_corrcoef(y,pred),'erros':int(np.sum(y!=pred))})
    pd.DataFrame(resultados).to_csv(os.path.join(PASTA_VALIDACAO,'comparacao_classificadores_56_features.csv'),index=False)
    # Teste de permutação no RF: H0 = rótulos não possuem relação com as features
    rf=RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    score, perm_scores, p = permutation_test_score(rf, Xw, y, scoring='accuracy', cv=cv, n_permutations=200, n_jobs=N_JOBS, random_state=RANDOM_STATE)
    pd.DataFrame({'score_real':[score],'p_valor_permutacao':[p],'media_scores_permutados':[perm_scores.mean()],'max_score_permutado':[perm_scores.max()]}).to_csv(os.path.join(PASTA_VALIDACAO,'teste_permutacao_RF.csv'),index=False)
    fig, ax=plt.subplots(figsize=(8,5)); ax.hist(perm_scores,bins=30,alpha=0.75); ax.axvline(score,linestyle='--',linewidth=2,label=f'Real={score:.4f}'); ax.set_title('Teste de permutação - Random Forest'); ax.set_xlabel('Accuracy CV'); ax.legend(); ax.grid(alpha=0.2)
    _salvar(fig, os.path.join(PASTA_VALIDACAO,'teste_permutacao_RF.png'))
    return resultados


def executar_analise_separabilidade():
    print('\n'+'='*80+'\nANÁLISE APROFUNDADA DE SEPARABILIDADE\n'+'='*80)
    X, y, ids = carregar_dataset()
    Xw, nomes = construir_matriz_wavelet(X)
    # Salva dados para reprodutibilidade
    pd.DataFrame(Xw, columns=nomes).assign(classe=y, id=ids).to_csv(os.path.join(PASTA_ANALISE,'matriz_56_features_completa.csv'),index=False)
    plotar_sinais(X,y)
    plotar_wavelet(X,y)
    auditoria_similaridade(X,y,ids)
    # 56 features: PCA e métricas
    sep56=metricas_separabilidade(Xw,y,'56_features')
    proj56=plotar_projecoes(Xw,y,'56 características Wavelet','56_features')
    # 14 features: EXPLORATÓRIO e explicitamente separado da validação OOF
    X14, nomes14, _ = selecionar_features_exploratorio(Xw,y,nomes)
    pd.DataFrame(X14,columns=nomes14).assign(classe=y,id=ids).to_csv(os.path.join(PASTA_ANALISE,'matriz_14_features_exploratoria.csv'),index=False)
    plotar_distribuicoes(X14,y,nomes14,'14_features')
    sep14=metricas_separabilidade(X14,y,'14_features')
    proj14=plotar_projecoes(X14,y,'14 características selecionadas globalmente para exploração','14_features')
    resultados=validar_classificadores(Xw,y)
    resumo={'separabilidade_56':sep56,'separabilidade_14_exploratoria':sep14,'pca_56':proj56,'pca_14':proj14,'classificadores':resultados,
            'AVISO':'A seleção global de 14 features é usada somente para visualização e nunca para reportar accuracy OOF.'}
    with open(os.path.join(PASTA_ANALISE,'resumo_analise.txt'),'w',encoding='utf-8') as f:
        import json; json.dump(resumo,f,ensure_ascii=False,indent=2)
    print('Análise concluída. Resultados em:', PASTA_ANALISE)


# =============================================================================
# 15. EXECUÇÃO COM CODECARBON
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print(
        "INICIANDO WAVELET + RANDOM FOREST"
    )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CodeCarbon
    # -------------------------------------------------------------------------

    tracker = EmissionsTracker(

        project_name=(
            "Wavelet_RandomForest_PQD"
        ),

        output_dir=(
            PASTA_CODECARBON
        ),

        measure_power_secs=1,

        save_to_file=True,

        log_level="error"
    )

    resultados = None

    try:

        print()
        print(
            "CodeCarbon iniciado."
        )

        print(
            "Diretório:"
        )

        print(
            PASTA_CODECARBON
        )

        print()

        tracker.start()

        resultados = executar_pipeline()

    finally:

        emissions = tracker.stop()

        if emissions is None:

            emissions = 0.0

        print()
        print("=" * 80)
        print("CODECARBON")
        print("=" * 80)

        print()

        print(
            f"Emissão estimada: "
            f"{emissions:.8f} kg CO2"
        )

        print(
            f"Emissão estimada: "
            f"{emissions * 1000:.4f} g CO2"
        )

        # ---------------------------------------------------------------------
        # Resumo CodeCarbon
        # ---------------------------------------------------------------------

        if resultados is not None:

            resumo_codecarbon = pd.DataFrame({

                "Modelo": [
                    "Wavelet + Random Forest"
                ],

                "Dataset": [
                    "Signal_Feature"
                ],

                "Dataset_Train_Utilizado": [
                    False
                ],

                "Emissao_kg_CO2": [
                    emissions
                ],

                "Emissao_g_CO2": [
                    emissions * 1000
                ],

                "Tempo_s": [
                    resultados["tempo"]
                ],

                "Accuracy": [
                    resultados["accuracy"]
                ],

                "F1_Macro": [
                    resultados["f1_macro"]
                ],

                "MCC": [
                    resultados["mcc"]
                ],

                "ROC_AUC_Macro": [
                    resultados["roc_auc"]
                ]
            })

            arquivo_resumo = os.path.join(

                PASTA_CODECARBON,

                "resumo_codecarbon_Wavelet_RF.csv"
            )

            resumo_codecarbon.to_csv(

                arquivo_resumo,

                index=False
            )

            print()

            print(
                "Resumo CodeCarbon:"
            )

            print(
                arquivo_resumo
            )

        print()
        print("=" * 80)
        print(
            "PROGRAMA FINALIZADO"
        )
        print("=" * 80)

# =============================================================================
# 16. EXECUÇÃO DA ANÁLISE EXPLORATÓRIA APÓS O EXPERIMENTO PRINCIPAL
# =============================================================================
# Esta etapa fica fora do CodeCarbon e fora do tempo do pipeline principal.
# Assim, o tempo/emissão do classificador não são contaminados pela geração de
# PCA, LDA, t-SNE, gráficos e teste de permutação.

if __name__ == "__main__":
    executar_analise_separabilidade()
