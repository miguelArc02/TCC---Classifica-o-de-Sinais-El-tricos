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
#     FILTRO CORRELAÇÃO                   |
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

PASTA_DATASET = (
    r"C:\Users\Rafael\Mestrado\Projeto de Pesquisa\Trabalho Murilo e Miguel\01 - Power_Quality_Disturbance_Dataset-master\Banco_de_Dados\CSV\Signal_Feature"
)


# -----------------------------------------------------------------------------
# PASTA DE RESULTADOS
# -----------------------------------------------------------------------------

PASTA_RESULTADOS = (
    r"C:\Users\Rafael\Mestrado\Projeto de Pesquisa\Trabalho Murilo e Miguel\01 - Power_Quality_Disturbance_Dataset-master\Codigos\4 - DWT_&_RF\Resultados\Resultados_REV_00"
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

N_JOBS = -1


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

    # =========================================================================
    # MATRIZ DE CONFUSÃO
    # =========================================================================

    plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Matriz de Confusão - Wavelet + Random Forest"
    )

    plt.xlabel(
        "Classe Predita"
    )

    plt.ylabel(
        "Classe Verdadeira"
    )

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=45
    )

    plt.yticks(
        range(len(classes)),
        classes
    )

    limite = cm.max() / 2

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(

                j,

                i,

                str(cm[i, j]),

                ha="center",

                va="center",

                color=(
                    "white"
                    if cm[i, j] > limite
                    else "black"
                )
            )

    plt.tight_layout()

    arquivo_cm = os.path.join(

        PASTA_MATRIZ,

        "matriz_confusao_Wavelet_RandomForest.png"
    )

    plt.savefig(

        arquivo_cm,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()

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

    df_finais = pd.DataFrame({

        "Metrica": [

            "Accuracy",

            "Precision Macro",

            "Recall Macro",

            "F1 Macro",

            "MCC",

            "Cohen's Kappa",

            "ROC-AUC Macro",

            "Log Loss"
        ],

        "Valor": [

            accuracy_global,

            precision_global,

            recall_global,

            f1_global,

            mcc_global,

            kappa_global,

            roc_auc_global,

            logloss_global
        ]
    })

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