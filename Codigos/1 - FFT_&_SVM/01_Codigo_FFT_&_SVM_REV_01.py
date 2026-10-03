# =============================================================================
# CLASSIFICAÇÃO DE DISTÚRBIOS DE QUALIDADE DE ENERGIA
#
# PIPELINE:
#
#       SINAL
#         |
#         v
#       FFT
#         |
#         v
#   FILTRO DE CORRELAÇÃO
#         |
#         v
# FISHER DISCRIMINANT RATIO
#         |
#         v
#    NORMALIZAÇÃO
#         |
#         v
#      SVM LINEAR
#         |
#         v
# STRATIFIED K-FOLD (5 FOLDS)
#         |
#         +----------------------------+
#         |                            |
#         v                            v
#     PREDIÇÕES                  SCORES / PROBABILIDADES
#         |                            |
#         v                            v
# Accuracy                     ROC / ROC-AUC
# Precision                    Precision-Recall
# Recall                       Log Loss
# F1                           ...
# MCC
# Kappa
# Balanced Accuracy
#
# Além disso:
#
#       CodeCarbon
#           |
#           v
#      Emissão de CO2
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

from pathlib import Path

# Machine Learning
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

# Métricas
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report
)

# Bootstrap
from sklearn.utils import resample

# CodeCarbon
from codecarbon import EmissionsTracker


# =============================================================================
# 2. CONFIGURAÇÕES PRINCIPAIS
# =============================================================================

# -----------------------------------------------------------------------------
# ATENÇÃO:
# ALTERE SOMENTE ESTA PASTA PARA O LOCAL DO SEU DATASET
# -----------------------------------------------------------------------------

DATASET_DIR = Path(
    r"C:\Users\Rafael\Mestrado\Projeto de Pesquisa\Trabalho Murilo e Miguel"
    r"\01 - Power_Quality_Disturbance_Dataset-master"
    r"\Signal_Feature"
)


# -----------------------------------------------------------------------------
# Pasta onde os resultados serão salvos
# -----------------------------------------------------------------------------

RESULTS_DIR = Path(
    r"C:\Users\Rafael\Mestrado\Projeto de Pesquisa\Trabalho Murilo e Miguel"
    r"\01 - Power_Quality_Disturbance_Dataset-master"
    r"\Codigos\1 - FFT_&_SVM"
    r"\Resultados\Resultados_REV_02"
)


# -----------------------------------------------------------------------------
# Criar automaticamente as subpastas
# -----------------------------------------------------------------------------

DIR_METRICAS = RESULTS_DIR / "Metricas"
DIR_MATRIZ = RESULTS_DIR / "Matriz_Confusao"
DIR_ROC = RESULTS_DIR / "ROC"
DIR_PR = RESULTS_DIR / "Precision_Recall"
DIR_CODECARBON = RESULTS_DIR / "CodeCarbon"
DIR_RELATORIO = RESULTS_DIR / "Relatorio"
DIR_FISHER = RESULTS_DIR / "Fisher"


for directory in [
    RESULTS_DIR,
    DIR_METRICAS,
    DIR_MATRIZ,
    DIR_ROC,
    DIR_PR,
    DIR_CODECARBON,
    DIR_RELATORIO,
    DIR_FISHER
]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 3. CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================

# Frequência fundamental da rede
FUNDAMENTAL_FREQUENCY = 60.0

# Frequência de amostragem
SAMPLING_FREQUENCY = 15360.0

# Número de folds
N_SPLITS = 5

# Estado aleatório para reprodutibilidade
RANDOM_STATE = 42

# Número de características selecionadas pelo Fisher
N_FISHER_FEATURES = 14

# Limite do filtro de correlação
CORRELATION_THRESHOLD = 0.95

# Parâmetro C do SVM
SVM_C = 10.0


# -----------------------------------------------------------------------------
# Classes do problema
# -----------------------------------------------------------------------------

CLASS_NAMES = [
    "capc",
    "har",
    "normal",
    "notch",
    "sag",
    "spike",
    "swell"
]


# =============================================================================
# 4. CONFIGURAÇÕES DE EXIBIÇÃO
# =============================================================================

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# =============================================================================
# 5. FUNÇÃO PARA LOCALIZAR OS ARQUIVOS CSV
# =============================================================================

def encontrar_arquivo_classe(dataset_dir, classe):
    """
    Procura automaticamente o arquivo CSV correspondente à classe.

    O programa procura primeiro por nomes contendo exatamente o nome da classe.

    Exemplos aceitos:

        signal_capc.csv
        signal_capc_01.csv
        Dataset_feature_capc.csv
        capc.csv

    Caso existam vários arquivos, será escolhido o arquivo cujo nome
    possuir maior correspondência com a classe.
    """

    arquivos = list(dataset_dir.rglob("*.csv"))

    if len(arquivos) == 0:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV foi encontrado em:\n{dataset_dir}"
        )

    candidatos = []

    classe_lower = classe.lower()

    for arquivo in arquivos:

        nome = arquivo.stem.lower()

        # Procura o nome da classe como parte do nome do arquivo
        if classe_lower in nome:
            candidatos.append(arquivo)

    if len(candidatos) == 0:
        raise FileNotFoundError(
            f"\nNão foi encontrado arquivo CSV para a classe '{classe}'.\n"
            f"Pasta pesquisada:\n{dataset_dir}\n"
        )

    # Prioridade:
    # 1 - signal_classe
    # 2 - classe
    # 3 - demais correspondências

    candidatos.sort(
        key=lambda x: (
            0 if f"signal_{classe_lower}" in x.stem.lower() else
            1 if x.stem.lower() == classe_lower else
            2,
            len(x.name)
        )
    )

    return candidatos[0]


# =============================================================================
# 6. FUNÇÃO PARA CARREGAR UM ARQUIVO CSV
# =============================================================================

def carregar_csv(caminho):
    """
    Carrega um CSV e garante que os dados sejam numéricos.

    A função tenta lidar com arquivos contendo ou não cabeçalho.
    """

    try:

        # Primeira tentativa
        df = pd.read_csv(caminho)

        # Converte tudo para numérico
        df_numeric = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Remove colunas totalmente não numéricas
        df_numeric = df_numeric.dropna(
            axis=1,
            how="all"
        )

        # Remove linhas totalmente vazias
        df_numeric = df_numeric.dropna(
            axis=0,
            how="all"
        )

        dados = df_numeric.to_numpy(dtype=float)

        # Se encontrou dados válidos
        if dados.size > 0:
            return dados

    except Exception:
        pass


    # Segunda tentativa: arquivo puramente numérico
    try:

        dados = np.loadtxt(
            caminho,
            delimiter=","
        )

        return np.asarray(
            dados,
            dtype=float
        )

    except Exception as erro:

        raise RuntimeError(
            f"Não foi possível carregar o arquivo:\n{caminho}\n\n"
            f"Erro:\n{erro}"
        )


# =============================================================================
# 7. CARREGAMENTO DO DATASET
# =============================================================================

print()
print("=" * 80)
print("CARREGANDO DATASET")
print("=" * 80)
print()


X_lista = []
y_lista = []


for classe in CLASS_NAMES:

    arquivo = encontrar_arquivo_classe(
        DATASET_DIR,
        classe
    )

    dados = carregar_csv(
        arquivo
    )

    # -------------------------------------------------------------------------
    # Garantir que cada linha represente um sinal
    # -------------------------------------------------------------------------

    if dados.ndim == 1:
        dados = dados.reshape(1, -1)

    print(
        f"{classe:<10} -> "
        f"arquivo = {arquivo.name:<35} "
        f"shape = {dados.shape}"
    )

    # -------------------------------------------------------------------------
    # Cada linha representa um sinal
    # -------------------------------------------------------------------------

    X_lista.append(dados)

    # Cria o vetor de rótulos
    y_lista.extend(
        [classe] * dados.shape[0]
    )


# =============================================================================
# 8. VERIFICAÇÃO DO DATASET
# =============================================================================

# Verifica se todas as classes possuem o mesmo número de características
n_features_amostras = [
    x.shape[1]
    for x in X_lista
]

if len(set(n_features_amostras)) != 1:

    raise ValueError(
        "\nOs arquivos possuem números diferentes de amostras por sinal.\n"
        f"Características encontradas: {n_features_amostras}"
    )


# Junta todas as classes
X = np.vstack(X_lista)

# Converte rótulos para numpy
y = np.asarray(y_lista)


print()
print(f"Número total de sinais: {X.shape[0]}")
print(f"Número de amostras por sinal: {X.shape[1]}")
print(f"Número de classes: {len(np.unique(y))}")

print()
print("Distribuição das classes:")
print(
    pd.Series(y).value_counts().sort_index()
)


# =============================================================================
# 9. TRANSFORMADA RÁPIDA DE FOURIER
# =============================================================================

print()
print("=" * 80)
print("ETAPA 1 - TRANSFORMADA RÁPIDA DE FOURIER")
print("=" * 80)


N_SAMPLES = X.shape[1]

# -------------------------------------------------------------------------
# FFT
#
# rfft é utilizado porque o sinal original é real.
#
# Para N = 2560:
#
# número de componentes = N/2 + 1
#                      = 1280 + 1
#                      = 1281
#
# O componente DC (0 Hz) é mantido.
# -------------------------------------------------------------------------

FFT_COMPLETA = np.abs(
    np.fft.rfft(
        X,
        axis=1
    )
)


# Frequências correspondentes
FREQUENCIES = np.fft.rfftfreq(
    N_SAMPLES,
    d=1.0 / SAMPLING_FREQUENCY
)


print()
print(f"Sinais: {X.shape[0]}")
print(f"Amostras por sinal: {X.shape[1]}")
print(f"Características FFT: {FFT_COMPLETA.shape[1]}")

if len(FREQUENCIES) > 1:

    resolution = (
        FREQUENCIES[1]
        - FREQUENCIES[0]
    )

    print(
        f"Resolução espectral: {resolution:.2f} Hz"
    )


# =============================================================================
# 10. FUNÇÃO - FILTRO DE CORRELAÇÃO
# =============================================================================

def filtro_correlacao(
    X_train,
    threshold=0.95
):
    """
    Remove características altamente correlacionadas.

    IMPORTANTE:
    A matriz de correlação é calculada SOMENTE usando o conjunto
    de treinamento do fold.

    Isso evita vazamento de informação.

    threshold = 0.95 significa:

        |correlação| > 0.95

    -> uma das características será removida.
    """

    # Matriz de correlação
    corr = np.corrcoef(
        X_train,
        rowvar=False
    )

    # Triângulo superior
    upper = np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)

    # Valores absolutos
    corr_upper = np.abs(corr)

    # Identificar características a remover
    to_remove = set()

    for i in range(
        corr_upper.shape[0]
    ):

        for j in range(
            i + 1,
            corr_upper.shape[1]
        ):

            if (
                corr_upper[i, j]
                > threshold
            ):

                # Remove a característica j
                to_remove.add(j)

    indices_keep = np.array(
        [
            i
            for i in range(
                X_train.shape[1]
            )
            if i not in to_remove
        ],
        dtype=int
    )

    return indices_keep


# =============================================================================
# 11. FUNÇÃO - FISHER DISCRIMINANT RATIO
# =============================================================================

def fisher_discriminant_ratio(
    X_train,
    y_train
):
    """
    Calcula o Fisher Discriminant Ratio para cada característica.

    Para cada característica:

              variância entre classes
    FDR = -------------------------------
              variância dentro classes

    Quanto maior o valor:

        maior a capacidade discriminativa.

    Para classificação multiclasses é utilizado:

        variância entre classes /
        variância dentro das classes
    """

    classes = np.unique(y_train)

    n_features = X_train.shape[1]

    # Média global
    global_mean = np.mean(
        X_train,
        axis=0
    )

    # Inicialização
    between_variance = np.zeros(
        n_features
    )

    within_variance = np.zeros(
        n_features
    )

    total_samples = len(y_train)

    # -------------------------------------------------------------------------
    # Variância entre classes
    # -------------------------------------------------------------------------

    for classe in classes:

        X_class = X_train[
            y_train == classe
        ]

        n_class = X_class.shape[0]

        class_mean = np.mean(
            X_class,
            axis=0
        )

        between_variance += (
            n_class
            *
            (
                class_mean
                - global_mean
            ) ** 2
        )

    # -------------------------------------------------------------------------
    # Variância dentro das classes
    # -------------------------------------------------------------------------

    for classe in classes:

        X_class = X_train[
            y_train == classe
        ]

        class_mean = np.mean(
            X_class,
            axis=0
        )

        within_variance += np.sum(
            (
                X_class
                - class_mean
            ) ** 2,
            axis=0
        )

    # Normalização
    between_variance /= total_samples

    # Evitar divisão por zero
    within_variance += 1e-12

    fisher_scores = (
        between_variance
        /
        within_variance
    )

    return fisher_scores


# =============================================================================
# 12. FUNÇÃO - BOOTSTRAP
# =============================================================================

def bootstrap_metric(
    y_true,
    y_pred,
    metric_function,
    n_bootstrap=1000,
    random_state=42
):
    """
    Calcula intervalo de confiança de 95% utilizando bootstrap.

    São realizadas 1000 reamostragens por padrão.
    """

    rng = np.random.default_rng(
        random_state
    )

    n = len(y_true)

    values = []

    for _ in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_true_boot = y_true[
            indices
        ]

        y_pred_boot = y_pred[
            indices
        ]

        try:

            value = metric_function(
                y_true_boot,
                y_pred_boot
            )

            values.append(value)

        except Exception:
            continue

    values = np.asarray(
        values
    )

    lower = np.percentile(
        values,
        2.5
    )

    upper = np.percentile(
        values,
        97.5
    )

    return lower, upper


# =============================================================================
# 13. INÍCIO DO CODECARBON
# =============================================================================

print()
print("=" * 80)
print("INICIANDO CODECARBON")
print("=" * 80)


# -----------------------------------------------------------------------------
# IMPORTANTE:
#
# O diretório é criado ANTES da criação do EmissionsTracker.
# Isso evita o erro:
#
# OSError: Folder 'codecarbon_results' doesn't exist!
# -----------------------------------------------------------------------------

DIR_CODECARBON.mkdir(
    parents=True,
    exist_ok=True
)


tracker = EmissionsTracker(
    project_name="FFT_SVM_PQD",
    output_dir=str(
        DIR_CODECARBON
    ),
    measure_power_secs=1,
    save_to_file=True,
    log_level="error"
)


# Iniciar medição
tracker.start()


# =============================================================================
# 14. ESTRUTURA DO STRATIFIED K-FOLD
# =============================================================================

print()
print("=" * 80)
print("ETAPA 2 - STRATIFIED K-FOLD")
print("=" * 80)


skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# -----------------------------------------------------------------------------
# Listas para armazenar os resultados
# -----------------------------------------------------------------------------

fold_results = []

# Predições out-of-fold
y_true_oof = []
y_pred_oof = []

# Scores do SVM
y_score_oof = []

# Probabilidades
y_prob_oof = []

# Índices das amostras
indices_oof = []


# -----------------------------------------------------------------------------
# Informações sobre as características
# -----------------------------------------------------------------------------

fisher_features_all_folds = []

correlation_features_all_folds = []


# =============================================================================
# 15. LOOP DOS 5 FOLDS
# =============================================================================

tempo_inicio_modelo = time.perf_counter()


for fold, (
    train_index,
    val_index
) in enumerate(
    skf.split(
        FFT_COMPLETA,
        y
    ),
    start=1
):

    print()
    print("=" * 80)
    print(f"FOLD {fold}/{N_SPLITS}")
    print("=" * 80)


    # -------------------------------------------------------------------------
    # Separação treino / validação
    # -------------------------------------------------------------------------

    X_train_fft = FFT_COMPLETA[
        train_index
    ]

    X_val_fft = FFT_COMPLETA[
        val_index
    ]

    y_train = y[
        train_index
    ]

    y_val = y[
        val_index
    ]


    # =========================================================================
    # ETAPA 3 - CORRELAÇÃO
    # =========================================================================

    print()
    print("ETAPA 3 - FILTRO DE CORRELAÇÃO")

    print(
        f"Características antes: "
        f"{X_train_fft.shape[1]}"
    )


    indices_corr = filtro_correlacao(
        X_train_fft,
        threshold=CORRELATION_THRESHOLD
    )


    X_train_corr = X_train_fft[
        :,
        indices_corr
    ]

    X_val_corr = X_val_fft[
        :,
        indices_corr
    ]


    correlation_features_all_folds.append(
        indices_corr
    )


    print(
        f"Características depois: "
        f"{X_train_corr.shape[1]}"
    )

    print(
        f"Características removidas: "
        f"{X_train_fft.shape[1] - X_train_corr.shape[1]}"
    )


    # =========================================================================
    # ETAPA 4 - FISHER DISCRIMINANT RATIO
    # =========================================================================

    print()
    print(
        "ETAPA 4 - FISHER DISCRIMINANT RATIO"
    )


    fisher_scores = fisher_discriminant_ratio(
        X_train_corr,
        y_train
    )


    # Ordenar do maior para o menor
    fisher_order = np.argsort(
        fisher_scores
    )[::-1]


    # Selecionar as 14 melhores
    n_features_select = min(
        N_FISHER_FEATURES,
        len(fisher_order)
    )


    indices_fisher_local = (
        fisher_order[
            :n_features_select
        ]
    )


    # Índices relativos ao vetor após correlação
    indices_fisher_global = (
        indices_corr[
            indices_fisher_local
        ]
    )


    fisher_features_all_folds.append(
        indices_fisher_global
    )


    X_train_selected = X_train_corr[
        :,
        indices_fisher_local
    ]

    X_val_selected = X_val_corr[
        :,
        indices_fisher_local
    ]


    # -------------------------------------------------------------------------
    # Frequências correspondentes
    # -------------------------------------------------------------------------

    selected_frequencies = (
        FREQUENCIES[
            indices_fisher_global
        ]
    )


    print(
        f"Características selecionadas: "
        f"{X_train_selected.shape[1]}"
    )

    print()
    print(
        "Frequências selecionadas (Hz):"
    )

    print(
        np.round(
            selected_frequencies,
            2
        )
    )


    # -------------------------------------------------------------------------
    # Salvar informações do Fisher
    # -------------------------------------------------------------------------

    fisher_fold_df = pd.DataFrame({
        "indice_fft": indices_fisher_global,
        "frequencia_Hz": selected_frequencies,
        "Fisher_score": fisher_scores[
            indices_fisher_local
        ]
    })

    fisher_fold_df.to_csv(
        DIR_FISHER /
        f"fisher_fold_{fold}.csv",
        index=False
    )


    # =========================================================================
    # ETAPA 5 - NORMALIZAÇÃO
    # =========================================================================

    print()
    print("ETAPA 5 - NORMALIZAÇÃO")


    scaler = StandardScaler()


    # IMPORTANTE:
    # O scaler é ajustado SOMENTE no treino.

    X_train_scaled = scaler.fit_transform(
        X_train_selected
    )


    X_val_scaled = scaler.transform(
        X_val_selected
    )


    # =========================================================================
    # ETAPA 6 - TREINAMENTO DO SVM
    # =========================================================================

    print()
    print("ETAPA 6 - TREINAMENTO DO SVM")


    # probability=True permite calcular:
    #
    # predict_proba()
    #
    # que será utilizada para Log Loss e curvas Precision-Recall.
    #
    # decision_function_shape='ovr' permite obter scores
    # no formato One-vs-Rest para ROC multiclasses.

    svm = SVC(
        kernel="linear",
        C=SVM_C,
        probability=True,
        decision_function_shape="ovr",
        random_state=RANDOM_STATE
    )


    tempo_inicio_fold = time.perf_counter()


    svm.fit(
        X_train_scaled,
        y_train
    )


    # =========================================================================
    # ETAPA 7 - CLASSIFICAÇÃO
    # =========================================================================

    print()
    print(
        "Classificando conjunto de validação..."
    )


    y_pred = svm.predict(
        X_val_scaled
    )


    # Scores de decisão
    y_score = svm.decision_function(
        X_val_scaled
    )


    # Probabilidades
    y_prob = svm.predict_proba(
        X_val_scaled
    )


    tempo_fold = (
        time.perf_counter()
        - tempo_inicio_fold
    )


    # =========================================================================
    # ETAPA 8 - MÉTRICAS
    # =========================================================================

    print()
    print("ETAPA 8 - MÉTRICAS")


    accuracy = accuracy_score(
        y_val,
        y_pred
    )


    precision_macro = precision_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )


    recall_macro = recall_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )


    f1_macro = f1_score(
        y_val,
        y_pred,
        average="macro",
        zero_division=0
    )


    balanced_acc = balanced_accuracy_score(
        y_val,
        y_pred
    )


    mcc = matthews_corrcoef(
        y_val,
        y_pred
    )


    kappa = cohen_kappa_score(
        y_val,
        y_pred
    )


    # -------------------------------------------------------------------------
    # ROC-AUC
    # -------------------------------------------------------------------------

    # Para multiclasses utilizamos One-vs-Rest.
    #
    # labels=CLASS_NAMES garante a ordem das classes.

    try:

        roc_auc_macro = roc_auc_score(
            y_val,
            y_prob,
            labels=CLASS_NAMES,
            multi_class="ovr",
            average="macro"
        )

    except Exception:

        roc_auc_macro = np.nan


    # -------------------------------------------------------------------------
    # Log Loss
    # -------------------------------------------------------------------------

    try:

        logloss = log_loss(
            y_val,
            y_prob,
            labels=CLASS_NAMES
        )

    except Exception:

        logloss = np.nan


    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Precision Macro   : {precision_macro:.4f}"
    )

    print(
        f"Recall Macro      : {recall_macro:.4f}"
    )

    print(
        f"F1 Macro          : {f1_macro:.4f}"
    )

    print(
        f"Balanced Accuracy : {balanced_acc:.4f}"
    )

    print(
        f"MCC               : {mcc:.4f}"
    )

    print(
        f"Cohen's Kappa     : {kappa:.4f}"
    )

    print(
        f"ROC-AUC Macro     : {roc_auc_macro:.4f}"
    )

    print(
        f"Log Loss          : {logloss:.4f}"
    )

    print(
        f"Tempo do fold     : {tempo_fold:.2f} s"
    )


    # -------------------------------------------------------------------------
    # Salvar resultado do fold
    # -------------------------------------------------------------------------

    fold_results.append({

        "Fold": fold,

        "Accuracy": accuracy,

        "Precision_macro":
            precision_macro,

        "Recall_macro":
            recall_macro,

        "F1_macro":
            f1_macro,

        "Balanced_accuracy":
            balanced_acc,

        "MCC":
            mcc,

        "Cohen_Kappa":
            kappa,

        "ROC_AUC_macro":
            roc_auc_macro,

        "Log_Loss":
            logloss,

        "Features_FFT":
            FFT_COMPLETA.shape[1],

        "Features_apos_correlacao":
            X_train_corr.shape[1],

        "Features_Fisher":
            X_train_selected.shape[1],

        "Tempo_s":
            tempo_fold
    })


    # =========================================================================
    # ARMAZENAR PREDIÇÕES OUT-OF-FOLD
    # =========================================================================

    y_true_oof.extend(
        y_val
    )

    y_pred_oof.extend(
        y_pred
    )

    y_score_oof.append(
        y_score
    )

    y_prob_oof.append(
        y_prob
    )

    indices_oof.extend(
        val_index
    )


# =============================================================================
# 16. CONVERTER RESULTADOS PARA NUMPY
# =============================================================================

y_true_oof = np.asarray(
    y_true_oof
)

y_pred_oof = np.asarray(
    y_pred_oof
)

y_score_oof = np.vstack(
    y_score_oof
)

y_prob_oof = np.vstack(
    y_prob_oof
)

indices_oof = np.asarray(
    indices_oof
)


# =============================================================================
# 17. ORGANIZAR PREDIÇÕES OUT-OF-FOLD
# =============================================================================

# O StratifiedKFold devolve os folds em ordens diferentes.
#
# Vamos reorganizar pelas posições originais das amostras.

ordem = np.argsort(
    indices_oof
)

y_true_oof = y_true_oof[
    ordem
]

y_pred_oof = y_pred_oof[
    ordem
]

y_score_oof = y_score_oof[
    ordem
]

y_prob_oof = y_prob_oof[
    ordem
]


# =============================================================================
# 18. MÉTRICAS GLOBAIS
# =============================================================================

print()
print("=" * 80)
print("RESULTADOS GLOBAIS - OUT-OF-FOLD")
print("=" * 80)


accuracy_global = accuracy_score(
    y_true_oof,
    y_pred_oof
)


precision_global = precision_score(
    y_true_oof,
    y_pred_oof,
    average="macro",
    zero_division=0
)


recall_global = recall_score(
    y_true_oof,
    y_pred_oof,
    average="macro",
    zero_division=0
)


f1_global = f1_score(
    y_true_oof,
    y_pred_oof,
    average="macro",
    zero_division=0
)


balanced_global = balanced_accuracy_score(
    y_true_oof,
    y_pred_oof
)


mcc_global = matthews_corrcoef(
    y_true_oof,
    y_pred_oof
)


kappa_global = cohen_kappa_score(
    y_true_oof,
    y_pred_oof
)


roc_auc_global = roc_auc_score(
    y_true_oof,
    y_prob_oof,
    labels=CLASS_NAMES,
    multi_class="ovr",
    average="macro"
)


logloss_global = log_loss(
    y_true_oof,
    y_prob_oof,
    labels=CLASS_NAMES
)


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
    f"Balanced Accuracy : "
    f"{balanced_global:.4f}"
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


# =============================================================================
# 19. MÉTRICAS POR CLASSE
# =============================================================================

print()
print("=" * 80)
print("MÉTRICAS POR CLASSE")
print("=" * 80)


report_dict = classification_report(
    y_true_oof,
    y_pred_oof,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report_dict
).T


print()
print(
    report_df
)


# =============================================================================
# 20. ROC-AUC POR CLASSE
# =============================================================================

# Binarização das classes
y_true_binary = label_binarize(
    y_true_oof,
    classes=CLASS_NAMES
)


roc_auc_per_class = {}


for i, classe in enumerate(
    CLASS_NAMES
):

    try:

        roc_auc_per_class[classe] = (
            roc_auc_score(
                y_true_binary[:, i],
                y_prob_oof[:, i]
            )
        )

    except Exception:

        roc_auc_per_class[classe] = np.nan


print()
print("ROC-AUC POR CLASSE:")

for classe in CLASS_NAMES:

    print(
        f"{classe:<10}: "
        f"{roc_auc_per_class[classe]:.4f}"
    )


# =============================================================================
# 21. PRECISION-RECALL POR CLASSE
# =============================================================================

average_precision_per_class = {}


for i, classe in enumerate(
    CLASS_NAMES
):

    try:

        average_precision_per_class[classe] = (
            average_precision_score(
                y_true_binary[:, i],
                y_prob_oof[:, i]
            )
        )

    except Exception:

        average_precision_per_class[classe] = np.nan


# =============================================================================
# 22. MATRIZ DE CONFUSÃO
# =============================================================================

print()
print("=" * 80)
print("MATRIZ DE CONFUSÃO")
print("=" * 80)


cm = confusion_matrix(
    y_true_oof,
    y_pred_oof,
    labels=CLASS_NAMES
)


print()
print(cm)


# =============================================================================
# 23. FUNÇÃO PARA CRIAR A MATRIZ DE CONFUSÃO
#     AZUL = ACERTOS
#     LARANJA = ERROS
#
#     Ambos em escala de intensidade.
# =============================================================================

def plot_confusion_matrix_custom(
    cm,
    classes,
    output_path
):

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    # -------------------------------------------------------------------------
    # Criar matriz RGBA manualmente
    #
    # Diagonal:
    #   azul claro -> azul mais forte
    #
    # Fora da diagonal:
    #   laranja claro -> laranja mais forte
    # -------------------------------------------------------------------------

    n = cm.shape[0]

    rgba = np.ones(
        (n, n, 4)
    )


    # Valor máximo da diagonal
    diagonal_values = np.diag(
        cm
    )

    max_diagonal = max(
        diagonal_values.max(),
        1
    )


    # Maior erro
    off_diag = cm.copy()

    np.fill_diagonal(
        off_diag,
        0
    )

    max_error = max(
        off_diag.max(),
        1
    )


    # -------------------------------------------------------------------------
    # Azul para acertos
    # -------------------------------------------------------------------------

    for i in range(n):

        for j in range(n):

            value = cm[i, j]

            if i == j:

                intensidade = (
                    value
                    /
                    max_diagonal
                )

                # Azul claro -> azul forte
                rgba[i, j, 0] = (
                    1.0
                    -
                    0.70 * intensidade
                )

                rgba[i, j, 1] = (
                    1.0
                    -
                    0.30 * intensidade
                )

                rgba[i, j, 2] = 1.0

            else:

                intensidade = (
                    value
                    /
                    max_error
                )

                # Laranja claro -> laranja forte
                rgba[i, j, 0] = 1.0

                rgba[i, j, 1] = (
                    1.0
                    -
                    0.65 * intensidade
                )

                rgba[i, j, 2] = (
                    1.0
                    -
                    0.85 * intensidade
                )


    # -------------------------------------------------------------------------
    # Mostrar imagem
    # -------------------------------------------------------------------------

    ax.imshow(
        rgba,
        interpolation="nearest"
    )


    # -------------------------------------------------------------------------
    # Título
    # -------------------------------------------------------------------------

    ax.set_title(
        "Matriz de Confusão - FFT & SVM",
        fontsize=16,
        fontweight="bold",
        pad=15
    )


    # -------------------------------------------------------------------------
    # Eixos
    # -------------------------------------------------------------------------

    ax.set_xlabel(
        "Classe Predita",
        fontsize=12
    )

    ax.set_ylabel(
        "Classe Real",
        fontsize=12
    )


    ax.set_xticks(
        np.arange(len(classes))
    )

    ax.set_yticks(
        np.arange(len(classes))
    )

    ax.set_xticklabels(
        classes,
        fontsize=10
    )

    ax.set_yticklabels(
        classes,
        fontsize=10
    )


    # -------------------------------------------------------------------------
    # Valores dentro das células
    # -------------------------------------------------------------------------

    for i in range(n):

        for j in range(n):

            value = cm[i, j]

            # Fundo escuro -> texto branco
            # Fundo claro -> texto preto

            if i == j:

                intensidade = (
                    value
                    /
                    max_diagonal
                )

                text_color = (
                    "white"
                    if intensidade > 0.55
                    else "black"
                )

            else:

                intensidade = (
                    value
                    /
                    max_error
                )

                text_color = (
                    "white"
                    if intensidade > 0.55
                    else "black"
                )

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=text_color
            )


    # -------------------------------------------------------------------------
    # Linhas de separação
    # -------------------------------------------------------------------------

    ax.set_xticks(
        np.arange(
            -0.5,
            n,
            1
        ),
        minor=True
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            n,
            1
        ),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.5
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 24. SALVAR MATRIZ DE CONFUSÃO
# =============================================================================

confusion_path = (
    DIR_MATRIZ
    /
    "matriz_confusao_FFT_SVM.png"
)


plot_confusion_matrix_custom(
    cm,
    CLASS_NAMES,
    confusion_path
)


print()
print(
    "Matriz de confusão salva em:"
)

print(
    confusion_path
)


# =============================================================================
# 25. CURVA ROC MULTICLASSE
# =============================================================================

print()
print("=" * 80)
print("CURVA ROC")
print("=" * 80)


plt.figure(
    figsize=(10, 8)
)


colors = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b"
]


fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}


# -----------------------------------------------------------------------------
# ROC de cada classe
# -----------------------------------------------------------------------------

for i, classe in enumerate(
    CLASS_NAMES
):

    fpr, tpr, _ = roc_curve(
        y_true_binary[:, i],
        y_prob_oof[:, i]
    )

    roc_auc_value = auc(
        fpr,
        tpr
    )


    fpr_dict[classe] = fpr
    tpr_dict[classe] = tpr
    roc_auc_dict[classe] = (
        roc_auc_value
    )


    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"{classe} "
            f"(AUC = {roc_auc_value:.3f})"
        )
    )


# -----------------------------------------------------------------------------
# ROC micro-average
# -----------------------------------------------------------------------------

fpr_micro, tpr_micro, _ = roc_curve(
    y_true_binary.ravel(),
    y_prob_oof.ravel()
)


roc_auc_micro = auc(
    fpr_micro,
    tpr_micro
)


plt.plot(
    fpr_micro,
    tpr_micro,
    linestyle="--",
    linewidth=2.5,
    label=(
        f"Micro-average "
        f"(AUC = {roc_auc_micro:.3f})"
    )
)


# -----------------------------------------------------------------------------
# ROC macro-average
# -----------------------------------------------------------------------------

all_fpr = np.unique(
    np.concatenate(
        [
            fpr_dict[classe]
            for classe in CLASS_NAMES
        ]
    )
)


mean_tpr = np.zeros_like(
    all_fpr
)


for classe in CLASS_NAMES:

    mean_tpr += np.interp(
        all_fpr,
        fpr_dict[classe],
        tpr_dict[classe]
    )


mean_tpr /= len(
    CLASS_NAMES
)


roc_auc_macro_curve = auc(
    all_fpr,
    mean_tpr
)


plt.plot(
    all_fpr,
    mean_tpr,
    linewidth=3,
    linestyle=":",
    label=(
        f"Macro-average "
        f"(AUC = {roc_auc_macro_curve:.3f})"
    )
)


# -----------------------------------------------------------------------------
# Linha aleatória
# -----------------------------------------------------------------------------

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Classificador aleatório"
)


plt.xlabel(
    "Taxa de Falsos Positivos",
    fontsize=12
)

plt.ylabel(
    "Taxa de Verdadeiros Positivos",
    fontsize=12
)

plt.title(
    "Curva ROC Multiclasse - FFT & SVM",
    fontsize=16,
    fontweight="bold"
)

plt.legend(
    loc="lower right",
    fontsize=9
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


roc_path = (
    DIR_ROC
    /
    "curva_ROC_FFT_SVM.png"
)


plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Curva ROC salva em:"
)

print(
    roc_path
)


# =============================================================================
# 26. CURVA PRECISION-RECALL
# =============================================================================

print()
print("=" * 80)
print("CURVA PRECISION-RECALL")
print("=" * 80)


plt.figure(
    figsize=(10, 8)
)


for i, classe in enumerate(
    CLASS_NAMES
):

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true_binary[:, i],
            y_prob_oof[:, i]
        )
    )


    ap = average_precision_score(
        y_true_binary[:, i],
        y_prob_oof[:, i]
    )


    plt.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=(
            f"{classe} "
            f"(AP = {ap:.3f})"
        )
    )


plt.xlabel(
    "Recall",
    fontsize=12
)

plt.ylabel(
    "Precision",
    fontsize=12
)

plt.title(
    "Curva Precision-Recall Multiclasse - FFT & SVM",
    fontsize=16,
    fontweight="bold"
)

plt.legend(
    loc="lower left",
    fontsize=9
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()


pr_path = (
    DIR_PR
    /
    "curva_Precision_Recall_FFT_SVM.png"
)


plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Curva Precision-Recall salva em:"
)

print(
    pr_path
)


# =============================================================================
# 27. MÉTRICAS POR CLASSE EM CSV
# =============================================================================

metrics_class_df = pd.DataFrame({

    "Classe":
        CLASS_NAMES,

    "Precision":
        [
            report_dict[c]["precision"]
            for c in CLASS_NAMES
        ],

    "Recall":
        [
            report_dict[c]["recall"]
            for c in CLASS_NAMES
        ],

    "F1_score":
        [
            report_dict[c]["f1-score"]
            for c in CLASS_NAMES
        ],

    "Support":
        [
            report_dict[c]["support"]
            for c in CLASS_NAMES
        ],

    "ROC_AUC":
        [
            roc_auc_per_class[c]
            for c in CLASS_NAMES
        ],

    "Average_Precision":
        [
            average_precision_per_class[c]
            for c in CLASS_NAMES
        ]
})


metrics_class_path = (
    DIR_METRICAS
    /
    "metricas_por_classe.csv"
)


metrics_class_df.to_csv(
    metrics_class_path,
    index=False
)


# =============================================================================
# 28. MÉTRICAS DOS FOLDS
# =============================================================================

fold_df = pd.DataFrame(
    fold_results
)


fold_metrics_path = (
    DIR_METRICAS
    /
    "metricas_por_fold.csv"
)


fold_df.to_csv(
    fold_metrics_path,
    index=False
)


print()
print("=" * 80)
print("MÉTRICAS DOS FOLDS")
print("=" * 80)

print()
print(
    fold_df
)


# =============================================================================
# 29. MÉDIA E DESVIO-PADRÃO DOS FOLDS
# =============================================================================

metric_columns = [
    "Accuracy",
    "Precision_macro",
    "Recall_macro",
    "F1_macro",
    "Balanced_accuracy",
    "MCC",
    "Cohen_Kappa",
    "ROC_AUC_macro",
    "Log_Loss"
]


mean_std_rows = []


for metric in metric_columns:

    mean_value = fold_df[
        metric
    ].mean()

    std_value = fold_df[
        metric
    ].std(ddof=1)


    mean_std_rows.append({

        "Metrica": metric,

        "Media": mean_value,

        "Desvio_padrao": std_value

    })


mean_std_df = pd.DataFrame(
    mean_std_rows
)


mean_std_path = (
    DIR_METRICAS
    /
    "media_desvio_padrao_folds.csv"
)


mean_std_df.to_csv(
    mean_std_path,
    index=False
)


# =============================================================================
# 30. INTERVALOS DE CONFIANÇA POR BOOTSTRAP
# =============================================================================

print()
print("=" * 80)
print("INTERVALOS DE CONFIANÇA - BOOTSTRAP")
print("=" * 80)


bootstrap_metrics = {

    "Accuracy":
        accuracy_score,

    "Precision_macro":
        lambda yt, yp:
            precision_score(
                yt,
                yp,
                average="macro",
                zero_division=0
            ),

    "Recall_macro":
        lambda yt, yp:
            recall_score(
                yt,
                yp,
                average="macro",
                zero_division=0
            ),

    "F1_macro":
        lambda yt, yp:
            f1_score(
                yt,
                yp,
                average="macro",
                zero_division=0
            ),

    "Balanced_accuracy":
        balanced_accuracy_score,

    "MCC":
        matthews_corrcoef,

    "Cohen_Kappa":
        cohen_kappa_score
}


bootstrap_results = []


for metric_name, metric_function in (
    bootstrap_metrics.items()
):

    lower, upper = bootstrap_metric(
        y_true_oof,
        y_pred_oof,
        metric_function,
        n_bootstrap=1000,
        random_state=RANDOM_STATE
    )


    bootstrap_results.append({

        "Metrica":
            metric_name,

        "Valor":
            metric_function(
                y_true_oof,
                y_pred_oof
            ),

        "IC_95_inferior":
            lower,

        "IC_95_superior":
            upper

    })


    print(
        f"{metric_name:<22}: "
        f"{lower:.4f} - {upper:.4f}"
    )


bootstrap_df = pd.DataFrame(
    bootstrap_results
)


bootstrap_path = (
    DIR_METRICAS
    /
    "intervalos_confianca_bootstrap.csv"
)


bootstrap_df.to_csv(
    bootstrap_path,
    index=False
)


# =============================================================================
# 31. RELATÓRIO FINAL
# =============================================================================

print()
print("=" * 80)
print("GERANDO RELATÓRIO")
print("=" * 80)


tempo_total = (
    time.perf_counter()
    - tempo_inicio_modelo
)


# -----------------------------------------------------------------------------
# Recuperar informação do CodeCarbon
# -----------------------------------------------------------------------------

try:

    emissions = tracker.stop()

except Exception as erro:

    print()
    print(
        "AVISO: erro ao finalizar CodeCarbon:"
    )

    print(erro)

    emissions = np.nan


# =============================================================================
# 32. RESULTADO DO CODECARBON
# =============================================================================

print()
print("=" * 80)
print("CODECARBON FINALIZADO")
print("=" * 80)


print()

if not np.isnan(emissions):

    print(
        f"Emissão estimada: "
        f"{emissions:.8f} kg CO2"
    )

    print(
        f"Emissão estimada: "
        f"{emissions * 1000:.4f} g CO2"
    )

else:

    print(
        "Emissão de CO2 não disponível."
    )


# =============================================================================
# 33. SALVAR RESUMO DO CODECARBON
# =============================================================================

codecarbon_summary = pd.DataFrame({

    "Projeto": [
        "FFT + SVM"
    ],

    "Tempo_s": [
        tempo_total
    ],

    "Tempo_min": [
        tempo_total / 60
    ],

    "CO2_kg": [
        emissions
    ],

    "CO2_g": [
        emissions * 1000
        if not np.isnan(emissions)
        else np.nan
    ]

})


codecarbon_summary_path = (
    DIR_CODECARBON
    /
    "resumo_codecarbon.csv"
)


codecarbon_summary.to_csv(
    codecarbon_summary_path,
    index=False
)


# =============================================================================
# 34. RESUMO FINAL
# =============================================================================

summary_df = pd.DataFrame({

    "Metrica": [

        "Accuracy",

        "Precision Macro",

        "Recall Macro",

        "F1 Macro",

        "Balanced Accuracy",

        "MCC",

        "Cohen Kappa",

        "ROC-AUC Macro",

        "Log Loss",

        "Tempo (s)",

        "Tempo (min)",

        "CO2 (kg)",

        "CO2 (g)"

    ],

    "Valor": [

        accuracy_global,

        precision_global,

        recall_global,

        f1_global,

        balanced_global,

        mcc_global,

        kappa_global,

        roc_auc_global,

        logloss_global,

        tempo_total,

        tempo_total / 60,

        emissions,

        emissions * 1000
        if not np.isnan(emissions)
        else np.nan

    ]

})


summary_path = (
    DIR_METRICAS
    /
    "metricas_finais.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# =============================================================================
# 35. RELATÓRIO TXT
# =============================================================================

report_path = (
    DIR_RELATORIO
    /
    "relatorio_FFT_SVM.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 80
        + "\n"
    )

    f.write(
        "RELATÓRIO FINAL - FFT + SVM\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )


    f.write(
        "CONFIGURAÇÕES DO EXPERIMENTO\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Quantidade de sinais: "
        f"{X.shape[0]}\n"
    )

    f.write(
        f"Amostras por sinal: "
        f"{X.shape[1]}\n"
    )

    f.write(
        f"Classes: "
        f"{len(CLASS_NAMES)}\n"
    )

    f.write(
        f"Folds: "
        f"{N_SPLITS}\n"
    )

    f.write(
        f"Threshold correlação: "
        f"{CORRELATION_THRESHOLD}\n"
    )

    f.write(
        f"Características Fisher: "
        f"{N_FISHER_FEATURES}\n"
    )

    f.write(
        f"SVM C: "
        f"{SVM_C}\n\n"
    )


    f.write(
        "RESULTADOS GLOBAIS\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Accuracy          : "
        f"{accuracy_global:.6f}\n"
    )

    f.write(
        f"Precision Macro   : "
        f"{precision_global:.6f}\n"
    )

    f.write(
        f"Recall Macro      : "
        f"{recall_global:.6f}\n"
    )

    f.write(
        f"F1 Macro          : "
        f"{f1_global:.6f}\n"
    )

    f.write(
        f"Balanced Accuracy : "
        f"{balanced_global:.6f}\n"
    )

    f.write(
        f"MCC               : "
        f"{mcc_global:.6f}\n"
    )

    f.write(
        f"Cohen's Kappa     : "
        f"{kappa_global:.6f}\n"
    )

    f.write(
        f"ROC-AUC Macro     : "
        f"{roc_auc_global:.6f}\n"
    )

    f.write(
        f"Log Loss          : "
        f"{logloss_global:.6f}\n\n"
    )


    f.write(
        "ROC-AUC POR CLASSE\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )


    for classe in CLASS_NAMES:

        f.write(
            f"{classe:<10}: "
            f"{roc_auc_per_class[classe]:.6f}\n"
        )


    f.write(
        "\n"
    )


    f.write(
        "MÉTRICAS POR CLASSE\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )


    f.write(
        report_df.to_string()
    )

    f.write(
        "\n\n"
    )


    f.write(
        "TEMPO E EMISSÕES\n"
    )

    f.write(
        "-" * 80
        + "\n"
    )

    f.write(
        f"Tempo total: "
        f"{tempo_total:.4f} s\n"
    )

    f.write(
        f"Tempo total: "
        f"{tempo_total / 60:.4f} min\n"
    )

    f.write(
        f"CO2: "
        f"{emissions:.10f} kg\n"
    )

    f.write(
        f"CO2: "
        f"{emissions * 1000:.6f} g\n"
    )


# =============================================================================
# 36. SALVAR MATRIZ DE CONFUSÃO EM CSV
# =============================================================================

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)


cm_csv_path = (
    DIR_MATRIZ
    /
    "matriz_confusao_FFT_SVM.csv"
)


cm_df.to_csv(
    cm_csv_path
)


# =============================================================================
# 37. SALVAR PREDIÇÕES OUT-OF-FOLD
# =============================================================================

oof_df = pd.DataFrame({

    "y_true":
        y_true_oof,

    "y_pred":
        y_pred_oof

})


for i, classe in enumerate(
    CLASS_NAMES
):

    oof_df[
        f"prob_{classe}"
    ] = y_prob_oof[:, i]


oof_path = (
    DIR_METRICAS
    /
    "predicoes_out_of_fold.csv"
)


oof_df.to_csv(
    oof_path,
    index=False
)


# =============================================================================
# 38. RESUMO FINAL NO TERMINAL
# =============================================================================

print()
print()
print("=" * 80)
print("RESULTADO FINAL - FFT + SVM")
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
    f"Balanced Accuracy : "
    f"{balanced_global:.4f}"
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
    f"Tempo total       : "
    f"{tempo_total:.2f} segundos"
)

print(
    f"Tempo total       : "
    f"{tempo_total / 60:.2f} minutos"
)

if not np.isnan(emissions):

    print(
        f"CO2               : "
        f"{emissions:.8f} kg"
    )

    print(
        f"CO2               : "
        f"{emissions * 1000:.4f} g"
    )


# =============================================================================
# 39. CAMINHOS DOS ARQUIVOS
# =============================================================================

print()
print("=" * 80)
print("ARQUIVOS GERADOS")
print("=" * 80)

print()

print(
    "Pasta principal:"
)

print(
    RESULTS_DIR
)


print()

print(
    "Matriz de confusão:"
)

print(
    confusion_path
)


print()

print(
    "Curva ROC:"
)

print(
    roc_path
)


print()

print(
    "Curva Precision-Recall:"
)

print(
    pr_path
)


print()

print(
    "Métricas por fold:"
)

print(
    fold_metrics_path
)


print()

print(
    "Métricas finais:"
)

print(
    summary_path
)


print()

print(
    "Métricas por classe:"
)

print(
    metrics_class_path
)


print()

print(
    "Intervalos de confiança:"
)

print(
    bootstrap_path
)


print()

print(
    "Relatório:"
)

print(
    report_path
)


print()

print(
    "CodeCarbon:"
)

print(
    DIR_CODECARBON
)


print()
print("=" * 80)
print("PROGRAMA FINALIZADO")
print("=" * 80)
print()