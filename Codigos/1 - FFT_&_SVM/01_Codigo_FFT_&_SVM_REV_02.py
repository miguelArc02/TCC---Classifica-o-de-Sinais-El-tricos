# =============================================================================
# CLASSIFICADOR DE DISTÚRBIOS DE QUALIDADE DE ENERGIA
# PIPELINE:

#       SINAL
         
#        FFT         
#    CORRELAÇÃO         
#       FISHER
        
#    NORMALIZAÇÃO
       
#        SVM
       
#   STRATIFIED K-FOLD
       
#      MÉTRICAS

# Métricas principais:
#   - Accuracy
#   - Precision Macro
#   - F1 Macro
#   - MCC
#   - Cohen's Kappa
#   - ROC-AUC Macro
#   - Log Loss
#   - Tempo
#   - CO2

# Resultados gráficos:
#   - Matriz de confusão
#   - Curva ROC
#   - Curva Precision-Recall



# =============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    log_loss,
    precision_recall_curve,
    average_precision_score
)

from sklearn.metrics import ConfusionMatrixDisplay

from codecarbon import EmissionsTracker

from pathlib import Path

# Ignora alguns avisos que não interferem no processamento
warnings.filterwarnings("ignore")


# =============================================================================
# 2. CONFIGURAÇÕES PRINCIPAIS
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTANTE:
# ALTERE SOMENTE ESTAS DUAS PASTAS.
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = BASE_DIR / "Banco_de_Dados" / "CSV" / "Signal_Feature"
OUTPUT_DIR = Path(__file__).resolve().parent / "Resultados" / "Resultados_REV_02"


# -----------------------------------------------------------------------------
# Parâmetros do experimento
# -----------------------------------------------------------------------------

FS = 15360                  # Frequência de amostragem [Hz]
FUNDAMENTAL = 60            # Frequência fundamental [Hz]

N_SPLITS = 5                # Número de folds
RANDOM_STATE = 42

CORRELATION_THRESHOLD = 0.95

N_FISHER_FEATURES = 14

# Configuração do SVM
SVM_KERNEL = "rbf"
SVM_C = 10
SVM_GAMMA = "scale"

# Número de reamostragens para IC
N_BOOTSTRAP = 1000


# Classes
CLASSES = [
    "capc",
    "har",
    "normal",
    "notch",
    "sag",
    "spike",
    "swell"
]


# =============================================================================
# 3. CRIAÇÃO DAS PASTAS
# =============================================================================

DIR_CONFUSION = os.path.join(
    OUTPUT_DIR,
    "Matriz_Confusao"
)

DIR_ROC = os.path.join(
    OUTPUT_DIR,
    "ROC"
)

DIR_PR = os.path.join(
    OUTPUT_DIR,
    "Precision_Recall"
)

DIR_METRICS = os.path.join(
    OUTPUT_DIR,
    "Metricas"
)

DIR_REPORT = os.path.join(
    OUTPUT_DIR,
    "Relatorio"
)

DIR_CODECARBON = os.path.join(
    OUTPUT_DIR,
    "CodeCarbon"
)


# Cria todas as pastas automaticamente
for directory in [
    OUTPUT_DIR,
    DIR_CONFUSION,
    DIR_ROC,
    DIR_PR,
    DIR_METRICS,
    DIR_REPORT,
    DIR_CODECARBON
]:

    os.makedirs(directory, exist_ok=True)


# =============================================================================
# 4. FUNÇÃO PARA LOCALIZAR OS ARQUIVOS
# =============================================================================

def localizar_arquivo(nome_classe):
    """
    Procura o arquivo CSV correspondente à classe.

    Exemplo:
        signal_capc.csv
        signal_har.csv
        signal_normal.csv
    """

    candidatos = [
        f"signal_{nome_classe}.csv",
        f"{nome_classe}.csv"
    ]

    for nome in candidatos:

        caminho = os.path.join(DATASET_DIR, nome)

        if os.path.exists(caminho):
            return caminho

    raise FileNotFoundError(
        f"\nArquivo da classe '{nome_classe}' não encontrado.\n"
        f"Procurei em:\n{DATASET_DIR}\n"
        f"Arquivos esperados: {candidatos}"
    )


# =============================================================================
# 5. CARREGAMENTO DO DATASET
# =============================================================================

def carregar_dataset():

    print("\n" + "=" * 78)
    print("CARREGANDO DATASET")
    print("=" * 78)

    X_lista = []
    y_lista = []

    for classe in CLASSES:

        caminho = localizar_arquivo(classe)

        # Os arquivos possuem 100 sinais × 2560 amostras
        dados = pd.read_csv(
            caminho,
            header=None
        )

        dados = dados.apply(
            pd.to_numeric,
            errors="coerce"
        )

        dados = dados.dropna(
            axis=1,
            how="all"
        )

        matriz = dados.values.astype(float)

        print(
            f"{classe:<10} -> "
            f"arquivo = {os.path.basename(caminho):<30} "
            f"shape = {matriz.shape}"
        )

        # Cada linha representa um sinal
        for sinal in matriz:

            X_lista.append(sinal)
            y_lista.append(classe)

    X = np.asarray(X_lista, dtype=float)
    y = np.asarray(y_lista)

    print("\nNúmero total de sinais:", len(X))
    print("Número de amostras por sinal:", X.shape[1])
    print("Número de classes:", len(np.unique(y)))

    print("\nDistribuição das classes:")
    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    return X, y


# =============================================================================
# 6. TRANSFORMADA RÁPIDA DE FOURIER - FFT
# =============================================================================

def extrair_fft(X, fs):

    """
    Calcula a FFT para cada sinal.

    Como utilizamos rFFT, somente o espectro de frequências positivas
    é utilizado.

    Para N = 2560:

        N/2 + 1 = 1281 características

    Portanto:

        0 Hz
        6 Hz
        12 Hz
        ...
        7680 Hz

    """

    N = X.shape[1]

    # FFT somente para frequências positivas
    fft_complex = np.fft.rfft(
        X,
        axis=1
    )

    # Magnitude da FFT
    fft_mag = np.abs(fft_complex)

    # Normalização pela quantidade de amostras
    fft_mag = fft_mag / N

    # Frequências correspondentes
    frequencies = np.fft.rfftfreq(
        N,
        d=1 / fs
    )

    return fft_mag, frequencies


# =============================================================================
# 7. FILTRO DE CORRELAÇÃO
# =============================================================================

def filtro_correlacao(X_train, threshold=0.95):

    """
    Remove características altamente correlacionadas.

    Se duas características apresentarem:

        |correlação| > 0.95

    uma delas é removida.

    IMPORTANTE:
    O filtro é calculado somente no conjunto de treinamento do fold.
    Isso evita data leakage.
    """

    correlation_matrix = np.corrcoef(
        X_train,
        rowvar=False
    )

    correlation_matrix = np.nan_to_num(
        correlation_matrix
    )

    upper = np.triu(
        np.abs(correlation_matrix),
        k=1
    )

    to_remove = set()

    n_features = X_train.shape[1]

    for i in range(n_features):

        for j in range(i + 1, n_features):

            if upper[i, j] > threshold:
                to_remove.add(j)

    indices_keep = np.array(
        [
            i for i in range(n_features)
            if i not in to_remove
        ],
        dtype=int
    )

    return indices_keep


# =============================================================================
# 8. FISHER DISCRIMINANT RATIO
# =============================================================================

def fisher_score_multiclass(X, y):

    """
    Calcula um Fisher Discriminant Ratio multiclasses.

    A ideia é comparar:

        variabilidade ENTRE classes

    com:

        variabilidade DENTRO das classes.

    Quanto maior o valor:

        maior o poder discriminativo da característica.
    """

    classes = np.unique(y)

    overall_mean = np.mean(
        X,
        axis=0
    )

    between = np.zeros(
        X.shape[1]
    )

    within = np.zeros(
        X.shape[1]
    )

    for classe in classes:

        X_c = X[y == classe]

        n_c = len(X_c)

        if n_c == 0:
            continue

        mean_c = np.mean(
            X_c,
            axis=0
        )

        var_c = np.var(
            X_c,
            axis=0
        )

        # Variância entre classes
        between += n_c * (
            mean_c - overall_mean
        ) ** 2

        # Variância dentro da classe
        within += n_c * var_c

    # Evita divisão por zero
    within = np.where(
        within == 0,
        np.finfo(float).eps,
        within
    )

    fisher = between / within

    fisher = np.nan_to_num(
        fisher,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return fisher


# =============================================================================
# 9. SELEÇÃO DAS CARACTERÍSTICAS PELO FISHER
# =============================================================================

def selecionar_fisher(
    X_train,
    y_train,
    n_features
):

    scores = fisher_score_multiclass(
        X_train,
        y_train
    )

    n_features = min(
        n_features,
        X_train.shape[1]
    )

    indices = np.argsort(
        scores
    )[::-1][:n_features]

    # Ordena os índices para manter a ordem espectral
    indices = np.sort(indices)

    return indices, scores


# =============================================================================
# 10. BOOTSTRAP PARA INTERVALOS DE CONFIANÇA
# =============================================================================

def bootstrap_ci(
    y_true,
    y_pred,
    n_bootstrap=1000,
    random_state=42
):

    """
    Calcula IC de 95% utilizando bootstrap.

    As métricas avaliadas são:

        Accuracy
        Precision Macro
        Recall Macro
        F1 Macro
        Balanced Accuracy
        MCC
        Cohen's Kappa
    """

    rng = np.random.default_rng(
        random_state
    )

    n = len(y_true)

    resultados = {
        "Accuracy": [],
        "Precision_macro": [],
        "Recall_macro": [],
        "F1_macro": [],
        "Balanced_accuracy": [],
        "MCC": [],
        "Cohen_Kappa": []
    }

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        yt = y_true[indices]
        yp = y_pred[indices]

        resultados["Accuracy"].append(
            accuracy_score(yt, yp)
        )

        resultados["Precision_macro"].append(
            precision_score(
                yt,
                yp,
                average="macro",
                zero_division=0
            )
        )

        resultados["Recall_macro"].append(
            recall_score(
                yt,
                yp,
                average="macro",
                zero_division=0
            )
        )

        resultados["F1_macro"].append(
            f1_score(
                yt,
                yp,
                average="macro",
                zero_division=0
            )
        )

        resultados["Balanced_accuracy"].append(
            balanced_accuracy_score(
                yt,
                yp
            )
        )

        resultados["MCC"].append(
            matthews_corrcoef(
                yt,
                yp
            )
        )

        resultados["Cohen_Kappa"].append(
            cohen_kappa_score(
                yt,
                yp
            )
        )

    linhas = []

    for nome, valores in resultados.items():

        valores = np.asarray(
            valores
        )

        linhas.append({
            "Metrica": nome,
            "Media": np.mean(valores),
            "IC_95_inf": np.percentile(
                valores,
                2.5
            ),
            "IC_95_sup": np.percentile(
                valores,
                97.5
            )
        })

    return pd.DataFrame(linhas)


# =============================================================================
# 11. MATRIZ DE CONFUSÃO
# =============================================================================

def salvar_matriz_confusao(
    cm,
    classes,
    caminho
):

    """
    Salva a matriz de confusão.

    Diagonal:
        azul em escala.

    Erros:
        laranja em escala.

    Quanto maior o número:
        mais forte a intensidade da cor.
    """

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    # Matriz base branca
    imagem = np.ones(
        (
            len(classes),
            len(classes),
            4
        )
    )

    max_acerto = np.max(
        np.diag(cm)
    )

    erros = cm.copy()

    np.fill_diagonal(
        erros,
        0
    )

    max_erro = np.max(
        erros
    )

    for i in range(len(classes)):

        for j in range(len(classes)):

            valor = cm[i, j]

            if i == j:

                # Azul claro → azul mais forte
                intensidade = (
                    valor / max_acerto
                    if max_acerto > 0
                    else 0
                )

                imagem[i, j] = [
                    1 - 0.75 * intensidade,
                    1 - 0.35 * intensidade,
                    1,
                    1
                ]

            elif valor > 0:

                # Laranja claro → laranja mais forte
                intensidade = (
                    valor / max_erro
                    if max_erro > 0
                    else 0
                )

                imagem[i, j] = [
                    1,
                    1 - 0.65 * intensidade,
                    1 - 0.85 * intensidade,
                    1
                ]

            else:

                imagem[i, j] = [
                    1,
                    1,
                    1,
                    1
                ]

    ax.imshow(
        imagem,
        interpolation="nearest"
    )

    ax.set_xticks(
        np.arange(len(classes))
    )

    ax.set_yticks(
        np.arange(len(classes))
    )

    ax.set_xticklabels(
        classes,
        fontsize=11
    )

    ax.set_yticklabels(
        classes,
        fontsize=11
    )

    ax.set_xlabel(
        "Classe Predita",
        fontsize=12
    )

    ax.set_ylabel(
        "Classe Real",
        fontsize=12
    )

    ax.set_title(
        "Matriz de Confusão - FFT & SVM",
        fontsize=14,
        fontweight="bold"
    )

    # Valores dentro das células
    for i in range(len(classes)):

        for j in range(len(classes)):

            valor = cm[i, j]

            if valor > 0:

                # Texto branco para valores grandes
                if i == j and valor > 50:
                    cor_texto = "white"
                elif i != j and valor > max_erro * 0.55:
                    cor_texto = "white"
                else:
                    cor_texto = "black"

                ax.text(
                    j,
                    i,
                    str(valor),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=cor_texto
                )

    plt.tight_layout()

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 12. CURVA ROC
# =============================================================================

def salvar_curva_roc(
    y_true,
    y_prob,
    classes,
    caminho
):

    y_bin = label_binarize(
        y_true,
        classes=classes
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for i, classe in enumerate(classes):

        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            y_prob[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{classe} (AUC = {roc_auc:.3f})"
        )

    # ROC macro
    fpr_macro, tpr_macro, _ = roc_curve(
        y_bin.ravel(),
        y_prob.ravel()
    )

    auc_macro = auc(
        fpr_macro,
        tpr_macro
    )

    ax.plot(
        fpr_macro,
        tpr_macro,
        linewidth=3,
        linestyle="--",
        label=f"Macro-average (AUC = {auc_macro:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle=":",
        linewidth=1.5
    )

    ax.set_xlabel(
        "Taxa de Falsos Positivos",
        fontsize=12
    )

    ax.set_ylabel(
        "Taxa de Verdadeiros Positivos",
        fontsize=12
    )

    ax.set_title(
        "Curva ROC - FFT & SVM",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        fontsize=9,
        loc="lower right"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 13. CURVA PRECISION-RECALL
# =============================================================================

def salvar_curva_pr(
    y_true,
    y_prob,
    classes,
    caminho
):

    y_bin = label_binarize(
        y_true,
        classes=classes
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    for i, classe in enumerate(classes):

        precision, recall, _ = precision_recall_curve(
            y_bin[:, i],
            y_prob[:, i]
        )

        ap = average_precision_score(
            y_bin[:, i],
            y_prob[:, i]
        )

        ax.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{classe} (AP = {ap:.3f})"
        )

    ax.set_xlabel(
        "Recall",
        fontsize=12
    )

    ax.set_ylabel(
        "Precision",
        fontsize=12
    )

    ax.set_title(
        "Curva Precision-Recall - FFT & SVM",
        fontsize=14,
        fontweight="bold"
    )

    ax.legend(
        fontsize=9,
        loc="lower left"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# 14. PIPELINE PRINCIPAL
# =============================================================================

def executar_pipeline():

    tempo_inicio_total = time.perf_counter()

    # -------------------------------------------------------------------------
    # CARREGAMENTO
    # -------------------------------------------------------------------------

    X_sinais, y = carregar_dataset()

    print("\n" + "=" * 78)
    print("PROCESSAMENTO FFT")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # FFT
    # -------------------------------------------------------------------------

    X_fft, frequencies = extrair_fft(
        X_sinais,
        FS
    )

    print(
        f"Sinais: {X_fft.shape[0]}"
    )

    print(
        f"Amostras por sinal: {X_sinais.shape[1]}"
    )

    print(
        f"Características FFT: {X_fft.shape[1]}"
    )

    print(
        f"Resolução espectral: "
        f"{FS / X_sinais.shape[1]:.2f} Hz"
    )

    # -------------------------------------------------------------------------
    # ESTRUTURAS PARA GUARDAR RESULTADOS
    # -------------------------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados_folds = []

    y_true_global = []
    y_pred_global = []
    y_prob_global = []

    frequencias_folds = []

    # -------------------------------------------------------------------------
    # K-FOLD
    # -------------------------------------------------------------------------

    for fold, (train_idx, test_idx) in enumerate(
        skf.split(X_fft, y),
        start=1
    ):

        tempo_fold_inicio = time.perf_counter()

        print(
            f"\nFold {fold}/{N_SPLITS}"
        )

        # -------------------------------------------------------------
        # Separação treinamento / validação
        # -------------------------------------------------------------

        X_train = X_fft[train_idx]
        X_test = X_fft[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # -------------------------------------------------------------
        # CORRELAÇÃO
        # -------------------------------------------------------------

        indices_corr = filtro_correlacao(
            X_train,
            CORRELATION_THRESHOLD
        )

        X_train_corr = X_train[
            :,
            indices_corr
        ]

        X_test_corr = X_test[
            :,
            indices_corr
        ]

        # -------------------------------------------------------------
        # FISHER
        # -------------------------------------------------------------

        indices_fisher_local, fisher_scores = selecionar_fisher(
            X_train_corr,
            y_train,
            N_FISHER_FEATURES
        )

        X_train_fisher = X_train_corr[
            :,
            indices_fisher_local
        ]

        X_test_fisher = X_test_corr[
            :,
            indices_fisher_local
        ]

        # Índices Fisher em relação à FFT original
        indices_fisher_global = indices_corr[
            indices_fisher_local
        ]

        freq_selecionadas = frequencies[
            indices_fisher_global
        ]

        frequencias_folds.append(
            {
                "Fold": fold,
                "Frequencias_Hz": ", ".join(
                    f"{f:.0f}"
                    for f in freq_selecionadas
                )
            }
        )

        # -------------------------------------------------------------
        # NORMALIZAÇÃO
        # -------------------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train_fisher
        )

        X_test_scaled = scaler.transform(
            X_test_fisher
        )

        # -------------------------------------------------------------
        # SVM
        # -------------------------------------------------------------

        modelo = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=True,
            random_state=RANDOM_STATE
        )

        modelo.fit(
            X_train_scaled,
            y_train
        )

        # -------------------------------------------------------------
        # PREDIÇÃO
        # -------------------------------------------------------------

        y_pred = modelo.predict(
            X_test_scaled
        )

        y_prob = modelo.predict_proba(
            X_test_scaled
        )

        # Garante que as colunas das probabilidades estejam na ordem
        # das classes globais
        classes_modelo = modelo.classes_

        y_prob_ordenado = np.zeros(
            (
                len(y_test),
                len(CLASSES)
            )
        )

        for i, classe in enumerate(classes_modelo):

            indice = CLASSES.index(
                classe
            )

            y_prob_ordenado[:, indice] = y_prob[:, i]

        y_prob = y_prob_ordenado

        # -------------------------------------------------------------
        # MÉTRICAS
        # -------------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision_macro = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall_macro = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1_macro = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        balanced_acc = balanced_accuracy_score(
            y_test,
            y_pred
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        kappa = cohen_kappa_score(
            y_test,
            y_pred
        )

        y_test_bin = label_binarize(
            y_test,
            classes=CLASSES
        )

        roc_auc_macro = roc_auc_score(
            y_test_bin,
            y_prob,
            average="macro",
            multi_class="ovr"
        )

        logloss = log_loss(
            y_test,
            y_prob,
            labels=CLASSES
        )

        tempo_fold = (
            time.perf_counter()
            - tempo_fold_inicio
        )

        resultados_folds.append(
            {
                "Fold": fold,
                "Accuracy": accuracy,
                "Precision_macro": precision_macro,
                "Recall_macro": recall_macro,
                "F1_macro": f1_macro,
                "Balanced_accuracy": balanced_acc,
                "MCC": mcc,
                "Cohen_Kappa": kappa,
                "ROC_AUC_macro": roc_auc_macro,
                "Log_Loss": logloss,
                "Features_FFT": X_fft.shape[1],
                "Features_apos_correlacao": len(indices_corr),
                "Features_Fisher": len(indices_fisher_local),
                "Tempo_s": tempo_fold
            }
        )

        # Guarda predições de todos os folds
        y_true_global.extend(
            y_test
        )

        y_pred_global.extend(
            y_pred
        )

        y_prob_global.extend(
            y_prob
        )


    # =============================================================================
    # 15. RESULTADOS GLOBAIS
    # =============================================================================

    y_true_global = np.asarray(
        y_true_global
    )

    y_pred_global = np.asarray(
        y_pred_global
    )

    y_prob_global = np.asarray(
        y_prob_global
    )

    # -------------------------------------------------------------------------
    # MÉTRICAS GLOBAIS
    # -------------------------------------------------------------------------

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

    balanced_global = balanced_accuracy_score(
        y_true_global,
        y_pred_global
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
        classes=CLASSES
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
        labels=CLASSES
    )

    # =============================================================================
    # 16. MATRIZ DE CONFUSÃO
    # =============================================================================

    cm = confusion_matrix(
        y_true_global,
        y_pred_global,
        labels=CLASSES
    )

    caminho_cm = os.path.join(
        DIR_CONFUSION,
        "matriz_confusao_FFT_SVM.png"
    )

    salvar_matriz_confusao(
        cm,
        CLASSES,
        caminho_cm
    )

    # =============================================================================
    # 17. CURVA ROC
    # =============================================================================

    caminho_roc = os.path.join(
        DIR_ROC,
        "curva_ROC_FFT_SVM.png"
    )

    salvar_curva_roc(
        y_true_global,
        y_prob_global,
        CLASSES,
        caminho_roc
    )

    # =============================================================================
    # 18. CURVA PRECISION-RECALL
    # =============================================================================

    caminho_pr = os.path.join(
        DIR_PR,
        "curva_Precision_Recall_FFT_SVM.png"
    )

    salvar_curva_pr(
        y_true_global,
        y_prob_global,
        CLASSES,
        caminho_pr
    )

    # =============================================================================
    # 19. MÉTRICAS POR CLASSE
    # =============================================================================

    relatorio_dict = classification_report(
        y_true_global,
        y_pred_global,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0
    )

    metricas_classes = []

    for classe in CLASSES:

        metricas_classes.append(
            {
                "Classe": classe,
                "Precision": relatorio_dict[classe]["precision"],
                "Recall": relatorio_dict[classe]["recall"],
                "F1_score": relatorio_dict[classe]["f1-score"],
                "Support": relatorio_dict[classe]["support"]
            }
        )

    df_classes = pd.DataFrame(
        metricas_classes
    )

    caminho_classes = os.path.join(
        DIR_METRICS,
        "metricas_por_classe.csv"
    )

    df_classes.to_csv(
        caminho_classes,
        index=False,
        encoding="utf-8-sig"
    )

    # =============================================================================
    # 20. MÉTRICAS POR FOLD
    # =============================================================================

    df_folds = pd.DataFrame(
        resultados_folds
    )

    caminho_folds = os.path.join(
        DIR_METRICS,
        "metricas_por_fold.csv"
    )

    df_folds.to_csv(
        caminho_folds,
        index=False,
        encoding="utf-8-sig"
    )

    # =============================================================================
    # 21. FREQUÊNCIAS SELECIONADAS
    # =============================================================================

    df_freq = pd.DataFrame(
        frequencias_folds
    )

    caminho_freq = os.path.join(
        DIR_METRICS,
        "frequencias_selecionadas_por_fold.csv"
    )

    df_freq.to_csv(
        caminho_freq,
        index=False,
        encoding="utf-8-sig"
    )

    # =============================================================================
    # 22. INTERVALOS DE CONFIANÇA
    # =============================================================================

    df_bootstrap = bootstrap_ci(
        y_true_global,
        y_pred_global,
        N_BOOTSTRAP,
        RANDOM_STATE
    )

    caminho_bootstrap = os.path.join(
        DIR_METRICS,
        "intervalos_confianca_bootstrap.csv"
    )

    df_bootstrap.to_csv(
        caminho_bootstrap,
        index=False,
        encoding="utf-8-sig"
    )

    # =============================================================================
    # 23. MÉTRICAS FINAIS
    # =============================================================================

    tempo_total = (
        time.perf_counter()
        - tempo_inicio_total
    )

    metricas_finais = pd.DataFrame(
        [
            {
                "Accuracy": accuracy_global,
                "Precision_macro": precision_global,
                "Recall_macro": recall_global,
                "F1_macro": f1_global,
                "Balanced_accuracy": balanced_global,
                "MCC": mcc_global,
                "Cohen_Kappa": kappa_global,
                "ROC_AUC_macro": roc_auc_global,
                "Log_Loss": logloss_global,
                "Tempo_total_s": tempo_total
            }
        ]
    )

    caminho_finais = os.path.join(
        DIR_METRICS,
        "metricas_finais.csv"
    )

    metricas_finais.to_csv(
        caminho_finais,
        index=False,
        encoding="utf-8-sig"
    )

    # =============================================================================
    # 24. RELATÓRIO TXT
    # =============================================================================

    caminho_relatorio = os.path.join(
        DIR_REPORT,
        "relatorio_FFT_SVM.txt"
    )

    with open(
        caminho_relatorio,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "=" * 80 + "\n"
        )

        arquivo.write(
            "RELATÓRIO FINAL - FFT & SVM\n"
        )

        arquivo.write(
            "=" * 80 + "\n\n"
        )

        arquivo.write(
            "CONFIGURAÇÃO DO EXPERIMENTO\n"
        )

        arquivo.write(
            "-" * 80 + "\n"
        )

        arquivo.write(
            f"Frequência de amostragem: {FS} Hz\n"
        )

        arquivo.write(
            f"Frequência fundamental: {FUNDAMENTAL} Hz\n"
        )

        arquivo.write(
            f"Folds: {N_SPLITS}\n"
        )

        arquivo.write(
            f"Threshold de correlação: "
            f"{CORRELATION_THRESHOLD}\n"
        )

        arquivo.write(
            f"Características Fisher: "
            f"{N_FISHER_FEATURES}\n"
        )

        arquivo.write(
            f"SVM kernel: {SVM_KERNEL}\n"
        )

        arquivo.write(
            f"SVM C: {SVM_C}\n\n"
        )

        arquivo.write(
            "MÉTRICAS GLOBAIS\n"
        )

        arquivo.write(
            "-" * 80 + "\n"
        )

        arquivo.write(
            f"Accuracy: "
            f"{accuracy_global:.4f} "
            f"({accuracy_global * 100:.2f}%)\n"
        )

        arquivo.write(
            f"Precision Macro: "
            f"{precision_global:.4f} "
            f"({precision_global * 100:.2f}%)\n"
        )

        arquivo.write(
            f"Recall Macro: "
            f"{recall_global:.4f} "
            f"({recall_global * 100:.2f}%)\n"
        )

        arquivo.write(
            f"F1 Macro: "
            f"{f1_global:.4f} "
            f"({f1_global * 100:.2f}%)\n"
        )

        arquivo.write(
            f"Balanced Accuracy: "
            f"{balanced_global:.4f}\n"
        )

        arquivo.write(
            f"MCC: "
            f"{mcc_global:.4f}\n"
        )

        arquivo.write(
            f"Cohen's Kappa: "
            f"{kappa_global:.4f}\n"
        )

        arquivo.write(
            f"ROC-AUC Macro: "
            f"{roc_auc_global:.4f}\n"
        )

        arquivo.write(
            f"Log Loss: "
            f"{logloss_global:.4f}\n"
        )

        arquivo.write(
            f"Tempo total: "
            f"{tempo_total:.4f} s\n\n"
        )

        arquivo.write(
            "MÉTRICAS POR CLASSE\n"
        )

        arquivo.write(
            "-" * 80 + "\n"
        )

        arquivo.write(
            df_classes.to_string(
                index=False
            )
        )

        arquivo.write(
            "\n\nMATRIZ DE CONFUSÃO\n"
        )

        arquivo.write(
            "-" * 80 + "\n"
        )

        arquivo.write(
            str(cm)
        )

        arquivo.write(
            "\n\nMÉTRICAS POR FOLD\n"
        )

        arquivo.write(
            "-" * 80 + "\n"
        )

        arquivo.write(
            df_folds.to_string(
                index=False
            )
        )

        arquivo.write(
            "\n\nINTERVALOS DE CONFIANÇA - BOOTSTRAP 95%\n"
        )

        arquivo.write(
            "-" * 80 + "\n"
        )

        arquivo.write(
            df_bootstrap.to_string(
                index=False
            )
        )

    # =============================================================================
    # 25. RESULTADOS FINAIS NO TERMINAL
    # =============================================================================

    print("\n")
    print("=" * 78)
    print("RESULTADOS FINAIS - FFT & SVM")
    print("=" * 78)

    print(
        f"\nAccuracy          : "
        f"{accuracy_global:.4f} "
        f"({accuracy_global * 100:.2f}%)"
    )

    print(
        f"Precision Macro   : "
        f"{precision_global:.4f} "
        f"({precision_global * 100:.2f}%)"
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

    print(
        f"Tempo total       : "
        f"{tempo_total:.2f} s"
    )

    print("\n" + "-" * 78)
    print("DESEMPENHO POR CLASSE")
    print("-" * 78)

    print(
        df_classes.to_string(
            index=False,
            formatters={
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1_score": "{:.4f}".format
            }
        )
    )

    print("\n" + "-" * 78)
    print("MATRIZ DE CONFUSÃO")
    print("-" * 78)

    print(cm)

    print("\n" + "-" * 78)
    print("ARQUIVOS GERADOS")
    print("-" * 78)

    print(
        f"Matriz de confusão:\n{caminho_cm}"
    )

    print(
        f"\nCurva ROC:\n{caminho_roc}"
    )

    print(
        f"\nCurva Precision-Recall:\n{caminho_pr}"
    )

    print(
        f"\nMétricas por fold:\n{caminho_folds}"
    )

    print(
        f"\nMétricas finais:\n{caminho_finais}"
    )

    print(
        f"\nMétricas por classe:\n{caminho_classes}"
    )

    print(
        f"\nIntervalos de confiança:\n{caminho_bootstrap}"
    )

    print(
        f"\nFrequências selecionadas:\n{caminho_freq}"
    )

    print(
        f"\nRelatório:\n{caminho_relatorio}"
    )

    print("\n" + "=" * 78)


# =============================================================================
# 26. EXECUÇÃO COM CODECARBON
# =============================================================================

def main():

    print("\n")
    print("=" * 78)
    print("FFT & SVM - CLASSIFICADOR DE DISTÚRBIOS DE QUALIDADE DE ENERGIA")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # CodeCarbon
    #
    # A pasta é criada antes de inicializar o tracker.
    # Isso evita o erro:
    # -------------------------------------------------------------------------

    os.makedirs(
        DIR_CODECARBON,
        exist_ok=True
    )

    emissions_file = os.path.join(
        DIR_CODECARBON,
        "emissions.csv"
    )

    tracker = EmissionsTracker(
        project_name="FFT_SVM_PQD",
        output_dir=DIR_CODECARBON,
        output_file="emissions.csv",
        log_level="error"
    )

    try:

        tracker.start()

        executar_pipeline()

    finally:

        emissions_kg = tracker.stop()

        if emissions_kg is None:
            emissions_kg = 0.0

        emissions_g = (
            emissions_kg * 1000
        )

        print("\n" + "=" * 78)
        print("CODECARBON")
        print("=" * 78)

        print(
            f"Emissão estimada: "
            f"{emissions_kg:.8f} kg CO2"
        )

        print(
            f"Emissão estimada: "
            f"{emissions_g:.4f} g CO2"
        )

        print(
            f"\nArquivo CodeCarbon:\n"
            f"{emissions_file}"
        )

        # ---------------------------------------------------------------------
        # Resumo do CodeCarbon
        # ---------------------------------------------------------------------

        resumo_codecarbon = pd.DataFrame(
            [
                {
                    "Emissao_kg_CO2": emissions_kg,
                    "Emissao_g_CO2": emissions_g
                }
            ]
        )

        caminho_resumo = os.path.join(
            DIR_CODECARBON,
            "resumo_codecarbon.csv"
        )

        resumo_codecarbon.to_csv(
            caminho_resumo,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"Resumo CodeCarbon:\n"
            f"{caminho_resumo}"
        )

        print("\n" + "=" * 78)
        print("PROGRAMA FINALIZADO")
        print("=" * 78)


# =============================================================================
# 27. EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    main()