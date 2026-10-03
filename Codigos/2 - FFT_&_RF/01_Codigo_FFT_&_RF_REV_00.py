# =============================================================================
# CLASSIFICADOR DE DISTÚRBIOS DE QUALIDADE DE ENERGIA
# FFT + CORRELAÇÃO + FISHER + RANDOM FOREST + K-FOLD
#
# Autor: Rafael Bruno da Silva
#
# DESCRIÇÃO:
# Este programa implementa um pipeline completo para classificação de
# distúrbios de qualidade de energia elétrica.
#
# Pipeline:
#
#   1. Carregamento dos sinais
#   2. Transformada Rápida de Fourier (FFT)
#   3. Filtro de correlação
#   4. Fisher Discriminant Ratio
#   5. Random Forest
#   6. Validação cruzada estratificada 5-Fold
#   7. Métricas de desempenho
#   8. Matriz de confusão
#   9. Curva ROC
#  10. Curva Precision-Recall
#  11. Intervalos de confiança por bootstrap
#  12. Estimativa de emissão de CO2 com CodeCarbon
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

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

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


# =============================================================================
# 2. CONFIGURAÇÕES GERAIS
# =============================================================================

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# CAMINHO DOS ARQUIVOS
# -----------------------------------------------------------------------------
#
# ALTERE SOMENTE ESTA PARTE caso seus arquivos estejam em outra pasta.
#
# Dentro desta pasta devem existir:
#
# signal_capc.csv
# signal_har.csv
# signal_normal.csv
# signal_notch.csv
# signal_sag.csv
# signal_spike.csv
# signal_swell.csv
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
# PASTA PRINCIPAL DOS RESULTADOS
# -----------------------------------------------------------------------------

PASTA_RESULTADOS = (
    Path(__file__).resolve().parent
    / "Resultados"
    / "Resultados_REV_00"
)


# =============================================================================
# 3. CRIAÇÃO DAS PASTAS DE RESULTADOS
# =============================================================================

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

PASTA_CODECARBON = os.path.join(
    PASTA_RESULTADOS,
    "CodeCarbon"
)


# Cria todas as pastas caso elas ainda não existam.

for pasta in [
    PASTA_RESULTADOS,
    PASTA_MATRIZ,
    PASTA_ROC,
    PASTA_PR,
    PASTA_METRICAS,
    PASTA_RELATORIO,
    PASTA_CODECARBON
]:
    os.makedirs(pasta, exist_ok=True)


# =============================================================================
# 4. CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================

FS = 15360

# Frequência fundamental da rede elétrica.
FUNDAMENTAL = 60

# Número de folds.
N_SPLITS = 5

# Estado aleatório.
RANDOM_STATE = 42

# Limite utilizado no filtro de correlação.
LIMIAR_CORRELACAO = 0.95

# Número de características selecionadas pelo Fisher.
N_FEATURES_FISHER = 14

# Número de árvores do Random Forest.
N_ESTIMATORS = 300


# =============================================================================
# 5. CLASSES
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


# =============================================================================
# 6. FUNÇÃO PARA CARREGAR O DATASET
# =============================================================================

def carregar_dataset():

    print("\n")
    print("=" * 80)
    print("CARREGANDO DATASET")
    print("=" * 80)

    sinais = []
    labels = []

    for classe in CLASSES:

        arquivo = os.path.join(
            PASTA_DATASET,
            f"signal_{classe}.csv"
        )

        if not os.path.exists(arquivo):

            raise FileNotFoundError(
                f"\nArquivo não encontrado:\n{arquivo}"
            )

        # Leitura do CSV.
        dados = pd.read_csv(
            arquivo,
            header=None
        )

        # Converte para numpy.
        dados = dados.values.astype(float)

        print(
            f"{classe:<10} -> "
            f"arquivo = {os.path.basename(arquivo):<30} "
            f"shape = {dados.shape}"
        )

        # Adiciona os sinais.
        sinais.append(dados)

        # Adiciona os rótulos.
        labels.extend(
            [classe] * len(dados)
        )

    # Junta todos os sinais.
    X = np.vstack(sinais)

    # Vetor de classes.
    y = np.array(labels)

    print("\nNúmero total de sinais:", X.shape[0])
    print("Número de amostras por sinal:", X.shape[1])
    print("Número de classes:", len(np.unique(y)))

    print("\nDistribuição das classes:")

    print(
        pd.Series(y).value_counts().sort_index()
    )

    return X, y


# =============================================================================
# 7. TRANSFORMADA RÁPIDA DE FOURIER
# =============================================================================
#
# A FFT transforma o sinal do domínio do tempo para o domínio da frequência.
#
# Como o sinal possui 2560 amostras, a FFT completa teria 2560 componentes.
#
# Como os sinais são reais, utilizamos rFFT:
#
#       np.fft.rfft()
#
# que retorna apenas a metade positiva do espectro.
#
# Para N = 2560:
#
#       N/2 + 1 = 1281
#
# Portanto:
#
#       2560 amostras
#              ↓
#       1281 características FFT
#
# =============================================================================

def extrair_fft(X):

    print("\n")
    print("=" * 80)
    print("ETAPA 2 - TRANSFORMADA RÁPIDA DE FOURIER")
    print("=" * 80)

    N = X.shape[1]

    # FFT unilateral.
    espectro = np.abs(
        np.fft.rfft(X, axis=1)
    )

    # Frequências correspondentes.
    frequencias = np.fft.rfftfreq(
        N,
        d=1 / FS
    )

    # Normalização da amplitude.
    espectro[:, 1:-1] *= 2

    print(
        f"Sinais: {X.shape[0]}"
    )

    print(
        f"Amostras por sinal: {N}"
    )

    print(
        f"Características FFT: {espectro.shape[1]}"
    )

    print(
        f"Resolução espectral: {FS / N:.2f} Hz"
    )

    return espectro, frequencias


# =============================================================================
# 8. FILTRO DE CORRELAÇÃO
# =============================================================================
#
# Muitas características FFT podem carregar praticamente a mesma informação.
#
# O filtro de correlação remove características altamente correlacionadas.
#
# Exemplo:
#
#       correlação > 0.95
#
# então uma das duas características é removida.
#
# IMPORTANTE:
# O filtro é calculado SOMENTE no conjunto de treinamento de cada fold.
#
# Isso evita vazamento de informação (data leakage).
#
# =============================================================================

def filtro_correlacao(X_train, limiar=0.95):

    matriz_corr = pd.DataFrame(
        X_train
    ).corr()

    # Triângulo superior da matriz.
    triangulo_superior = np.triu(
        np.ones(matriz_corr.shape),
        k=1
    ).astype(bool)

    corr_superior = matriz_corr.where(
        triangulo_superior
    )

    # Identifica as colunas que devem ser removidas.
    colunas_remover = [
        coluna
        for coluna in corr_superior.columns
        if any(
            abs(corr_superior[coluna]) > limiar
        )
    ]

    indices_remover = [
        matriz_corr.columns.get_loc(coluna)
        for coluna in colunas_remover
    ]

    indices_manter = [
        i
        for i in range(X_train.shape[1])
        if i not in indices_remover
    ]

    return np.array(indices_manter)


# =============================================================================
# 9. FISHER DISCRIMINANT RATIO
# =============================================================================
#
# O Fisher Discriminant Ratio (FDR) mede o quanto uma característica consegue
# separar as classes.
#
# Uma característica é interessante quando:
#
#       variância ENTRE classes é alta
#
# e
#
#       variância DENTRO das classes é baixa.
#
# Quanto maior o FDR, mais discriminativa é a característica.
#
# =============================================================================

def calcular_fisher(X, y):

    classes = np.unique(y)

    media_global = np.mean(
        X,
        axis=0
    )

    numerador = np.zeros(
        X.shape[1]
    )

    denominador = np.zeros(
        X.shape[1]
    )

    for classe in classes:

        X_classe = X[y == classe]

        n_classe = X_classe.shape[0]

        media_classe = np.mean(
            X_classe,
            axis=0
        )

        # Dispersão entre classes.
        numerador += (
            n_classe
            *
            (media_classe - media_global) ** 2
        )

        # Dispersão dentro da classe.
        denominador += np.sum(
            (X_classe - media_classe) ** 2,
            axis=0
        )

    # Evita divisão por zero.
    denominador += 1e-12

    fisher = (
        numerador /
        denominador
    )

    return fisher


# =============================================================================
# 10. SELEÇÃO DAS CARACTERÍSTICAS PELO FISHER
# =============================================================================

def selecionar_fisher(
    X_train,
    y_train,
    n_features
):

    fisher_scores = calcular_fisher(
        X_train,
        y_train
    )

    # Ordena do maior para o menor.
    indices_ordenados = np.argsort(
        fisher_scores
    )[::-1]

    # Seleciona as N melhores.
    indices_selecionados = (
        indices_ordenados[:n_features]
    )

    return indices_selecionados, fisher_scores


# =============================================================================
# 11. RANDOM FOREST
# =============================================================================
#
# O Random Forest é um conjunto de árvores de decisão.
#
# Cada árvore recebe uma amostra diferente dos dados e utiliza subconjuntos
# de características.
#
# A decisão final é obtida pela combinação das árvores.
#
# Aqui utilizamos:
#
#       n_estimators = 300
#
# ou seja, 300 árvores.
#
# n_jobs=-1:
# utiliza todos os núcleos disponíveis do processador.
#
# =============================================================================

def criar_random_forest():

    modelo = RandomForestClassifier(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        # Critério utilizado para divisão das árvores.
        criterion="gini",

        # Evita árvores excessivamente profundas.
        max_depth=None,

        # Número mínimo de amostras em uma folha.
        min_samples_leaf=1,

        # Utiliza todas as características disponíveis.
        max_features="sqrt"

    )

    return modelo


# =============================================================================
# 12. FUNÇÃO PARA CALCULAR MÉTRICAS
# =============================================================================

def calcular_metricas(
    y_true,
    y_pred,
    y_prob,
    classes
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    # Converte as classes para representação binária.
    y_true_bin = label_binarize(
        y_true,
        classes=classes
    )

    roc_auc = roc_auc_score(
        y_true_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    loss = log_loss(
        y_true,
        y_prob,
        labels=classes
    )

    return {
        "Accuracy": accuracy,
        "Precision_macro": precision,
        "Recall_macro": recall,
        "F1_macro": f1,
        "MCC": mcc,
        "Cohen_Kappa": kappa,
        "ROC_AUC_macro": roc_auc,
        "Log_Loss": loss
    }


# =============================================================================
# 13. CURVA ROC
# =============================================================================

def salvar_curva_roc(
    y_true,
    y_prob,
    classes
):

    y_bin = label_binarize(
        y_true,
        classes=classes
    )

    plt.figure(
        figsize=(9, 7)
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

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{classe} (AUC = {roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    macro_auc = roc_auc_score(
        y_bin,
        y_prob,
        average="macro",
        multi_class="ovr"
    )

    plt.title(
        "Curva ROC - FFT & Random Forest"
    )

    plt.xlabel(
        "Taxa de Falsos Positivos"
    )

    plt.ylabel(
        "Taxa de Verdadeiros Positivos"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    arquivo = os.path.join(
        PASTA_ROC,
        "curva_ROC_FFT_RandomForest.png"
    )

    plt.savefig(
        arquivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return arquivo


# =============================================================================
# 14. CURVA PRECISION-RECALL
# =============================================================================

def salvar_curva_precision_recall(
    y_true,
    y_prob,
    classes
):

    y_bin = label_binarize(
        y_true,
        classes=classes
    )

    plt.figure(
        figsize=(9, 7)
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

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{classe} (AP = {ap:.3f})"
        )

    plt.title(
        "Curva Precision-Recall - FFT & Random Forest"
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.legend(
        loc="lower left"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    arquivo = os.path.join(
        PASTA_PR,
        "curva_Precision_Recall_FFT_RandomForest.png"
    )

    plt.savefig(
        arquivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return arquivo


# =============================================================================
# 15. MATRIZ DE CONFUSÃO
# =============================================================================
#
# A diagonal representa os acertos.
#
# Os valores fora da diagonal representam erros.
#
# A escala azul representa os acertos.
# A escala laranja representa os erros.
#
# Quanto maior o número:
#
#       azul mais forte = mais acertos
#
#       laranja mais forte = mais erros
#
# =============================================================================

def salvar_matriz_confusao(
    cm,
    classes
):

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    maior_valor = np.max(cm)

    # Desenha a matriz.
    ax.imshow(
        np.zeros_like(cm),
        cmap="Blues",
        vmin=0,
        vmax=maior_valor
    )

    # Coloca cada célula individualmente.
    for i in range(len(classes)):

        for j in range(len(classes)):

            valor = cm[i, j]

            # Diagonal = azul.
            if i == j:

                # Quanto maior o acerto, mais escuro.
                intensidade = (
                    valor / maior_valor
                    if maior_valor > 0
                    else 0
                )

                cor = plt.cm.Blues(
                    0.15 + 0.75 * intensidade
                )

            # Fora da diagonal = laranja.
            else:

                max_erro = np.max(
                    cm - np.diag(np.diag(cm))
                )

                if max_erro > 0:

                    intensidade = (
                        valor / max_erro
                    )

                else:

                    intensidade = 0

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

            # Cor do texto.
            if (
                i == j and
                intensidade > 0.55
            ) or (
                i != j and
                intensidade > 0.55
            ):

                cor_texto = "white"

            else:

                cor_texto = "black"

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

    ax.set_xticks(
        np.arange(len(classes))
    )

    ax.set_yticks(
        np.arange(len(classes))
    )

    ax.set_xticklabels(
        classes
    )

    ax.set_yticklabels(
        classes
    )

    ax.set_xlabel(
        "Classe Predita"
    )

    ax.set_ylabel(
        "Classe Real"
    )

    ax.set_title(
        "Matriz de Confusão - FFT & Random Forest"
    )

    plt.tight_layout()

    arquivo = os.path.join(
        PASTA_MATRIZ,
        "matriz_confusao_FFT_RandomForest.png"
    )

    plt.savefig(
        arquivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return arquivo


# =============================================================================
# 16. BOOTSTRAP
# =============================================================================
#
# O bootstrap é utilizado para estimar a incerteza das métricas.
#
# Aqui são realizadas 1000 reamostragens.
#
# O intervalo utilizado é de 95%.
#
# =============================================================================

def bootstrap_intervalo(
    y_true,
    y_pred,
    n_bootstrap=1000
):

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    n = len(y_true)

    resultados = {
        "Accuracy": [],
        "Precision_macro": [],
        "F1_macro": [],
        "MCC": [],
        "Cohen_Kappa": []
    }

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            n
        )

        yt = y_true[indices]
        yp = y_pred[indices]

        resultados["Accuracy"].append(
            accuracy_score(
                yt,
                yp
            )
        )

        resultados["Precision_macro"].append(
            precision_score(
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

    for metrica, valores in resultados.items():

        valores = np.array(
            valores
        )

        linhas.append({

            "Metrica": metrica,

            "Media": np.mean(
                valores
            ),

            "IC_95_inferior": np.percentile(
                valores,
                2.5
            ),

            "IC_95_superior": np.percentile(
                valores,
                97.5
            )
        })

    return pd.DataFrame(
        linhas
    )


# =============================================================================
# 17. FUNÇÃO PRINCIPAL
# =============================================================================

def executar_pipeline():

    inicio_total = time.time()

    # -------------------------------------------------------------------------
    # CARREGAMENTO
    # -------------------------------------------------------------------------

    X_sinais, y = carregar_dataset()

    # -------------------------------------------------------------------------
    # FFT
    # -------------------------------------------------------------------------

    X_fft, frequencias = extrair_fft(
        X_sinais
    )

    classes = np.array(
        CLASSES
    )

    # -------------------------------------------------------------------------
    # VALIDAÇÃO CRUZADA
    # -------------------------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados_folds = []

    # Armazena todas as predições.
    y_true_global = []
    y_pred_global = []
    y_prob_global = []

    # Armazena frequências selecionadas.
    frequencias_folds = []

    # -------------------------------------------------------------------------
    # LOOP DOS FOLDS
    # -------------------------------------------------------------------------

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        skf.split(
            X_fft,
            y
        ),
        start=1
    ):

        print(
            f"\nFold {fold}/{N_SPLITS}"
        )

        inicio_fold = time.time()

        # ---------------------------------------------------------------------
        # SEPARAÇÃO TREINAMENTO / VALIDAÇÃO
        # ---------------------------------------------------------------------

        X_train = X_fft[
            train_idx
        ]

        X_test = X_fft[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # ---------------------------------------------------------------------
        # FILTRO DE CORRELAÇÃO
        # ---------------------------------------------------------------------

        indices_corr = filtro_correlacao(
            X_train,
            LIMIAR_CORRELACAO
        )

        X_train_corr = X_train[
            :,
            indices_corr
        ]

        X_test_corr = X_test[
            :,
            indices_corr
        ]

        # ---------------------------------------------------------------------
        # FISHER
        # ---------------------------------------------------------------------

        indices_fisher_local, fisher_scores = selecionar_fisher(
            X_train_corr,
            y_train,
            N_FEATURES_FISHER
        )

        X_train_final = X_train_corr[
            :,
            indices_fisher_local
        ]

        X_test_final = X_test_corr[
            :,
            indices_fisher_local
        ]

        # Índices absolutos das características FFT.
        indices_fisher_global = (
            indices_corr[
                indices_fisher_local
            ]
        )

        frequencias_selecionadas = (
            frequencias[
                indices_fisher_global
            ]
        )

        frequencias_folds.append(
            frequencias_selecionadas
        )

        # ---------------------------------------------------------------------
        # RANDOM FOREST
        # ---------------------------------------------------------------------

        modelo = criar_random_forest()

        modelo.fit(
            X_train_final,
            y_train
        )

        # ---------------------------------------------------------------------
        # PREDIÇÃO
        # ---------------------------------------------------------------------

        y_pred = modelo.predict(
            X_test_final
        )

        y_prob = modelo.predict_proba(
            X_test_final
        )

        # ---------------------------------------------------------------------
        # MÉTRICAS
        # ---------------------------------------------------------------------

        metricas = calcular_metricas(
            y_test,
            y_pred,
            y_prob,
            classes
        )

        tempo_fold = (
            time.time() -
            inicio_fold
        )

        metricas["Fold"] = fold

        metricas["Features_FFT"] = X_fft.shape[1]

        metricas["Features_apos_correlacao"] = len(
            indices_corr
        )

        metricas["Features_Fisher"] = len(
            indices_fisher_local
        )

        metricas["Tempo_s"] = tempo_fold

        resultados_folds.append(
            metricas
        )

        # ---------------------------------------------------------------------
        # ARMAZENAMENTO DAS PREDIÇÕES
        # ---------------------------------------------------------------------

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
    # 18. CONVERSÃO PARA NUMPY
    # =============================================================================

    y_true_global = np.array(
        y_true_global
    )

    y_pred_global = np.array(
        y_pred_global
    )

    y_prob_global = np.array(
        y_prob_global
    )

    # =============================================================================
    # 19. MÉTRICAS GLOBAIS
    # =============================================================================

    metricas_global = calcular_metricas(
        y_true_global,
        y_pred_global,
        y_prob_global,
        classes
    )

    # =============================================================================
    # 20. MATRIZ DE CONFUSÃO
    # =============================================================================

    cm = confusion_matrix(
        y_true_global,
        y_pred_global,
        labels=classes
    )

    arquivo_cm = salvar_matriz_confusao(
        cm,
        classes
    )

    # =============================================================================
    # 21. CURVA ROC
    # =============================================================================

    arquivo_roc = salvar_curva_roc(
        y_true_global,
        y_prob_global,
        classes
    )

    # =============================================================================
    # 22. CURVA PRECISION-RECALL
    # =============================================================================

    arquivo_pr = salvar_curva_precision_recall(
        y_true_global,
        y_prob_global,
        classes
    )

    # =============================================================================
    # 23. MÉTRICAS POR CLASSE
    # =============================================================================

    relatorio_dict = classification_report(
        y_true_global,
        y_pred_global,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    metricas_classes = pd.DataFrame(
        relatorio_dict
    ).transpose()

    arquivo_classes = os.path.join(
        PASTA_METRICAS,
        "metricas_por_classe_FFT_RandomForest.csv"
    )

    metricas_classes.to_csv(
        arquivo_classes
    )

    # =============================================================================
    # 24. MÉTRICAS POR FOLD
    # =============================================================================

    df_folds = pd.DataFrame(
        resultados_folds
    )

    # Reordena as colunas.
    colunas_folds = [
        "Fold",
        "Accuracy",
        "Precision_macro",
        "Recall_macro",
        "F1_macro",
        "MCC",
        "Cohen_Kappa",
        "ROC_AUC_macro",
        "Log_Loss",
        "Features_FFT",
        "Features_apos_correlacao",
        "Features_Fisher",
        "Tempo_s"
    ]

    df_folds = df_folds[
        colunas_folds
    ]

    arquivo_folds = os.path.join(
        PASTA_METRICAS,
        "metricas_por_fold_FFT_RandomForest.csv"
    )

    df_folds.to_csv(
        arquivo_folds,
        index=False
    )

    # =============================================================================
    # 25. INTERVALOS DE CONFIANÇA
    # =============================================================================

    df_bootstrap = bootstrap_intervalo(
        y_true_global,
        y_pred_global,
        n_bootstrap=1000
    )

    arquivo_bootstrap = os.path.join(
        PASTA_METRICAS,
        "intervalos_confianca_bootstrap_FFT_RandomForest.csv"
    )

    df_bootstrap.to_csv(
        arquivo_bootstrap,
        index=False
    )

    # =============================================================================
    # 26. FREQUÊNCIAS SELECIONADAS
    # =============================================================================

    dados_freq = []

    for i, frequencias_fold in enumerate(
        frequencias_folds,
        start=1
    ):

        for ordem, freq in enumerate(
            frequencias_fold,
            start=1
        ):

            dados_freq.append({

                "Fold": i,

                "Ordem": ordem,

                "Frequencia_Hz": freq
            })

    df_freq = pd.DataFrame(
        dados_freq
    )

    arquivo_freq = os.path.join(
        PASTA_METRICAS,
        "frequencias_selecionadas_por_fold_FFT_RandomForest.csv"
    )

    df_freq.to_csv(
        arquivo_freq,
        index=False
    )

    # =============================================================================
    # 27. MÉTRICAS FINAIS
    # =============================================================================

    metricas_finais = {

        "Accuracy": metricas_global[
            "Accuracy"
        ],

        "Precision_macro": metricas_global[
            "Precision_macro"
        ],

        "Recall_macro": metricas_global[
            "Recall_macro"
        ],

        "F1_macro": metricas_global[
            "F1_macro"
        ],

        "MCC": metricas_global[
            "MCC"
        ],

        "Cohen_Kappa": metricas_global[
            "Cohen_Kappa"
        ],

        "ROC_AUC_macro": metricas_global[
            "ROC_AUC_macro"
        ],

        "Log_Loss": metricas_global[
            "Log_Loss"
        ]
    }

    df_finais = pd.DataFrame(
        [metricas_finais]
    )

    arquivo_finais = os.path.join(
        PASTA_METRICAS,
        "metricas_finais_FFT_RandomForest.csv"
    )

    df_finais.to_csv(
        arquivo_finais,
        index=False
    )

    # =============================================================================
    # 28. TEMPO TOTAL
    # =============================================================================

    tempo_total = (
        time.time() -
        inicio_total
    )

    # =============================================================================
    # 29. RELATÓRIO TXT
    # =============================================================================

    arquivo_relatorio = os.path.join(
        PASTA_RELATORIO,
        "relatorio_FFT_RandomForest.txt"
    )

    with open(
        arquivo_relatorio,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 80 +
            "\n"
        )

        f.write(
            "RELATÓRIO - FFT + RANDOM FOREST\n"
        )

        f.write(
            "=" * 80 +
            "\n\n"
        )

        f.write(
            "CONFIGURAÇÕES\n"
        )

        f.write(
            f"Frequência de amostragem: {FS} Hz\n"
        )

        f.write(
            f"Frequência fundamental: {FUNDAMENTAL} Hz\n"
        )

        f.write(
            f"Folds: {N_SPLITS}\n"
        )

        f.write(
            f"Random Forest - árvores: {N_ESTIMATORS}\n"
        )

        f.write(
            f"Limite correlação: {LIMIAR_CORRELACAO}\n"
        )

        f.write(
            f"Características Fisher: {N_FEATURES_FISHER}\n\n"
        )

        f.write(
            "MÉTRICAS FINAIS\n"
        )

        f.write(
            "-" * 80 +
            "\n"
        )

        for chave, valor in metricas_global.items():

            f.write(
                f"{chave:<20}: {valor:.6f}\n"
            )

        f.write(
            f"\nTempo total: {tempo_total:.2f} s\n"
        )

        f.write(
            f"Tempo total: {tempo_total / 60:.2f} min\n"
        )

        f.write(
            "\n\nMÉTRICAS POR CLASSE\n"
        )

        f.write(
            "-" * 80 +
            "\n"
        )

        f.write(
            classification_report(
                y_true_global,
                y_pred_global,
                labels=classes,
                target_names=classes,
                zero_division=0
            )
        )

        f.write(
            "\n\nMATRIZ DE CONFUSÃO\n"
        )

        f.write(
            "-" * 80 +
            "\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\nFREQUÊNCIAS SELECIONADAS POR FOLD\n"
        )

        f.write(
            "-" * 80 +
            "\n"
        )

        for i, freq in enumerate(
            frequencias_folds,
            start=1
        ):

            f.write(
                f"Fold {i}: "
                f"{np.round(freq, 2)} Hz\n"
            )

    # =============================================================================
    # 30. RESULTADOS NO TERMINAL
    # =============================================================================

    print("\n")
    print("=" * 80)
    print("RESULTADOS FINAIS - FFT + RANDOM FOREST")
    print("=" * 80)

    print(
        f"\nAccuracy          : "
        f"{metricas_global['Accuracy']:.4f} "
        f"({metricas_global['Accuracy'] * 100:.2f}%)"
    )

    print(
        f"Precision Macro   : "
        f"{metricas_global['Precision_macro']:.4f} "
        f"({metricas_global['Precision_macro'] * 100:.2f}%)"
    )

    print(
        f"F1 Macro          : "
        f"{metricas_global['F1_macro']:.4f} "
        f"({metricas_global['F1_macro'] * 100:.2f}%)"
    )

    print(
        f"MCC               : "
        f"{metricas_global['MCC']:.4f}"
    )

    print(
        f"Cohen's Kappa     : "
        f"{metricas_global['Cohen_Kappa']:.4f}"
    )

    print(
        f"ROC-AUC Macro     : "
        f"{metricas_global['ROC_AUC_macro']:.4f}"
    )

    print(
        f"Log Loss          : "
        f"{metricas_global['Log_Loss']:.4f}"
    )

    print(
        f"Tempo total       : "
        f"{tempo_total:.2f} segundos"
    )

    print(
        f"Tempo total       : "
        f"{tempo_total / 60:.2f} minutos"
    )

    # =============================================================================
    # 31. DESEMPENHO POR CLASSE
    # =============================================================================

    print("\n")
    print("=" * 80)
    print("DESEMPENHO POR CLASSE")
    print("=" * 80)

    print(
        classification_report(
            y_true_global,
            y_pred_global,
            labels=classes,
            target_names=classes,
            zero_division=0
        )
    )

    # =============================================================================
    # 32. MATRIZ DE CONFUSÃO NO TERMINAL
    # =============================================================================

    print("\n")
    print("=" * 80)
    print("MATRIZ DE CONFUSÃO")
    print("=" * 80)

    print(cm)

    # =============================================================================
    # 33. ARQUIVOS GERADOS
    # =============================================================================

    print("\n")
    print("=" * 80)
    print("ARQUIVOS GERADOS")
    print("=" * 80)

    print(
        "\nMatriz de confusão:"
    )

    print(
        arquivo_cm
    )

    print(
        "\nCurva ROC:"
    )

    print(
        arquivo_roc
    )

    print(
        "\nCurva Precision-Recall:"
    )

    print(
        arquivo_pr
    )

    print(
        "\nMétricas por fold:"
    )

    print(
        arquivo_folds
    )

    print(
        "\nMétricas finais:"
    )

    print(
        arquivo_finais
    )

    print(
        "\nMétricas por classe:"
    )

    print(
        arquivo_classes
    )

    print(
        "\nIntervalos de confiança:"
    )

    print(
        arquivo_bootstrap
    )

    print(
        "\nFrequências selecionadas:"
    )

    print(
        arquivo_freq
    )

    print(
        "\nRelatório:"
    )

    print(
        arquivo_relatorio
    )

    return (
        metricas_global,
        cm,
        y_true_global,
        y_pred_global,
        y_prob_global
    )


# =============================================================================
# 34. EXECUÇÃO COM CODECARBON
# =============================================================================
#
# O CodeCarbon deve ser iniciado UMA ÚNICA VEZ para o experimento inteiro.
#
# Não devemos iniciar um tracker dentro de cada fold.
#
# Isso evita o aviso:
#
#   Multiple instances of codecarbon are allowed to run at the same time.
#
# =============================================================================

def main():

    print("\n")
    print("=" * 80)
    print("FFT + RANDOM FOREST")
    print("CLASSIFICAÇÃO DE DISTÚRBIOS DE QUALIDADE DE ENERGIA")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CONFIGURAÇÃO DO CODECARBON
    # -------------------------------------------------------------------------

    tracker = EmissionsTracker(

        output_dir=PASTA_CODECARBON,

        project_name="FFT_RandomForest_PQD",

        measure_power_secs=1,

        save_to_file=True,

        log_level="error"
    )

    print("\nCodeCarbon iniciado.")

    print(
        "Diretório:"
    )

    print(
        PASTA_CODECARBON
    )

    # -------------------------------------------------------------------------
    # INÍCIO DO MONITORAMENTO
    # -------------------------------------------------------------------------

    tracker.start()

    try:

        resultados = executar_pipeline()

    finally:

        # ---------------------------------------------------------------------
        # FINALIZAÇÃO DO CODECARBON
        # ---------------------------------------------------------------------

        emissao = tracker.stop()

        print("\n")
        print("=" * 80)
        print("CODECARBON")
        print("=" * 80)

        print(
            f"\nEmissão estimada: "
            f"{emissao:.8f} kg CO2"
        )

        print(
            f"Emissão estimada: "
            f"{emissao * 1000:.4f} g CO2"
        )

        # ---------------------------------------------------------------------
        # RESUMO DO CODECARBON
        # ---------------------------------------------------------------------

        arquivo_emissions = os.path.join(
            PASTA_CODECARBON,
            "emissions.csv"
        )

        if os.path.exists(
            arquivo_emissions
        ):

            try:

                df_emissions = pd.read_csv(
                    arquivo_emissions
                )

                resumo = pd.DataFrame({

                    "Emissao_kg_CO2": [
                        emissao
                    ],

                    "Emissao_g_CO2": [
                        emissao * 1000
                    ],

                    "Tempo_total_s": [
                        df_emissions["duration"].iloc[-1]
                        if "duration" in df_emissions.columns
                        else np.nan
                    ]
                })

                arquivo_resumo = os.path.join(
                    PASTA_CODECARBON,
                    "resumo_codecarbon.csv"
                )

                resumo.to_csv(
                    arquivo_resumo,
                    index=False
                )

                print(
                    "\nResumo CodeCarbon:"
                )

                print(
                    arquivo_resumo
                )

            except Exception as erro:

                print(
                    "\nNão foi possível criar o resumo do CodeCarbon."
                )

                print(
                    erro
                )

    print("\n")
    print("=" * 80)
    print("PROGRAMA FINALIZADO")
    print("=" * 80)


# =============================================================================
# 35. EXECUTA O PROGRAMA
# =============================================================================

if __name__ == "__main__":

    main()