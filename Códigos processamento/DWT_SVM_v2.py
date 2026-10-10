# # =============================================================================
# CLASSIFICADOR DE DISTÚRBIOS DE QUALIDADE DE ENERGIA
# WAVELET + SVM-RBF
# =============================================================================
#
# PIPELINE:
#
# SINAL
#   ↓
# DWT
#   ↓
# 56 CARACTERÍSTICAS WAVELET
#   ↓
# STRATIFIED K-FOLD
#   ↓
# FILTRO DE CORRELAÇÃO
#   ↓
# FISHER DISCRIMINANT RATIO
#   ↓
# 14 CARACTERÍSTICAS
#   ↓
# STANDARD SCALER
#   ↓
# SVM-RBF
#   ↓
# PREDIÇÕES OOF
#   ↓
# MÉTRICAS
#   ↓
# MATRIZ DE CONFUSÃO
# ROC
# PRECISION-RECALL
# BOOTSTRAP
# CODECARBON
#
# IMPORTANTE:
#
# O processamento supervisionado é realizado dentro de cada fold.
#
# Dessa forma:
#
#     Correlação → somente treino
#     Fisher     → somente treino
#     Scaler     → somente treino
#     SVM        → somente treino
#
# O conjunto de teste de cada fold é utilizado somente para avaliação.
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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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

from sklearn.preprocessing import label_binarize

from codecarbon import EmissionsTracker

from pathlib import Path


# =============================================================================
# 2. CONFIGURAÇÕES GERAIS
# =============================================================================

# Evita que mensagens de warning poluam a saída do programa.
warnings.filterwarnings("ignore")


# =============================================================================
# 3. CAMINHOS DO PROJETO
# =============================================================================

# -----------------------------------------------------------------------------
# Pasta contendo exclusivamente os arquivos Signal_Feature.
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

PASTA_DATASET = (
    BASE_DIR
    / "Banco_de_Dados"
    / "CSV"
    / "Signal_Feature"
)


# -----------------------------------------------------------------------------
# Pasta principal onde todos os resultados serão armazenados.
# -----------------------------------------------------------------------------

PASTA_RESULTADOS = (
    Path(__file__).resolve().parent
    / "Resultados"
    / "Resultados Testes"
    / "Teste 1 - DWT & SVM"
)


# =============================================================================
# 4. SUBPASTAS DOS RESULTADOS
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


# -----------------------------------------------------------------------------
# Criação automática das pastas.
# -----------------------------------------------------------------------------

for pasta in [
    PASTA_RESULTADOS,
    PASTA_CODECARBON,
    PASTA_MATRIZ,
    PASTA_ROC,
    PASTA_PR,
    PASTA_METRICAS,
    PASTA_RELATORIO,
    PASTA_WAVELET,
    PASTA_DOMINIO_WAVELET
]:

    os.makedirs(
        pasta,
        exist_ok=True
    )


# =============================================================================
# 5. CONFIGURAÇÃO DO DATASET
# =============================================================================

# Classes presentes no dataset.
CLASSES = [
    "capc",
    "har",
    "normal",
    "notch",
    "sag",
    "spike",
    "swell"
]


# Frequência de amostragem do sinal.
FS = 15360


# Frequência fundamental do sistema elétrico.
FUNDAMENTAL = 60


# =============================================================================
# 6. CONFIGURAÇÃO DA TRANSFORMADA WAVELET
# =============================================================================

# Wavelet-mãe utilizada na decomposição.
WAVELET = "db4"


# Nível máximo de decomposição.
NIVEL_WAVELET = 6


# =============================================================================
# 7. CONFIGURAÇÃO DO FISHER DISCRIMINANT RATIO
# =============================================================================

# Número final de características selecionadas.
N_FEATURES_FISHER = 14


# =============================================================================
# 8. CONFIGURAÇÃO DO FILTRO DE CORRELAÇÃO
# =============================================================================

# Duas características são consideradas altamente correlacionadas
# quando |r| > 0.95.
LIMITE_CORRELACAO = 0.95


# =============================================================================
# 9. CONFIGURAÇÃO DO SVM
# =============================================================================

# Kernel utilizado pelo SVM.
SVM_KERNEL = "rbf"


# Parâmetro de regularização.
SVM_C = 10


# Gamma utilizado pelo kernel RBF.
#
# "scale" faz o próprio sklearn calcular o valor de gamma
# com base nos dados de treinamento.
SVM_GAMMA = "scale"


# Ativado para permitir:
#
#     predict_proba()
#
# necessário para:
#
#     ROC-AUC
#     Log Loss
#     Precision-Recall
SVM_PROBABILITY = True


# =============================================================================
# 10. CONFIGURAÇÃO DA VALIDAÇÃO CRUZADA
# =============================================================================

# Número de folds.
N_SPLITS = 5


# Semente aleatória para permitir reprodução.
RANDOM_STATE = 42


# =============================================================================
# 11. CONFIGURAÇÃO DO BOOTSTRAP
# =============================================================================

# Número de reamostragens.
N_BOOTSTRAP = 1000


# =============================================================================
# 12. CARREGAMENTO DO DATASET
# =============================================================================

def carregar_dataset():

    """
    Carrega os sinais pertencentes às sete classes.

    Cada arquivo possui sinais de uma determinada classe.

    Exemplo:

        signal_capc.csv
        signal_har.csv
        signal_normal.csv
        ...

    Ao final:

        X → sinais
        y → classes correspondentes
    """

    print()
    print("=" * 80)
    print("CARREGANDO DATASET - SIGNAL_FEATURE")
    print("=" * 80)
    print()

    sinais = []
    labels = []


    # -------------------------------------------------------------------------
    # Percorre todas as classes.
    # -------------------------------------------------------------------------

    for classe in CLASSES:

        arquivo = os.path.join(
            PASTA_DATASET,
            f"signal_{classe}.csv"
        )


        # ---------------------------------------------------------------------
        # Verifica se o arquivo existe.
        # ---------------------------------------------------------------------

        if not os.path.exists(arquivo):

            raise FileNotFoundError(
                f"\nArquivo não encontrado:\n{arquivo}\n"
            )


        # ---------------------------------------------------------------------
        # Lê o arquivo CSV.
        #
        # header=None porque o arquivo não possui cabeçalho.
        # ---------------------------------------------------------------------

        dados = pd.read_csv(
            arquivo,
            header=None
        )


        # Converte o DataFrame para matriz NumPy.
        dados = dados.values.astype(float)


        print(
            f"{classe:<10} -> "
            f"{os.path.basename(arquivo):<30} "
            f"shape = {dados.shape}"
        )


        # ---------------------------------------------------------------------
        # Verifica NaN e infinito.
        # ---------------------------------------------------------------------

        if not np.isfinite(dados).all():

            raise ValueError(
                f"O arquivo {arquivo} contém "
                f"NaN ou infinito."
            )


        # Armazena os sinais.
        sinais.append(dados)


        # Cria os respectivos rótulos.
        labels.extend(
            [classe] * dados.shape[0]
        )


    # =========================================================================
    # CONCATENAÇÃO
    # =========================================================================

    # Junta os arquivos verticalmente.
    X = np.vstack(sinais)


    # Converte os rótulos para NumPy.
    y = np.array(labels)


    # =========================================================================
    # INFORMAÇÕES DO DATASET
    # =========================================================================

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
    print("Distribuição das classes:")

    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )


    return X, y


# =============================================================================
# 13. EXTRAÇÃO DAS CARACTERÍSTICAS WAVELET
# =============================================================================

def extrair_caracteristicas_wavelet(sinal):

    """
    Extrai características estatísticas dos coeficientes Wavelet.

    A DWT produz:

        A6
        D6
        D5
        D4
        D3
        D2
        D1

    Para cada componente são calculadas oito características:

        1. Energia
        2. Energia relativa
        3. Média
        4. Desvio padrão
        5. RMS
        6. Máximo absoluto
        7. Mínimo absoluto
        8. Entropia

    Portanto:

        7 componentes × 8 características = 56 características
    """


    # =========================================================================
    # DWT
    # =========================================================================

    coeficientes = pywt.wavedec(
        sinal,
        WAVELET,
        level=NIVEL_WAVELET
    )


    # =========================================================================
    # ENERGIA TOTAL
    # =========================================================================

    # Soma das energias de todos os conjuntos de coeficientes.
    energia_total = sum(
        np.sum(c ** 2)
        for c in coeficientes
    )


    caracteristicas = []
    nomes = []


    # =========================================================================
    # NOMES DOS COMPONENTES
    # =========================================================================

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


    # =========================================================================
    # EXTRAÇÃO
    # =========================================================================

    for nome_nivel, coef in zip(
        nomes_niveis,
        coeficientes
    ):

        coef = np.asarray(coef)


        # ---------------------------------------------------------------------
        # Energia
        #
        # Mede a quantidade de energia presente naquele conjunto
        # de coeficientes.
        # ---------------------------------------------------------------------

        energia = np.sum(
            coef ** 2
        )


        # ---------------------------------------------------------------------
        # Energia relativa
        #
        # Representa a proporção da energia daquele nível em relação
        # à energia total do sinal.
        # ---------------------------------------------------------------------

        if energia_total > 0:

            energia_relativa = (
                energia /
                energia_total
            )

        else:

            energia_relativa = 0.0


        # ---------------------------------------------------------------------
        # Média
        # ---------------------------------------------------------------------

        media = np.mean(coef)


        # ---------------------------------------------------------------------
        # Desvio padrão
        # ---------------------------------------------------------------------

        desvio = np.std(coef)


        # ---------------------------------------------------------------------
        # RMS
        #
        # RMS = raiz da média dos quadrados.
        # ---------------------------------------------------------------------

        rms = np.sqrt(
            np.mean(coef ** 2)
        )


        # ---------------------------------------------------------------------
        # Máximo absoluto
        # ---------------------------------------------------------------------

        maximo = np.max(
            np.abs(coef)
        )


        # ---------------------------------------------------------------------
        # Mínimo absoluto
        # ---------------------------------------------------------------------

        minimo = np.min(
            np.abs(coef)
        )


        # ---------------------------------------------------------------------
        # Entropia de Shannon
        #
        # A energia de cada coeficiente é transformada em uma distribuição
        # de probabilidade.
        # ---------------------------------------------------------------------

        energia_coef = coef ** 2

        soma_energia = np.sum(
            energia_coef
        )


        if soma_energia > 0:

            probabilidade = (
                energia_coef /
                soma_energia
            )


            # Remove valores exatamente iguais a zero para evitar
            # log2(0).
            probabilidade = (
                probabilidade[
                    probabilidade > 0
                ]
            )


            entropia = -np.sum(
                probabilidade *
                np.log2(probabilidade)
            )

        else:

            entropia = 0.0


        # ---------------------------------------------------------------------
        # Junta as oito características.
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
        # Cria nomes para as características.
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
        np.array(caracteristicas),
        nomes
    )


# =============================================================================
# 13B. [NOVO] IDENTIFICADORES DOS SINAIS
# =============================================================================
#
# Este arquivo (diferente do DWT_RF.py) não gera um identificador por sinal
# durante o carregamento. A função abaixo cria esse identificador depois,
# no formato "classe_000", "classe_001", ... apenas para rastrear cada
# sinal nos novos arquivos de saída (não é usado como característica).
#
# =============================================================================

def gerar_identificadores(y):

    contadores = {}
    ids = []

    for classe in y:

        contadores[classe] = contadores.get(classe, 0)

        ids.append(
            f"{classe}_{contadores[classe]:03d}"
        )

        contadores[classe] += 1

    return np.array(ids)


# =============================================================================
# 13C. [NOVO] REPRESENTAÇÃO BRUTA DO SINAL NO DOMÍNIO WAVELET
# =============================================================================
#
# A função "extrair_caracteristicas_wavelet" (acima) calcula ESTATÍSTICAS
# (energia, média, RMS, entropia, etc.) a partir dos coeficientes da DWT.
#
# A função abaixo é diferente: ela NÃO resume nada. Ela devolve os
# PRÓPRIOS coeficientes da DWT, nível a nível — a representação do sinal
# no domínio da transformada wavelet.
#
# =============================================================================

def extrair_representacao_wavelet(sinal):

    coeficientes = pywt.wavedec(
        sinal,
        WAVELET,
        level=NIVEL_WAVELET
    )

    nomes_niveis = [f"A{NIVEL_WAVELET}"]

    for nivel in range(NIVEL_WAVELET, 0, -1):

        nomes_niveis.append(f"D{nivel}")

    representacao = {
        nome_nivel: np.asarray(coef)
        for nome_nivel, coef in zip(nomes_niveis, coeficientes)
    }

    return representacao, nomes_niveis


def construir_matriz_dominio_wavelet(X, y, ids):

    print()
    print("=" * 80)
    print("[NOVO] REPRESENTAÇÃO BRUTA NO DOMÍNIO WAVELET")
    print("=" * 80)
    print()

    inicio = time.perf_counter()

    coeficientes_por_nivel = None
    nomes_niveis_ref = None

    for i, sinal in enumerate(X):

        representacao, nomes_niveis = extrair_representacao_wavelet(sinal)

        if coeficientes_por_nivel is None:

            nomes_niveis_ref = nomes_niveis

            coeficientes_por_nivel = {
                nivel: [] for nivel in nomes_niveis_ref
            }

        for nivel in nomes_niveis_ref:

            coeficientes_por_nivel[nivel].append(
                representacao[nivel]
            )

        if (i + 1) % 100 == 0 or i == len(X) - 1:

            print(f"Processados: {i + 1}/{len(X)}")

    # Um bloco de colunas por nível. Dentro de um mesmo nível, todos os
    # sinais têm o mesmo número de coeficientes (depende só do tamanho do
    # sinal, da wavelet e do nível — nunca do conteúdo do sinal).
    blocos = []
    total_coeficientes = 0

    for nivel in nomes_niveis_ref:

        matriz_nivel = np.array(coeficientes_por_nivel[nivel])

        total_coeficientes += matriz_nivel.shape[1]

        colunas_nivel = [
            f"{nivel}_coef_{indice:04d}"
            for indice in range(matriz_nivel.shape[1])
        ]

        blocos.append(
            pd.DataFrame(matriz_nivel, columns=colunas_nivel)
        )

    df_dominio_wavelet = pd.concat(blocos, axis=1)

    df_dominio_wavelet["classe"] = y
    df_dominio_wavelet["ID_sinal"] = ids

    tempo = time.perf_counter() - inicio

    print()
    print(f"Coeficientes por sinal (todos os níveis): {total_coeficientes}")
    print(f"Tempo de extração: {tempo:.2f} segundos")

    return df_dominio_wavelet


# =============================================================================
# 14. EXTRAÇÃO WAVELET PARA TODO O DATASET
# =============================================================================

def construir_matriz_wavelet(X):

    """
    Aplica a DWT a todos os sinais.

    Esta etapa pode ser realizada antes do K-Fold porque a DWT
    é uma transformação determinística e não aprende parâmetros
    estatísticos a partir do conjunto de dados.
    """

    print()
    print("=" * 80)
    print("ETAPA 1 - TRANSFORMADA WAVELET")
    print("=" * 80)
    print()

    print(
        f"Wavelet utilizada     : {WAVELET}"
    )

    print(
        f"Nível de decomposição : {NIVEL_WAVELET}"
    )


    inicio = time.perf_counter()


    matriz = []
    nomes = None


    # -------------------------------------------------------------------------
    # Processa cada sinal individualmente.
    # -------------------------------------------------------------------------

    for i, sinal in enumerate(X):

        caracteristicas, nomes_temp = (
            extrair_caracteristicas_wavelet(
                sinal
            )
        )


        matriz.append(
            caracteristicas
        )


        # Os nomes das características são iguais para todos os sinais,
        # portanto só precisamos armazená-los uma vez.
        if nomes is None:

            nomes = nomes_temp


        # Mostra o progresso.
        if (
            (i + 1) % 100 == 0
            or
            i == len(X) - 1
        ):

            print(
                f"Processados: "
                f"{i + 1}/{len(X)}"
            )


    # Converte a lista para matriz NumPy.
    X_wavelet = np.array(matriz)


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
# 15. FILTRO DE CORRELAÇÃO
# =============================================================================

def filtro_correlacao(
    X_train,
    nomes_features
):

    """
    Remove características altamente correlacionadas.

    IMPORTANTE:

    A correlação é calculada somente sobre o conjunto de treinamento
    do fold atual.

    Dessa forma, o conjunto de teste não influencia a seleção.
    """


    print()
    print("ETAPA 2 - FILTRO DE CORRELAÇÃO")


    # Converte as características em DataFrame para facilitar
    # o cálculo da matriz de correlação.
    df = pd.DataFrame(
        X_train,
        columns=nomes_features
    )


    # Calcula a correlação de Pearson.
    #
    # abs() considera tanto correlação positiva quanto negativa.
    matriz_corr = df.corr(
        method="pearson"
    ).abs()


    # Mantém somente a parte superior da matriz.
    #
    # Isso evita analisar duas vezes:
    #
    # A-B
    # B-A
    superior = matriz_corr.where(
        np.triu(
            np.ones(
                matriz_corr.shape
            ),
            k=1
        ).astype(bool)
    )


    # Identifica características que possuem correlação
    # maior que o limite definido.
    remover = [

        coluna

        for coluna in superior.columns

        if any(
            superior[coluna]
            >
            LIMITE_CORRELACAO
        )

    ]


    # Características que permanecerão.
    manter = [

        coluna

        for coluna in df.columns

        if coluna not in remover

    ]


    # Converte os nomes preservados para seus respectivos índices.
    indices_mantidos = [

        nomes_features.index(nome)

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
# 16. FISHER DISCRIMINANT RATIO
# =============================================================================

def fisher_score(
    X,
    y
):

    """
    Calcula o Fisher Discriminant Ratio.

    A ideia é comparar:

        dispersão ENTRE classes
        -----------------------
        dispersão DENTRO das classes

    Quanto maior o Fisher Score, maior tende a ser a capacidade
    discriminativa da característica.
    """


    # Obtém as classes presentes no conjunto de treinamento.
    classes = np.unique(y)


    scores = []


    # Média global de cada característica.
    media_global = np.mean(
        X,
        axis=0
    )


    # -------------------------------------------------------------------------
    # Calcula um Fisher Score para cada característica.
    # -------------------------------------------------------------------------

    for j in range(
        X.shape[1]
    ):

        numerador = 0.0
        denominador = 0.0


        # Percorre cada classe.
        for classe in classes:

            valores = X[
                y == classe,
                j
            ]


            if len(valores) == 0:
                continue


            # Média da característica dentro da classe.
            media_classe = np.mean(
                valores
            )


            # Variância dentro da classe.
            variancia_classe = np.var(
                valores
            )


            # Número de amostras da classe.
            n = len(valores)


            # -----------------------------------------------------------------
            # Dispersão entre classes.
            # -----------------------------------------------------------------

            numerador += (

                n *
                (
                    media_classe
                    -
                    media_global[j]
                ) ** 2

            )


            # -----------------------------------------------------------------
            # Dispersão dentro das classes.
            # -----------------------------------------------------------------

            denominador += (

                n *
                variancia_classe

            )


        # Evita divisão por zero.
        if denominador > 0:

            score = (
                numerador /
                denominador
            )

        else:

            score = 0.0


        scores.append(score)


    return np.array(scores)


# =============================================================================
# 17. SELEÇÃO DAS CARACTERÍSTICAS PELO FISHER
# =============================================================================

def selecionar_fisher(
    X_train,
    y_train,
    nomes_features
):

    """
    Seleciona as N_FEATURES_FISHER características com maior Fisher Score.

    Esta etapa é executada separadamente dentro de cada fold.
    """


    print()
    print(
        "ETAPA 3 - FISHER DISCRIMINANT RATIO"
    )


    # Calcula o score de todas as características restantes.
    scores = fisher_score(
        X_train,
        y_train
    )


    # Ordena os índices do maior para o menor score.
    indices_ordenados = np.argsort(
        scores
    )[::-1]


    # Garante que não tentaremos selecionar mais características
    # do que existem disponíveis.
    n_selecionar = min(
        N_FEATURES_FISHER,
        X_train.shape[1]
    )


    # Seleciona as melhores características.
    indices_selecionados = (
        indices_ordenados[
            :n_selecionar
        ]
    )


    # Recupera os nomes.
    nomes_selecionados = [

        nomes_features[i]

        for i in indices_selecionados

    ]


    # Recupera os respectivos scores.
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
        "Características selecionadas:"
    )


    for nome, score in zip(
        nomes_selecionados,
        scores_selecionados
    ):

        print(
            f"{nome:<35} "
            f"Fisher = {score:.6e}"
        )


    return (
        indices_selecionados,
        nomes_selecionados,
        scores_selecionados
    )


# =============================================================================
# 18. BOOTSTRAP
# =============================================================================

def bootstrap_intervalo(
    y_true,
    y_pred,
    metrica,
    n_bootstrap=1000
):

    """
    Calcula intervalo de confiança de 95% utilizando bootstrap.

    Processo:

        1. Reamostra as observações com reposição.
        2. Calcula a métrica.
        3. Repete N_BOOTSTRAP vezes.
        4. Obtém os percentis 2,5% e 97,5%.

    Resultado:

        IC 95%
    """


    rng = np.random.default_rng(
        RANDOM_STATE
    )


    valores = []


    n = len(y_true)


    for _ in range(
        n_bootstrap
    ):

        # Gera uma amostra bootstrap.
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

            # Algumas métricas podem falhar em uma reamostragem
            # extremamente particular.
            continue


    if len(valores) == 0:

        return (
            np.nan,
            np.nan
        )


    # Percentil inferior.
    inferior = np.percentile(
        valores,
        2.5
    )


    # Percentil superior.
    superior = np.percentile(
        valores,
        97.5
    )


    return (
        inferior,
        superior
    )


# =============================================================================
# 19. FUNÇÃO PRINCIPAL
# =============================================================================

def executar_pipeline():

    """
    Executa todo o pipeline:

        Dataset
        ↓
        DWT
        ↓
        Stratified K-Fold
        ↓
        Correlação
        ↓
        Fisher
        ↓
        StandardScaler
        ↓
        SVM
        ↓
        OOF
        ↓
        Métricas
    """


    inicio_total = time.perf_counter()


    # =========================================================================
    # CARREGAMENTO
    # =========================================================================

    X_sinais, y = carregar_dataset()


    # =========================================================================
    # TRANSFORMADA WAVELET
    # =========================================================================

    X_wavelet, nomes_wavelet = (
        construir_matriz_wavelet(
            X_sinais
        )
    )


    # Verificação das características.
    if not np.isfinite(
        X_wavelet
    ).all():

        raise ValueError(
            "As características Wavelet "
            "contêm NaN ou infinito."
        )


    # =========================================================================
    # SALVAMENTO DAS CARACTERÍSTICAS WAVELET
    # =========================================================================

    df_wavelet = pd.DataFrame(
        X_wavelet,
        columns=nomes_wavelet
    )


    # Adiciona a classe apenas para permitir análise posterior.
    df_wavelet["classe"] = y


    arquivo_wavelet = os.path.join(
        PASTA_WAVELET,
        "caracteristicas_wavelet_SVM.csv"
    )


    df_wavelet.to_csv(
        arquivo_wavelet,
        index=False
    )


    # =========================================================================
    # [NOVO] SALVA A REPRESENTAÇÃO BRUTA NO DOMÍNIO WAVELET
    # =========================================================================
    #
    # Diferente do bloco acima (56 características estatísticas), aqui são
    # salvos os coeficientes da DWT propriamente ditos, nível a nível, para
    # cada sinal.
    # =========================================================================

    ids = gerar_identificadores(y)

    df_dominio_wavelet = construir_matriz_dominio_wavelet(
        X_sinais,
        y,
        ids
    )

    arquivo_dominio_wavelet = os.path.join(
        PASTA_DOMINIO_WAVELET,
        "representacao_dominio_wavelet_SVM.csv"
    )

    df_dominio_wavelet.to_csv(
        arquivo_dominio_wavelet,
        index=False
    )

    print()
    print(f"Representação no domínio wavelet salva em: {arquivo_dominio_wavelet}")


    # =========================================================================
    # STRATIFIED K-FOLD
    # =========================================================================

    # StratifiedKFold mantém aproximadamente a mesma proporção de classes
    # em cada fold.
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    # Lista para armazenar métricas individuais de cada fold.
    resultados_folds = []


    # Lista para registrar quais características foram selecionadas
    # em cada fold.
    frequencias_folds = []


    # Classes globais.
    classes = np.unique(y)


    # =========================================================================
    # ESTRUTURAS PARA PREDIÇÕES OOF
    # =========================================================================

    # Cada posição corresponde ao índice original da amostra.
    #
    # Ao final:
    #
    # oof_predicoes[i]
    #
    # contém a predição da amostra i quando ela estava no conjunto de teste.
    oof_predicoes = np.full(
        len(y),
        None,
        dtype=object
    )


    # Matriz para armazenar as probabilidades OOF.
    #
    # Cada linha = uma amostra.
    #
    # Cada coluna = uma classe.
    oof_probabilidades = np.full(
        (
            len(y),
            len(classes)
        ),
        np.nan
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


        inicio_fold = time.perf_counter()


        # =====================================================================
        # SEPARAÇÃO TREINO / TESTE
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
        # FILTRO DE CORRELAÇÃO
        # =====================================================================

        # IMPORTANTE:
        #
        # O filtro é ajustado somente no treinamento.
        indices_corr, nomes_corr = (
            filtro_correlacao(
                X_train,
                nomes_wavelet
            )
        )


        # Aplica ao treinamento.
        X_train_corr = (
            X_train[
                :,
                indices_corr
            ]
        )


        # Aplica ao teste usando exatamente os mesmos índices.
        X_test_corr = (
            X_test[
                :,
                indices_corr
            ]
        )


        # =====================================================================
        # FISHER
        # =====================================================================

        # O Fisher também é calculado somente no treinamento.
        (
            indices_fisher,
            nomes_fisher,
            scores_fisher
        ) = selecionar_fisher(
            X_train_corr,
            y_train,
            nomes_corr
        )


        # Aplica os índices escolhidos ao treino.
        X_train_selected = (
            X_train_corr[
                :,
                indices_fisher
            ]
        )


        # Aplica exatamente os mesmos índices ao teste.
        X_test_selected = (
            X_test_corr[
                :,
                indices_fisher
            ]
        )


        # =====================================================================
        # STANDARD SCALER
        # =====================================================================

        # O scaler aprende:
        #
        # média
        # desvio padrão
        #
        # somente do treinamento.
        scaler = StandardScaler()


        X_train_scaled = (
            scaler.fit_transform(
                X_train_selected
            )
        )


        # O teste é apenas transformado.
        #
        # Nenhum parâmetro é aprendido do teste.
        X_test_scaled = (
            scaler.transform(
                X_test_selected
            )
        )


        # =====================================================================
        # SVM
        # =====================================================================

        print()
        print(
            "ETAPA 4 - SVM RBF"
        )


        print(
            f"Kernel : {SVM_KERNEL}"
        )

        print(
            f"C      : {SVM_C}"
        )

        print(
            f"Gamma  : {SVM_GAMMA}"
        )


        # Criação do classificador.
        modelo = SVC(

            kernel=SVM_KERNEL,

            C=SVM_C,

            gamma=SVM_GAMMA,

            probability=SVM_PROBABILITY,

            random_state=RANDOM_STATE

        )


        # O SVM é treinado exclusivamente com o treinamento do fold.
        modelo.fit(
            X_train_scaled,
            y_train
        )


        # =====================================================================
        # PREDIÇÃO
        # =====================================================================

        # Classe prevista.
        y_pred = modelo.predict(
            X_test_scaled
        )


        # Probabilidade estimada para cada classe.
        y_prob = modelo.predict_proba(
            X_test_scaled
        )


        # Confirma que a ordem das probabilidades corresponde às classes
        # globais.
        if not np.array_equal(
            modelo.classes_,
            classes
        ):

            raise RuntimeError(
                "A ordem das classes do SVM "
                "não corresponde à ordem global."
            )


        # =====================================================================
        # MÉTRICAS DO FOLD
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

        # Converte as classes verdadeiras para representação binária.
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


        # Tempo do fold.
        tempo_fold = (
            time.perf_counter()
            -
            inicio_fold
        )


        # ---------------------------------------------------------------------
        # Exibição
        # ---------------------------------------------------------------------

        print()

        print(
            f"Accuracy        : {accuracy:.4f}"
        )

        print(
            f"Precision Macro : {precision:.4f}"
        )

        print(
            f"Recall Macro    : {recall:.4f}"
        )

        print(
            f"F1 Macro        : {f1:.4f}"
        )

        print(
            f"MCC             : {mcc:.4f}"
        )

        print(
            f"Cohen Kappa     : {kappa:.4f}"
        )

        print(
            f"ROC-AUC Macro   : {roc_auc:.4f}"
        )

        print(
            f"Log Loss        : {logloss:.4f}"
        )

        print(
            f"Tempo           : {tempo_fold:.2f} s"
        )


        # =====================================================================
        # SALVA RESULTADOS DO FOLD
        # =====================================================================

        resultados_folds.append({

            "Fold": fold,

            "Accuracy": accuracy,

            "Precision_macro": precision,

            "Recall_macro": recall,

            "F1_macro": f1,

            "MCC": mcc,

            "Cohen_Kappa": kappa,

            "ROC_AUC_macro": roc_auc,

            "Log_Loss": logloss,

            "Features_Wavelet":
                X_wavelet.shape[1],

            "Features_apos_correlacao":
                X_train_corr.shape[1],

            "Features_Fisher":
                X_train_selected.shape[1],

            "SVM_kernel":
                SVM_KERNEL,

            "SVM_C":
                SVM_C,

            "SVM_gamma":
                SVM_GAMMA,

            "Tempo_s":
                tempo_fold

        })


        # =====================================================================
        # REGISTRA AS CARACTERÍSTICAS SELECIONADAS
        # =====================================================================

        for nome, score in zip(
            nomes_fisher,
            scores_fisher
        ):

            frequencias_folds.append({

                "Fold": fold,

                "Caracteristica": nome,

                "Fisher_Score": score

            })


        # =====================================================================
        # ARMAZENA AS PREDIÇÕES OOF
        # =====================================================================

        for posicao, indice in enumerate(
            test_idx
        ):


            # Cada amostra deve receber uma única predição OOF.
            if oof_predicoes[indice] is not None:

                raise RuntimeError(
                    f"Amostra {indice} recebeu "
                    f"mais de uma predição OOF."
                )


            # Armazena a classe prevista na posição original.
            oof_predicoes[indice] = (
                y_pred[posicao]
            )


            # Armazena as probabilidades na posição original.
            oof_probabilidades[
                indice,
                :
            ] = y_prob[
                posicao,
                :
            ]


    # =========================================================================
    # VERIFICAÇÃO FINAL DAS PREDIÇÕES OOF
    # =========================================================================

    quantidade_oof = np.sum(
        oof_predicoes != None
    )


    print()
    print("=" * 80)
    print("PREDIÇÕES OOF")
    print("=" * 80)


    print(
        f"Total de sinais : {len(y)}"
    )

    print(
        f"Predições OOF   : {quantidade_oof}"
    )


    if quantidade_oof != len(y):

        raise RuntimeError(
            "Existem amostras sem predição OOF."
        )


    # Converte para arrays NumPy.
    y_true_global = np.asarray(y)

    y_pred_global = np.asarray(
        oof_predicoes
    )


    y_prob_global = np.asarray(
        oof_probabilidades
    )


    # Confirma que não existem probabilidades ausentes.
    if np.isnan(
        y_prob_global
    ).any():

        raise RuntimeError(
            "Existem probabilidades OOF ausentes."
        )


    # =========================================================================
    # MÉTRICAS GLOBAIS
    # =========================================================================

    # IMPORTANTE:
    #
    # As métricas finais são calculadas diretamente sobre as predições OOF.
    #
    # Assim, cada amostra é avaliada pela predição produzida quando ela
    # estava no conjunto de teste.
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


    # =========================================================================
    # ROC-AUC GLOBAL
    # =========================================================================

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


    # =========================================================================
    # LOG LOSS GLOBAL
    # =========================================================================

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
        "metricas_por_classe_Wavelet_SVM.csv"
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
    # MATRIZ DE CONFUSÃO - FIGURA
    # =========================================================================

    arquivo_cm = os.path.join(
        PASTA_MATRIZ,
        "matriz_confusao_Wavelet_SVM.png"
    )


    salvar_matriz_confusao_fft_rf(
        cm,
        classes,
        arquivo_cm,
        "Matriz de Confusão - Wavelet & SVM"
    )


    # =========================================================================
    # CURVA ROC
    # =========================================================================

    plt.figure(
        figsize=(8, 7)
    )


    # Uma curva ROC é calculada para cada classe.
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


    # Linha correspondente ao classificador aleatório.
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
        "Curva ROC - Wavelet & SVM"
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
        "curva_ROC_Wavelet_SVM.png"
    )


    plt.savefig(
        arquivo_roc,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =========================================================================
    # CURVA PRECISION-RECALL
    # =========================================================================

    plt.figure(
        figsize=(8, 7)
    )


    # Calcula uma curva PR para cada classe.
    for i, classe in enumerate(
        classes
    ):

        precision_curve, recall_curve, _ = (
            precision_recall_curve(

                y_global_bin[:, i],

                y_prob_global[:, i]

            )
        )


        # Average Precision.
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
        "Curva Precision-Recall - Wavelet & SVM"
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
        "curva_Precision_Recall_Wavelet_SVM.png"
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
        "metricas_por_fold_Wavelet_SVM.csv"
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
        "metricas_finais_Wavelet_SVM.csv"
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
        "intervalos_confianca_bootstrap_Wavelet_SVM.csv"
    )


    df_bootstrap.to_csv(
        arquivo_bootstrap,
        index=False
    )


    # =========================================================================
    # CARACTERÍSTICAS SELECIONADAS
    # =========================================================================

    df_wavelet_selecionadas = pd.DataFrame(
        frequencias_folds
    )


    arquivo_wavelet_selecionadas = os.path.join(
        PASTA_METRICAS,
        "caracteristicas_Wavelet_selecionadas_por_fold_SVM.csv"
    )


    df_wavelet_selecionadas.to_csv(
        arquivo_wavelet_selecionadas,
        index=False
    )


    # =========================================================================
    # TEMPO TOTAL
    # =========================================================================

    tempo_total = (
        time.perf_counter()
        -
        inicio_total
    )


    # =========================================================================
    # RELATÓRIO TXT
    # =========================================================================

    arquivo_relatorio = os.path.join(
        PASTA_RELATORIO,
        "relatorio_Wavelet_SVM.txt"
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
            "RELATÓRIO - WAVELET + SVM\n"
        )

        arquivo.write(
            "=" * 80 + "\n\n"
        )


        arquivo.write(
            "CONFIGURAÇÕES\n"
        )

        arquivo.write(
            f"Dataset: Signal_Feature\n"
        )

        arquivo.write(
            f"Wavelet: {WAVELET}\n"
        )

        arquivo.write(
            f"Nível: {NIVEL_WAVELET}\n"
        )

        arquivo.write(
            f"Limite correlação: "
            f"{LIMITE_CORRELACAO}\n"
        )

        arquivo.write(
            f"Características Fisher: "
            f"{N_FEATURES_FISHER}\n"
        )

        arquivo.write(
            f"SVM Kernel: "
            f"{SVM_KERNEL}\n"
        )

        arquivo.write(
            f"SVM C: "
            f"{SVM_C}\n"
        )

        arquivo.write(
            f"SVM Gamma: "
            f"{SVM_GAMMA}\n"
        )

        arquivo.write(
            f"K-Fold: "
            f"{N_SPLITS}\n"
        )

        arquivo.write(
            f"Bootstrap: "
            f"{N_BOOTSTRAP}\n\n"
        )


        arquivo.write(
            "MÉTRICAS FINAIS\n\n"
        )


        arquivo.write(
            f"Accuracy: "
            f"{accuracy_global:.4f}\n"
        )

        arquivo.write(
            f"Precision Macro: "
            f"{precision_global:.4f}\n"
        )

        arquivo.write(
            f"Recall Macro: "
            f"{recall_global:.4f}\n"
        )

        arquivo.write(
            f"F1 Macro: "
            f"{f1_global:.4f}\n"
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
            f"\nTempo total: "
            f"{tempo_total:.2f} segundos\n"
        )

        arquivo.write(
            f"Tempo total: "
            f"{tempo_total / 60:.2f} minutos\n"
        )


        arquivo.write(
            "\nMATRIZ DE CONFUSÃO\n\n"
        )

        arquivo.write(
            str(cm)
        )


        arquivo.write(
            "\n\nRELATÓRIO POR CLASSE\n\n"
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
    # RESULTADOS FINAIS
    # =========================================================================

    print()
    print("=" * 80)
    print("RESULTADOS FINAIS - WAVELET + SVM")
    print("=" * 80)
    print()


    print(
        f"Accuracy        : "
        f"{accuracy_global:.4f} "
        f"({accuracy_global * 100:.2f}%)"
    )


    print(
        f"Precision Macro : "
        f"{precision_global:.4f} "
        f"({precision_global * 100:.2f}%)"
    )


    print(
        f"Recall Macro    : "
        f"{recall_global:.4f} "
        f"({recall_global * 100:.2f}%)"
    )


    print(
        f"F1 Macro        : "
        f"{f1_global:.4f} "
        f"({f1_global * 100:.2f}%)"
    )


    print(
        f"MCC             : "
        f"{mcc_global:.4f}"
    )


    print(
        f"Cohen's Kappa   : "
        f"{kappa_global:.4f}"
    )


    print(
        f"ROC-AUC Macro   : "
        f"{roc_auc_global:.4f}"
    )


    print(
        f"Log Loss        : "
        f"{logloss_global:.4f}"
    )


    print()

    print(
        "Matriz de confusão:"
    )

    print(cm)


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


    # Retorna os principais resultados para o bloco CodeCarbon.
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
            tempo_total

    }


# =============================================================================
# 20. EXECUÇÃO COM CODECARBON
# =============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("INICIANDO WAVELET + SVM")
    print("=" * 80)


    # -------------------------------------------------------------------------
    # Cria o rastreador de consumo energético.
    #
    # O CodeCarbon acompanha a execução do pipeline e estima a emissão
    # de CO2 equivalente.
    # -------------------------------------------------------------------------

    tracker = EmissionsTracker(

        project_name=
        "Wavelet_SVM_PQD",

        output_dir=
        PASTA_CODECARBON,

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


        tracker.start()


        # Executa todo o pipeline.
        resultados = executar_pipeline()


    finally:

        # Encerra o monitoramento.
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
        # Salva um resumo dos resultados juntamente com a emissão estimada.
        # ---------------------------------------------------------------------

        if resultados is not None:

            resumo_codecarbon = pd.DataFrame({

                "Modelo": [
                    "Wavelet + SVM"
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

                "resumo_codecarbon_Wavelet_SVM.csv"

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
    print("PROGRAMA FINALIZADO")
    print("=" * 80)