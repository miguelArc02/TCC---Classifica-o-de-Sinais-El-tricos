# ============================================================
# CLASSIFICADOR DE DISTÚRBIOS DE QUALIDADE DE ENERGIA
# ============================================================
#
# Este programa implementa o seguinte pipeline:
#
#   1. Leitura dos sinais
#   2. Transformada Rápida de Fourier (FFT)
#   3. Remoção de características altamente correlacionadas
#   4. Seleção de características pelo Fisher Discriminant Ratio
#   5. Normalização das características
#   6. Classificação utilizando SVM
#   7. Validação cruzada Stratified K-Fold
#   8. Cálculo das métricas:
#        - Accuracy
#        - Precision
#        - Recall
#        - F1-Score
#   9. Matriz de confusão
#  10. Medição de tempo de execução
#  11. Estimativa de emissão de CO2 com CodeCarbon
#
#
# CLASSES DO DATASET:
#
#   normal -> sinal normal
#   sag    -> afundamento de tensão
#   swell  -> elevação de tensão
#   har    -> harmônicas
#   notch  -> entalhe
#   spike  -> transitório impulsivo
#   capc   -> transitório oscilatório
#
#
# ESTRUTURA ESPERADA DAS PASTAS:
#
# Power_Quality_Disturbance_Dataset-master
# |
# +--- CSV
#      |
#      +--- Signal_Feature
#           |
#           +--- signal_normal.csv
#           +--- signal_sag.csv
#           +--- signal_swell.csv
#           +--- signal_har.csv
#           +--- signal_notch.csv
#           +--- signal_spike.csv
#           +--- signal_capc.csv
#
#
# Cada arquivo deve possuir:
#
#       100 sinais x 2560 amostras
#
# Portanto:
#
#       7 classes x 100 sinais = 700 sinais
#
# ============================================================


# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

# Biblioteca para trabalhar com caminhos e diretórios
import os

# Biblioteca para medir tempo
import time

# Biblioteca para cálculos numéricos
import numpy as np

# Biblioteca para manipulação de tabelas
import pandas as pd

# Biblioteca para gráficos
import matplotlib.pyplot as plt

# Biblioteca para ignorar avisos não críticos
import warnings


# ------------------------------------------------------------
# Scikit-learn
# ------------------------------------------------------------

# Classificador SVM
from sklearn.svm import SVC

# Normalização dos dados
from sklearn.preprocessing import StandardScaler

# Validação cruzada estratificada
from sklearn.model_selection import StratifiedKFold

# Métricas de avaliação
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ------------------------------------------------------------
# CodeCarbon
# ------------------------------------------------------------

from codecarbon import EmissionsTracker


# Ignorar alguns avisos que não interferem no funcionamento
warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURAÇÕES DO USUÁRIO
# ============================================================
#
# Nesta seção estão as principais configurações.
#
# Normalmente você só precisará alterar o DATA_DIR.
#
# ============================================================


# ------------------------------------------------------------
# CAMINHO DA PASTA Signal_Feature
# ------------------------------------------------------------
#
# IMPORTANTE:
#
# Substitua o caminho abaixo pelo local onde o seu dataset
# está armazenado no computador.
#
# Exemplo:
#
# C:\Users\Rafael\Downloads\Power_Quality_Disturbance_Dataset-master\CSV\Signal_Feature
#
# O "r" antes da string evita problemas com as barras "\".
# ------------------------------------------------------------

DATA_DIR = r"C:\Users\Rafael\Mestrado\Projeto de Pesquisa\Trabalho Murilo e Miguel\Power_Quality_Disturbance_Dataset-master\Power_Quality_Disturbance_Dataset-master\CSV\Signal_Feature"


# ------------------------------------------------------------
# ARQUIVOS DO DATASET
# ------------------------------------------------------------
#
# O primeiro elemento é o nome que será utilizado internamente
# para identificar a classe.
#
# O segundo elemento é o nome do arquivo CSV.
# ------------------------------------------------------------

FILES = {

    "normal": "signal_normal.csv",

    "sag": "signal_sag.csv",

    "swell": "signal_swell.csv",

    "har": "signal_har.csv",

    "notch": "signal_notch.csv",

    "spike": "signal_spike.csv",

    "capc": "signal_capc.csv"
}


# ------------------------------------------------------------
# PARÂMETROS DO SINAL
# ------------------------------------------------------------

# Frequência de amostragem:
#
# 15360 amostras por segundo
FS = 15360


# Frequência fundamental da rede:
F0 = 60


# Número de amostras de cada sinal:
N_SAMPLES = 2560


# ------------------------------------------------------------
# PARÂMETROS DA CORRELAÇÃO
# ------------------------------------------------------------

# Se duas características tiverem correlação absoluta maior
# que 0.95, consideramos que existe forte redundância entre
# elas e uma delas será removida.
CORR_THRESHOLD = 0.95


# ------------------------------------------------------------
# PARÂMETROS DO FISHER DISCRIMINANT RATIO
# ------------------------------------------------------------

# Número de características que serão mantidas depois
# do cálculo do Fisher.
#
# O dataset original utiliza 14 características.
#
# Portanto, inicialmente utilizaremos 14 para permitir
# uma comparação futura com o método original.
N_FDR_FEATURES = 14


# ------------------------------------------------------------
# PARÂMETROS DO SVM
# ------------------------------------------------------------

# Tipo de kernel:
#
# "rbf" = Radial Basis Function
# "linear" = SVM linear
#
# Aqui utilizaremos inicialmente RBF.
SVM_KERNEL = "rbf"


# Parâmetro C do SVM
SVM_C = 10


# Gamma do kernel RBF.
#
# "scale" permite que o próprio Scikit-learn determine
# um valor adequado com base nos dados.
SVM_GAMMA = "scale"


# ------------------------------------------------------------
# PARÂMETROS DO K-FOLD
# ------------------------------------------------------------

# Número de divisões
N_SPLITS = 5


# Semente aleatória para permitir reprodução dos resultados
RANDOM_STATE = 42


# ============================================================
# 3. CRIAÇÃO DAS PASTAS DE RESULTADOS
# ============================================================
#
# O programa irá salvar os resultados nesta pasta:
#
# resultados_PQD/
#
# Dentro dela teremos:
#
#   - resultados_por_fold.csv
#   - relatorio_final.txt
#   - matriz_confusao.png
#   - codecarbon/
#
# ============================================================

RESULTS_DIR = "resultados_PQD"

CODECARBON_DIR = os.path.join(
    RESULTS_DIR,
    "codecarbon"
)


# Criar as pastas caso elas ainda não existam
os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    CODECARBON_DIR,
    exist_ok=True
)


# ============================================================
# 4. FUNÇÃO PARA CARREGAR OS DADOS
# ============================================================
#
# Esta função abre os sete arquivos CSV e junta todos os
# sinais em uma única matriz.
#
# Cada arquivo possui:
#
#       100 sinais x 2560 amostras
#
# Portanto, no final teremos:
#
#       700 sinais x 2560 amostras
#
# ============================================================

def load_dataset(data_dir, files):

    print("\n")
    print("=" * 70)
    print("1 - CARREGANDO DATASET")
    print("=" * 70)


    # Listas temporárias
    X_list = []
    y_list = []


    # Percorrer cada classe
    for label, filename in files.items():

        # Construir o caminho completo do arquivo
        filepath = os.path.join(
            data_dir,
            filename
        )


        # Verificar se o arquivo existe
        if not os.path.exists(filepath):

            raise FileNotFoundError(
                "\nArquivo não encontrado:\n"
                + filepath
                + "\n\n"
                "Verifique se o caminho DATA_DIR está correto."
            )


        # Ler o CSV
        #
        # header=None significa que todas as linhas são
        # consideradas dados.
        df = pd.read_csv(
            filepath,
            header=None
        )


        # Converter para matriz NumPy
        data = df.values.astype(float)


        # Mostrar informações
        print(
            f"{label:10s} -> "
            f"shape = {data.shape}"
        )


        # Adicionar os sinais à lista
        X_list.append(data)


        # Criar os rótulos das amostras
        #
        # Se existem 100 sinais no arquivo, serão criados
        # 100 rótulos com o nome da classe.
        y_list.extend(
            [label] * data.shape[0]
        )


    # --------------------------------------------------------
    # Juntar todos os arquivos verticalmente
    # --------------------------------------------------------

    X_time = np.vstack(
        X_list
    )


    # Transformar os rótulos em array NumPy
    y = np.array(
        y_list
    )


    return X_time, y


# ============================================================
# 5. FUNÇÃO PARA CALCULAR A FFT
# ============================================================
#
# A FFT transforma o sinal:
#
#       domínio do tempo
#              ↓
#       domínio da frequência
#
#
# Nosso sinal possui:
#
#       N = 2560 amostras
#       Fs = 15360 Hz
#
# Portanto:
#
#       resolução = Fs/N
#                = 15360/2560
#                = 6 Hz
#
#
# Como os sinais são reais, utilizamos rFFT, que retorna
# somente a metade positiva do espectro.
#
# ============================================================

def extract_fft_features(X, fs):

    # Número de amostras de cada sinal
    N = X.shape[1]


    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------
    #
    # axis=1 significa que a FFT será calculada em cada linha.
    #
    # Cada linha representa um sinal.
    # --------------------------------------------------------

    X_fft_complex = np.fft.rfft(
        X,
        axis=1
    )


    # --------------------------------------------------------
    # Magnitude da FFT
    # --------------------------------------------------------
    #
    # A FFT retorna números complexos.
    #
    # Para classificação, utilizaremos a magnitude:
    #
    #       |X[k]|
    #
    # que representa a amplitude de cada componente espectral.
    # --------------------------------------------------------

    X_fft = np.abs(
        X_fft_complex
    )


    # --------------------------------------------------------
    # Normalização da amplitude da FFT
    # --------------------------------------------------------

    X_fft = (
        2.0 / N
    ) * X_fft


    # A componente DC não deve ser multiplicada por 2
    X_fft[:, 0] = (
        X_fft[:, 0] / 2.0
    )


    # --------------------------------------------------------
    # Eixo de frequência
    # --------------------------------------------------------

    freqs = np.fft.rfftfreq(
        N,
        d=1 / fs
    )


    # --------------------------------------------------------
    # Remover a componente DC
    # --------------------------------------------------------
    #
    # DC corresponde a 0 Hz.
    #
    # Como estamos interessados principalmente no conteúdo
    # oscilatório do sinal, retiramos essa componente.
    # --------------------------------------------------------

    X_fft = X_fft[:, 1:]

    freqs = freqs[1:]


    return X_fft, freqs


# ============================================================
# 6. FILTRO DE CORRELAÇÃO
# ============================================================
#
# Muitas características da FFT podem conter praticamente
# a mesma informação.
#
# Por exemplo:
#
#       característica A
#       característica B
#
# podem apresentar:
#
#       correlação = 0.99
#
# Nesse caso, manter as duas é desnecessário.
#
# Portanto, eliminamos características cuja correlação
# absoluta seja maior que 0.95.
#
# IMPORTANTE:
#
# A correlação é calculada SOMENTE no conjunto de treinamento
# de cada fold.
#
# Isso evita DATA LEAKAGE.
#
# ============================================================

def correlation_filter(
    X_train,
    threshold=0.95
):

    # Calcular matriz de correlação
    #
    # rowvar=False significa que cada coluna representa
    # uma característica.
    corr_matrix = np.corrcoef(
        X_train,
        rowvar=False
    )


    # Número de características
    n_features = corr_matrix.shape[0]


    # Conjunto para armazenar características removidas
    to_remove = set()


    # --------------------------------------------------------
    # Comparar cada característica com as demais
    # --------------------------------------------------------

    for i in range(n_features):

        # Se a característica já foi removida, ignorar
        if i in to_remove:
            continue


        for j in range(
            i + 1,
            n_features
        ):

            # Se já foi removida, ignorar
            if j in to_remove:
                continue


            # Correlação entre as características i e j
            corr = corr_matrix[i, j]


            # Se a correlação for muito alta, removemos j
            if np.abs(corr) > threshold:

                to_remove.add(j)


    # --------------------------------------------------------
    # Índices das características que permaneceram
    # --------------------------------------------------------

    selected_indices = np.array(
        [
            i
            for i in range(n_features)
            if i not in to_remove
        ],
        dtype=int
    )


    return selected_indices


# ============================================================
# 7. FISHER DISCRIMINANT RATIO
# ============================================================
#
# O Fisher Discriminant Ratio (FDR) mede o poder de uma
# característica para separar as diferentes classes.
#
#
# A ideia é:
#
#   boa característica:
#
#       médias das classes muito diferentes
#       +
#       baixa variabilidade dentro das classes
#
#
# Uma formulação multiclasses é:
#
#
#               Σ n_c (μ_c - μ)^2
#       F = ---------------------------
#               Σ n_c σ_c²
#
#
# Quanto maior F:
#
#       maior o poder discriminativo.
#
# ============================================================

def fisher_scores(X, y):

    # Identificar classes existentes
    classes = np.unique(y)


    # Número de características
    n_features = X.shape[1]


    # Média global
    global_mean = np.mean(
        X,
        axis=0
    )


    # Inicializar numerador e denominador
    numerator = np.zeros(
        n_features
    )

    denominator = np.zeros(
        n_features
    )


    # --------------------------------------------------------
    # Calcular contribuição de cada classe
    # --------------------------------------------------------

    for c in classes:

        # Selecionar somente as amostras da classe c
        Xc = X[
            y == c
        ]


        # Número de amostras da classe
        n_c = Xc.shape[0]


        # Média da classe
        mean_c = np.mean(
            Xc,
            axis=0
        )


        # Variância da classe
        var_c = np.var(
            Xc,
            axis=0,
            ddof=1
        )


        # Evitar divisão por zero
        var_c = np.where(
            var_c == 0,
            1e-12,
            var_c
        )


        # Parte entre classes
        numerator += (
            n_c *
            (mean_c - global_mean) ** 2
        )


        # Parte dentro das classes
        denominator += (
            n_c *
            var_c
        )


    # Calcular Fisher
    scores = (
        numerator /
        denominator
    )


    # Remover NaN e infinito
    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    return scores


# ============================================================
# 8. SELEÇÃO DAS CARACTERÍSTICAS PELO FISHER
# ============================================================
#
# Depois de calcular o Fisher para todas as características,
# ordenamos do maior para o menor.
#
# Exemplo:
#
#       Feature 15 -> FDR = 25.3
#       Feature 82 -> FDR = 20.1
#       Feature 34 -> FDR = 18.7
#       ...
#
# Se N_FDR_FEATURES = 14, selecionamos as 14 melhores.
#
# ============================================================

def select_fisher_features(
    X_train,
    y_train,
    n_features=14
):

    # Calcular Fisher para todas as características
    scores = fisher_scores(
        X_train,
        y_train
    )


    # Ordenar do maior para o menor
    ranking = np.argsort(
        scores
    )[::-1]


    # Selecionar somente as N melhores
    n_features = min(
        n_features,
        X_train.shape[1]
    )


    selected_indices = ranking[
        :n_features
    ]


    return (
        selected_indices,
        scores
    )


# ============================================================
# 9. FUNÇÃO PRINCIPAL DO PIPELINE
# ============================================================

def run_pipeline(
    X_time,
    y
):

    print("\n")
    print("=" * 70)
    print("INICIANDO PIPELINE")
    print("=" * 70)


    # ========================================================
    # CODECARBON
    # ========================================================
    #
    # O CodeCarbon estima a quantidade de CO2 equivalente
    # associada ao processamento computacional.
    #
    # A medição começa aqui e inclui:
    #
    #       FFT
    #       Correlação
    #       Fisher
    #       SVM
    #       K-Fold
    #
    # ========================================================

    tracker = EmissionsTracker(

        project_name=
            "PQD_FFT_Correlation_Fisher_SVM",

        measure_power_secs=1,

        save_to_file=True,

        output_dir=CODECARBON_DIR
    )


    # Iniciar cronômetro
    start_time = time.time()


    # Iniciar CodeCarbon
    tracker.start()


    try:

        # ====================================================
        # ETAPA 1 — FFT
        # ====================================================

        print("\n")
        print("=" * 70)
        print("ETAPA 1 — TRANSFORMADA DE FOURIER")
        print("=" * 70)


        X_fft, freqs = extract_fft_features(
            X_time,
            FS
        )


        # Informações
        print(
            f"Número de sinais: "
            f"{X_time.shape[0]}"
        )

        print(
            f"Amostras por sinal: "
            f"{X_time.shape[1]}"
        )

        print(
            f"Características FFT: "
            f"{X_fft.shape[1]}"
        )

        print(
            f"Resolução espectral: "
            f"{FS / N_SAMPLES:.2f} Hz"
        )


        # ====================================================
        # ETAPA 2 — STRATIFIED K-FOLD
        # ====================================================
        #
        # Dividimos os dados em 5 partes.
        #
        # Em cada rodada:
        #
        #       4 partes -> treinamento
        #       1 parte  -> validação
        #
        # "Stratified" significa que a proporção das classes
        # é preservada.
        #
        # ====================================================

        print("\n")
        print("=" * 70)
        print("ETAPA 2 — STRATIFIED K-FOLD")
        print("=" * 70)


        skf = StratifiedKFold(

            n_splits=N_SPLITS,

            shuffle=True,

            random_state=RANDOM_STATE
        )


        # ----------------------------------------------------
        # Listas para guardar resultados
        # ----------------------------------------------------

        y_true_all = []

        y_pred_all = []

        fold_results = []

        selected_features_per_fold = []


        # ====================================================
        # EXECUTAR OS FOLDS
        # ====================================================

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


            print("\n")
            print("-" * 70)
            print(
                f"FOLD {fold}/{N_SPLITS}"
            )
            print("-" * 70)


            # ------------------------------------------------
            # Separar treinamento e validação
            # ------------------------------------------------

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


            # =================================================
            # ETAPA 3 — CORRELAÇÃO
            # =================================================
            #
            # A correlação é calculada somente no treinamento.
            #
            # Isso é fundamental para evitar data leakage.
            #
            # =================================================

            print(
                "\nETAPA 3 — CORRELAÇÃO"
            )


            corr_indices = correlation_filter(

                X_train,

                threshold=CORR_THRESHOLD
            )


            # Aplicar a seleção também na validação.
            #
            # IMPORTANTE:
            # Não recalculamos a correlação no teste.
            X_train_corr = X_train[
                :,
                corr_indices
            ]

            X_test_corr = X_test[
                :,
                corr_indices
            ]


            print(
                "Características antes da correlação: "
                f"{X_train.shape[1]}"
            )

            print(
                "Características depois da correlação: "
                f"{X_train_corr.shape[1]}"
            )


            # =================================================
            # ETAPA 4 — FISHER DISCRIMINANT RATIO
            # =================================================

            print(
                "\nETAPA 4 — FISHER DISCRIMINANT RATIO"
            )


            fdr_indices_local, fdr_scores = (
                select_fisher_features(

                    X_train_corr,

                    y_train,

                    N_FDR_FEATURES
                )
            )


            # ------------------------------------------------
            # Converter os índices locais para índices da FFT
            # original
            # ------------------------------------------------

            fdr_indices_global = (
                corr_indices[
                    fdr_indices_local
                ]
            )


            selected_features_per_fold.append(
                fdr_indices_global
            )


            # Aplicar as características selecionadas
            X_train_selected = (
                X_train_corr[
                    :,
                    fdr_indices_local
                ]
            )

            X_test_selected = (
                X_test_corr[
                    :,
                    fdr_indices_local
                ]
            )


            print(
                "Características selecionadas pelo Fisher: "
                f"{X_train_selected.shape[1]}"
            )


            # ------------------------------------------------
            # Mostrar as frequências correspondentes
            # ------------------------------------------------
            #
            # Cada índice da FFT corresponde a uma frequência.
            # Isso permite interpretar quais regiões espectrais
            # foram consideradas importantes.
            # ------------------------------------------------

            selected_frequencies = (
                freqs[
                    fdr_indices_global
                ]
            )


            print(
                "Frequências selecionadas (Hz):"
            )

            print(
                np.round(
                    selected_frequencies,
                    2
                )
            )


            # =================================================
            # ETAPA 5 — NORMALIZAÇÃO
            # =================================================
            #
            # O SVM funciona melhor quando as características
            # possuem escalas semelhantes.
            #
            # StandardScaler transforma cada característica
            # aproximadamente para:
            #
            #       média = 0
            #       desvio-padrão = 1
            #
            # O scaler é ajustado SOMENTE no treinamento.
            # =================================================

            print(
                "\nETAPA 5 — NORMALIZAÇÃO"
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


            # =================================================
            # ETAPA 6 — SVM
            # =================================================

            print(
                "\nETAPA 6 — TREINAMENTO DO SVM"
            )


            svm = SVC(

                kernel=SVM_KERNEL,

                C=SVM_C,

                gamma=SVM_GAMMA
            )


            # Treinar
            svm.fit(

                X_train_scaled,

                y_train
            )


            # =================================================
            # ETAPA 7 — CLASSIFICAÇÃO
            # =================================================

            print(
                "Classificando conjunto de validação..."
            )


            y_pred = svm.predict(
                X_test_scaled
            )


            # =================================================
            # ETAPA 8 — MÉTRICAS
            # =================================================

            print(
                "\nETAPA 8 — MÉTRICAS"
            )


            # Accuracy
            accuracy = accuracy_score(
                y_test,
                y_pred
            )


            # Precision Macro
            precision = precision_score(

                y_test,

                y_pred,

                average="macro"
            )


            # Recall Macro
            recall = recall_score(

                y_test,

                y_pred,

                average="macro"
            )


            # F1 Macro
            f1 = f1_score(

                y_test,

                y_pred,

                average="macro"
            )


            # Mostrar
            print(
                f"Accuracy : {accuracy:.4f}"
            )

            print(
                f"Precision: {precision:.4f}"
            )

            print(
                f"Recall   : {recall:.4f}"
            )

            print(
                f"F1-Score : {f1:.4f}"
            )


            # ------------------------------------------------
            # Salvar resultados deste fold
            # ------------------------------------------------

            fold_results.append({

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

                "Features_antes_correlacao":
                    X_train.shape[1],

                "Features_apos_correlacao":
                    X_train_corr.shape[1],

                "Features_FDR":
                    X_train_selected.shape[1]
            })


            # Guardar previsões
            y_true_all.extend(
                y_test
            )

            y_pred_all.extend(
                y_pred
            )


        # ====================================================
        # FIM DO K-FOLD
        # ====================================================

        # Converter para arrays
        y_true_all = np.array(
            y_true_all
        )

        y_pred_all = np.array(
            y_pred_all
        )


        # ====================================================
        # MÉTRICAS GLOBAIS
        # ====================================================

        print("\n")
        print("=" * 70)
        print("RESULTADOS GLOBAIS")
        print("=" * 70)


        accuracy_global = accuracy_score(

            y_true_all,

            y_pred_all
        )


        precision_global = precision_score(

            y_true_all,

            y_pred_all,

            average="macro"
        )


        recall_global = recall_score(

            y_true_all,

            y_pred_all,

            average="macro"
        )


        f1_global = f1_score(

            y_true_all,

            y_pred_all,

            average="macro"
        )


        # ====================================================
        # RESULTADOS POR FOLD
        # ====================================================

        results_df = pd.DataFrame(
            fold_results
        )


        print("\n")
        print("=" * 70)
        print("MÉTRICAS POR FOLD")
        print("=" * 70)


        print(
            results_df.to_string(
                index=False
            )
        )


        # ====================================================
        # MÉDIA E DESVIO-PADRÃO
        # ====================================================

        print("\n")
        print("=" * 70)
        print("MÉDIA ± DESVIO-PADRÃO")
        print("=" * 70)


        for metric in [

            "Accuracy",

            "Precision_macro",

            "Recall_macro",

            "F1_macro"
        ]:


            mean = results_df[
                metric
            ].mean()


            std = results_df[
                metric
            ].std()


            print(

                f"{metric:20s}: "
                f"{mean:.4f} ± {std:.4f}"
            )


        # ====================================================
        # TEMPO TOTAL
        # ====================================================

        total_time = (
            time.time()
            -
            start_time
        )


        # ====================================================
        # FINALIZAR CODECARBON
        # ====================================================
        #
        # Aqui paramos a medição.
        # ====================================================

        emissions = tracker.stop()


        # ====================================================
        # RESULTADO FINAL
        # ====================================================

        print("\n")
        print("=" * 70)
        print("RESULTADO FINAL DO CLASSIFICADOR")
        print("=" * 70)


        print(
            f"\nAccuracy global        : "
            f"{accuracy_global:.4f} "
            f"({accuracy_global * 100:.2f}%)"
        )


        print(
            f"Precision Macro global : "
            f"{precision_global:.4f}"
        )


        print(
            f"Recall Macro global    : "
            f"{recall_global:.4f}"
        )


        print(
            f"F1-Score Macro global  : "
            f"{f1_global:.4f}"
        )


        print(
            f"\nTempo total            : "
            f"{total_time:.2f} segundos"
        )


        print(
            f"Tempo total            : "
            f"{total_time / 60:.2f} minutos"
        )


        print(
            f"Emissão de CO2         : "
            f"{emissions:.8f} kg"
        )


        print(
            f"Emissão de CO2         : "
            f"{emissions * 1000:.4f} g"
        )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        classes = np.unique(
            y
        )


        report = classification_report(

            y_true_all,

            y_pred_all,

            labels=classes,

            target_names=classes,

            digits=4
        )


        print("\n")
        print("=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)


        print(
            report
        )


        # ====================================================
        # MATRIZ DE CONFUSÃO
        # ====================================================

        cm = confusion_matrix(

            y_true_all,

            y_pred_all,

            labels=classes
        )


        plt.figure(
            figsize=(9, 7)
        )


        plt.imshow(
            cm
        )


        plt.title(
            "Matriz de Confusão\n"
            "FFT + Correlação + Fisher + SVM"
        )


        plt.colorbar()


        plt.xticks(

            range(len(classes)),

            classes,

            rotation=45
        )


        plt.yticks(

            range(len(classes)),

            classes
        )


        plt.xlabel(
            "Classe Predita"
        )


        plt.ylabel(
            "Classe Real"
        )


        # Escrever os valores dentro da matriz
        for i in range(
            len(classes)
        ):

            for j in range(
                len(classes)
            ):

                plt.text(

                    j,

                    i,

                    cm[i, j],

                    ha="center",

                    va="center"
                )


        plt.tight_layout()


        # Salvar figura
        confusion_path = os.path.join(

            RESULTS_DIR,

            "matriz_confusao.png"
        )


        plt.savefig(

            confusion_path,

            dpi=300,

            bbox_inches="tight"
        )


        plt.show()


        # ====================================================
        # SALVAR RESULTADOS
        # ====================================================

        # ----------------------------------------------------
        # Resultados de cada fold
        # ----------------------------------------------------

        fold_path = os.path.join(

            RESULTS_DIR,

            "resultados_por_fold.csv"
        )


        results_df.to_csv(

            fold_path,

            index=False
        )


        # ----------------------------------------------------
        # Relatório TXT
        # ----------------------------------------------------

        report_path = os.path.join(

            RESULTS_DIR,

            "relatorio_final.txt"
        )


        with open(

            report_path,

            "w",

            encoding="utf-8"
        ) as f:


            f.write(
                "CLASSIFICADOR DE DISTÚRBIOS "
                "DE QUALIDADE DE ENERGIA\n"
            )


            f.write(
                "=" * 70
                + "\n\n"
            )


            f.write(
                "PIPELINE\n"
            )


            f.write(
                "FFT -> Correlação -> "
                "Fisher FDR -> StandardScaler -> "
                "SVM -> Stratified K-Fold\n\n"
            )


            f.write(
                "CONFIGURAÇÕES\n"
            )


            f.write(
                f"Frequência de amostragem: "
                f"{FS} Hz\n"
            )


            f.write(
                f"Frequência fundamental: "
                f"{F0} Hz\n"
            )


            f.write(
                f"Amostras por sinal: "
                f"{N_SAMPLES}\n"
            )


            f.write(
                f"Threshold correlação: "
                f"{CORR_THRESHOLD}\n"
            )


            f.write(
                f"Número de características FDR: "
                f"{N_FDR_FEATURES}\n"
            )


            f.write(
                f"SVM kernel: "
                f"{SVM_KERNEL}\n"
            )


            f.write(
                f"SVM C: "
                f"{SVM_C}\n"
            )


            f.write(
                f"K-Fold: "
                f"{N_SPLITS}\n\n"
            )


            f.write(
                "RESULTADOS GLOBAIS\n"
            )


            f.write(
                f"Accuracy: "
                f"{accuracy_global:.6f}\n"
            )


            f.write(
                f"Precision Macro: "
                f"{precision_global:.6f}\n"
            )


            f.write(
                f"Recall Macro: "
                f"{recall_global:.6f}\n"
            )


            f.write(
                f"F1-Score Macro: "
                f"{f1_global:.6f}\n"
            )


            f.write(
                f"Tempo: "
                f"{total_time:.6f} s\n"
            )


            f.write(
                f"Tempo: "
                f"{total_time / 60:.6f} min\n"
            )


            f.write(
                f"CO2: "
                f"{emissions:.10f} kg\n"
            )


            f.write(
                f"CO2: "
                f"{emissions * 1000:.6f} g\n\n"
            )


            f.write(
                "CLASSIFICATION REPORT\n"
            )


            f.write(
                "=" * 70
                + "\n"
            )


            f.write(
                report
            )


        # ====================================================
        # RETORNAR RESULTADOS
        # ====================================================

        return {

            "accuracy":
                accuracy_global,

            "precision_macro":
                precision_global,

            "recall_macro":
                recall_global,

            "f1_macro":
                f1_global,

            "time_seconds":
                total_time,

            "co2_kg":
                emissions,

            "co2_g":
                emissions * 1000,

            "fold_results":
                results_df,

            "confusion_matrix":
                cm,

            "frequencies":
                freqs,

            "selected_features_per_fold":
                selected_features_per_fold
        }


    # ========================================================
    # TRATAMENTO DE ERROS
    # ========================================================
    #
    # Se ocorrer algum erro durante o treinamento, precisamos
    # parar o CodeCarbon para não deixar uma instância aberta.
    # ========================================================

    except Exception:

        tracker.stop()

        raise


# ============================================================
# 10. PROGRAMA PRINCIPAL
# ============================================================
#
# Esta parte é executada quando você roda:
#
#       python FFT_SVM_PQD.py
#
# ============================================================

if __name__ == "__main__":


    print("\n")
    print("=" * 70)
    print("CLASSIFICADOR DE DISTÚRBIOS DE QUALIDADE DE ENERGIA")
    print("=" * 70)


    # ========================================================
    # CARREGAR OS DADOS
    # ========================================================

    X_time, y = load_dataset(

        DATA_DIR,

        FILES
    )


    # ========================================================
    # VERIFICAR DATASET
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INFORMAÇÕES DO DATASET")
    print("=" * 70)


    print(
        f"Número total de sinais: "
        f"{X_time.shape[0]}"
    )


    print(
        f"Número de amostras por sinal: "
        f"{X_time.shape[1]}"
    )


    print(
        f"Número de classes: "
        f"{len(np.unique(y))}"
    )


    # --------------------------------------------------------
    # Distribuição das classes
    # --------------------------------------------------------

    print("\nDistribuição das classes:")


    class_distribution = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )


    print(
        class_distribution
    )


    # ========================================================
    # VERIFICAÇÃO BÁSICA
    # ========================================================

    # Verificar se cada sinal possui 2560 amostras

    if X_time.shape[1] != N_SAMPLES:

        raise ValueError(

            "Quantidade inesperada de amostras por sinal.\n"

            f"Esperado: {N_SAMPLES}\n"

            f"Encontrado: {X_time.shape[1]}"
        )


    # Verificar quantidade de classes

    if len(np.unique(y)) != 7:

        raise ValueError(

            "Quantidade inesperada de classes.\n"

            f"Esperado: 7\n"

            f"Encontrado: {len(np.unique(y))}"
        )


    # ========================================================
    # EXECUTAR O PIPELINE
    # ========================================================

    results = run_pipeline(

        X_time,

        y
    )


    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PROGRAMA FINALIZADO")
    print("=" * 70)


    print(
        "\nOs resultados foram salvos em:"
    )


    print(
        os.path.abspath(
            RESULTS_DIR
        )
    )


    print("\n")
    print(
        "Principais arquivos:"
    )


    print(
        " - resultados_por_fold.csv"
    )


    print(
        " - relatorio_final.txt"
    )


    print(
        " - matriz_confusao.png"
    )


    print(
        " - codecarbon/"
    )