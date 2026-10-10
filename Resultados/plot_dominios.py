"""
Gera a representação (plot) dos sinais no domínio de Fourier e no domínio
Wavelet — um GRÁFICO INDIVIDUAL por tipo de distúrbio (classe) — a partir
dos CSVs de "representação no domínio da transformada" já gerados pelo
pipeline (representacao_dominio_fourier*.csv e
representacao_dominio_wavelet*.csv).

Como usar:
    - Ajuste ARQUIVO_FFT e ARQUIVO_DWT para o CSV do teste que quiser usar.
    - Ajuste NIVEL_WAVELET_ESCOLHIDO se quiser plotar outro nível (A6, D6..D1).
    - Ajuste SUFIXO_ID se quiser outro sinal de exemplo por classe
      (por padrão usa o primeiro sinal de cada classe, ID terminado em "_000").
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

ARQUIVO_FFT = "representacao_dominio_fourier_SVM.csv"
ARQUIVO_DWT = "representacao_dominio_wavelet_SVM.csv"

NIVEL_WAVELET_ESCOLHIDO = None  # None = escolhe automaticamente pelo F-ANOVA
SUFIXO_ID = "_000"              # qual sinal de cada classe usar como exemplo

PASTA_SAIDA_FOURIER = "Domínio Fourier"
PASTA_SAIDA_WAVELET = "Domínio Wavelet"

CLASSES_ORDEM = ["normal", "sag", "swell", "notch", "spike", "har", "capc"]
NOMES_BONITOS = {
    "normal": "Normal",
    "sag":    "Sag (afundamento)",
    "swell":  "Swell (elevação)",
    "notch":  "Notch (corte)",
    "spike":  "Spike (pico)",
    "har":    "Harmônicos",
    "capc":   "Chaveamento de capacitor",
}

FREQ_MAX_PLOT = 3000  # Hz — ajuste conforme a energia do seu dataset


# =============================================================================
# ESCOLHA DO NÍVEL WAVELET (se não foi fixado manualmente)
# =============================================================================
#
# Critério: para cada nível, calcula-se a energia (soma dos quadrados dos
# coeficientes) de cada sinal, e mede-se o quão bem essa energia separa as
# classes via ANOVA de um fator (estatística F). O nível com maior F é o
# que mais discrimina os tipos de distúrbio nesse dataset.
#
# =============================================================================

def escolher_nivel_wavelet(df_dwt):

    niveis = ["A6", "D6", "D5", "D4", "D3", "D2", "D1"]

    resultados = []

    for nivel in niveis:

        cols = [c for c in df_dwt.columns if c.startswith(nivel + "_coef_")]

        energia = (df_dwt[cols].values ** 2).sum(axis=1)

        grupos = [
            energia[df_dwt["classe"] == c]
            for c in sorted(df_dwt["classe"].unique())
        ]

        f, p = stats.f_oneway(*grupos)

        resultados.append((nivel, f, p))

    resultados.sort(key=lambda x: -x[1])

    print("Separabilidade por nível (ANOVA F-ratio da energia):")
    for nivel, f, p in resultados:
        print(f"  {nivel}: F={f:.1f}  p={p:.2e}")

    return resultados[0][0]


# =============================================================================
# CARREGAMENTO
# =============================================================================

df_fft = pd.read_csv(ARQUIVO_FFT)
df_dwt = pd.read_csv(ARQUIVO_DWT)

if NIVEL_WAVELET_ESCOLHIDO is None:
    NIVEL_WAVELET_ESCOLHIDO = escolher_nivel_wavelet(df_dwt)
    print(f"\nNível escolhido automaticamente: {NIVEL_WAVELET_ESCOLHIDO}\n")

freq_cols = [c for c in df_fft.columns if c.startswith("FFT_")]
freqs = np.array([float(c.replace("FFT_", "").replace("Hz", "")) for c in freq_cols])

coef_cols = [c for c in df_dwt.columns if c.startswith(NIVEL_WAVELET_ESCOLHIDO + "_coef_")]
indices = np.arange(len(coef_cols))

os.makedirs(PASTA_SAIDA_FOURIER, exist_ok=True)
os.makedirs(PASTA_SAIDA_WAVELET, exist_ok=True)


# =============================================================================
# UM GRÁFICO POR CLASSE — DOMÍNIO DE FOURIER
# =============================================================================

for classe in CLASSES_ORDEM:

    linha = df_fft[
        (df_fft["classe"] == classe)
        & (df_fft["ID_sinal"].str.endswith(SUFIXO_ID))
    ].iloc[0]

    vals = linha[freq_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(freqs, vals, linewidth=1.0, color="#1f6feb")
    ax.set_xlim(0, FREQ_MAX_PLOT)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Domínio de Fourier (FFT) — {NOMES_BONITOS.get(classe, classe)}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_SAIDA_FOURIER, f"fourier_{classe}.png"), dpi=150)
    plt.close(fig)


# =============================================================================
# UM GRÁFICO POR CLASSE — DOMÍNIO WAVELET (nível escolhido)
# =============================================================================

for classe in CLASSES_ORDEM:

    linha = df_dwt[
        (df_dwt["classe"] == classe)
        & (df_dwt["ID_sinal"].str.endswith(SUFIXO_ID))
    ].iloc[0]

    vals = linha[coef_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(indices, vals, linewidth=1.0, color="#c0392b")
    ax.set_xlabel(f"Índice do coeficiente (nível {NIVEL_WAVELET_ESCOLHIDO})")
    ax.set_ylabel("Valor do coeficiente")
    ax.set_title(f"Domínio Wavelet (nível {NIVEL_WAVELET_ESCOLHIDO}) — {NOMES_BONITOS.get(classe, classe)}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA_SAIDA_WAVELET, f"wavelet_{NIVEL_WAVELET_ESCOLHIDO}_{classe}.png"), dpi=150)
    plt.close(fig)

print("Gráficos individuais gerados com sucesso.")