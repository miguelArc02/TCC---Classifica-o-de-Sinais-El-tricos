"""
curva_ROC_combinada.py - grafico unico com a curva ROC macro-average das 4
combinacoes (FFT/Wavelet x SVM/Random Forest).

Coloque este arquivo na pasta "Codigos" (junto do roc_padrao.py), rode antes os
4 pipelines atualizados e depois execute:

    python curva_ROC_combinada.py

Ele le os .npz salvos em "Resultados_ROC_Comparacao" e usa o MESMO calculo de
AUC macro da tabela de resultados (roc_auc_score, ovr, macro).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score

from roc_padrao import PASTA_COMUM, caminho_dados, curva_macro, preparar

# Ordem e nomes devem ser iguais aos usados em salvar_roc_macro(nome=...).
COMBINACOES = [
    "FFT + SVM",
    "FFT + Random Forest",
    "Wavelet + SVM",
    "Wavelet + Random Forest",
]

SAIDA = PASTA_COMUM / "curva_ROC_combinada.png"

# Estilos diferentes para que curvas sobrepostas continuem visiveis.
ESTILOS = [
    {"color": "#c0392b", "linestyle": "-", "linewidth": 5.0},
    {"color": "#1f6feb", "linestyle": "--", "linewidth": 3.5},
    {"color": "#8e44ad", "linestyle": "-", "linewidth": 2.5},
    {"color": "#2e8b57", "linestyle": ":", "linewidth": 3.0},
]


def main():
    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(15, 6.5))

    print(f"{'Combinacao':<26}{'AUC macro (tabela)':>20}{'AUC da curva':>16}")

    plotadas = 0
    for nome, estilo in zip(COMBINACOES, ESTILOS):
        caminho = caminho_dados(nome)
        if not caminho.exists():
            print(f"AVISO: '{nome}' ignorada, arquivo nao encontrado: {caminho}")
            continue

        dados = np.load(caminho, allow_pickle=False)
        y_bin, y_prob, _ = preparar(
            dados["y_true"], dados["y_prob"], dados["classes"]
        )

        auc_macro = roc_auc_score(
            y_bin, y_prob, average="macro", multi_class="ovr"
        )
        xs, ys = curva_macro(y_bin, y_prob)

        print(f"{nome:<26}{auc_macro:>20.6f}{auc(xs, ys):>16.6f}")

        for eixo in (ax, ax_zoom):
            eixo.plot(
                xs, ys, label=f"{nome} (AUC = {auc_macro:.4f})", **estilo
            )
        plotadas += 1

    if plotadas == 0:
        raise SystemExit("Nenhum .npz encontrado. Rode os pipelines primeiro.")

    for eixo in (ax, ax_zoom):
        eixo.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        eixo.set_xlabel("Taxa de Falsos Positivos")
        eixo.set_ylabel("Taxa de Verdadeiros Positivos")
        eixo.grid(alpha=0.3)

    ax.set_title("Curva ROC macro-average — comparação entre combinações")
    ax.legend(loc="lower right", fontsize=9)

    ax_zoom.set_xlim(-0.005, 0.25)
    ax_zoom.set_ylim(0.75, 1.01)
    ax_zoom.set_title("Zoom na região superior esquerda")

    plt.tight_layout()
    plt.savefig(SAIDA, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigura salva em: {SAIDA}")


if __name__ == "__main__":
    main()
