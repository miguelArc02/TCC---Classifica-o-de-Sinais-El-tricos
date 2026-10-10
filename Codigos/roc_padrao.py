"""
roc_padrao.py - curva ROC padronizada (macro-average) para os 4 pipelines.

Coloque este arquivo na pasta "Codigos", no mesmo nivel das pastas
"1 - FFT_&_SVM", "2 - FFT_&_RF", "3 - DWT_&_SVM" e "4 - DWT_&_RF".

Cada pipeline chama salvar_roc_macro(...) no lugar do seu codigo antigo de ROC.
A funcao:
  1) calcula o AUC macro com roc_auc_score(..., average="macro",
     multi_class="ovr"), o MESMO calculo usado na tabela de resultados;
  2) salva a figura individual no formato padrao (uma curva por classe +
     curva macro-average);
  3) salva y_true, y_prob e classes em um .npz na pasta comum
     "Resultados_ROC_Comparacao", que o curva_ROC_combinada.py usa para
     montar o grafico unico das 4 combinacoes.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

PASTA_COMUM = Path(__file__).resolve().parent / "Resultados_ROC_Comparacao"


def nome_arquivo(nome):
    """'FFT + Random Forest' -> 'FFT_Random_Forest'."""
    return re.sub(r"[^0-9A-Za-z]+", "_", nome).strip("_")


def caminho_dados(nome, pasta=PASTA_COMUM):
    return Path(pasta) / f"roc_dados_{nome_arquivo(nome)}.npz"


def _limites_na_grade(x, y, grade):
    """
    Valores da curva ROC (x = FPR, y = TPR) nas posicoes da grade, tomados pela
    esquerda e pela direita. A ROC tem trechos verticais (varios pontos com o
    mesmo FPR); sem esse cuidado a interpolacao superestima a area.
    """
    direita = np.interp(grade, x, y)
    idx = np.minimum(np.searchsorted(x, grade, side="left"), len(x) - 1)
    em_vertice = x[idx] == grade
    esquerda = np.where(em_vertice, y[idx], direita)
    return esquerda, direita


def curva_macro(y_bin, y_prob):
    """
    Curva ROC macro-average (One-vs-Rest): media das curvas por classe.
    A area dessa curva e exatamente a media dos AUCs por classe, ou seja,
    igual a roc_auc_score(..., average="macro", multi_class="ovr").
    """
    curvas = []
    for i in range(y_bin.shape[1]):
        # drop_intermediate=False mantem todos os pontos da curva.
        fpr, tpr, _ = roc_curve(
            y_bin[:, i], y_prob[:, i], drop_intermediate=False
        )
        curvas.append((fpr, tpr))

    grade = np.unique(np.concatenate([f for f, _ in curvas]))

    esq = np.zeros_like(grade)
    dir_ = np.zeros_like(grade)
    for fpr, tpr in curvas:
        e, d = _limites_na_grade(fpr, tpr, grade)
        esq += e / len(curvas)
        dir_ += d / len(curvas)

    # Polilinha: em cada FPR da grade, chega pela esquerda e sobe ate a direita.
    xs = np.repeat(grade, 2)
    ys = np.column_stack([esq, dir_]).ravel()
    return xs, ys


def preparar(y_true, y_prob, classes):
    """Valida os dados e devolve (y_bin, y_prob, classes_str)."""
    classes = [str(c) for c in classes]
    y_true = np.asarray(y_true).astype(str)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_prob.shape != (len(y_true), len(classes)):
        raise ValueError(
            f"y_prob tem formato {y_prob.shape}; "
            f"esperado {(len(y_true), len(classes))} "
            "(colunas na mesma ordem de 'classes')."
        )
    desconhecidas = set(y_true) - set(classes)
    if desconhecidas:
        raise ValueError(f"Rotulos fora de 'classes': {sorted(desconhecidas)}")
    if not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-3):
        print("AVISO: as linhas de y_prob nao somam 1 (sao probabilidades?).")

    y_bin = label_binarize(y_true, classes=classes)
    return y_bin, y_prob, classes


def salvar_roc_macro(
    y_true,
    y_prob,
    classes,
    nome,
    caminho_png,
    salvar_dados=True,
):
    """
    Gera a figura ROC padrao do pipeline `nome` (ex.: "FFT + SVM") em
    `caminho_png` e salva os dados para o grafico combinado.
    Devolve o caminho da figura e imprime o AUC macro para conferir com a tabela.
    """
    y_bin, y_prob, classes = preparar(y_true, y_prob, classes)

    auc_macro = roc_auc_score(
        y_bin, y_prob, average="macro", multi_class="ovr"
    )
    xs, ys = curva_macro(y_bin, y_prob)

    fig, ax = plt.subplots(figsize=(8, 7))
    paleta = plt.get_cmap("tab10")

    for i, classe in enumerate(classes):
        fpr, tpr, _ = roc_curve(
            y_bin[:, i], y_prob[:, i], drop_intermediate=False
        )
        ax.plot(
            fpr,
            tpr,
            color=paleta(i % 10),
            linewidth=1.6,
            alpha=0.85,
            label=f"{classe} (AUC = {auc(fpr, tpr):.3f})",
        )

    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=3,
        linestyle="--",
        label=f"Macro-average (AUC = {auc_macro:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.2)

    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
    ax.set_title(f"Curva ROC macro-average — {nome}", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(caminho_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if salvar_dados:
        PASTA_COMUM.mkdir(parents=True, exist_ok=True)
        np.savez(
            caminho_dados(nome),
            y_true=np.asarray(y_true).astype(str),
            y_prob=y_prob,
            classes=np.array(classes),
            auc_macro=auc_macro,
            nome=nome,
        )

    print(f"ROC macro-average ({nome}): AUC = {auc_macro:.6f}")
    return caminho_png
