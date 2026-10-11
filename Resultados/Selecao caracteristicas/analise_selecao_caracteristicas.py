"""
analise_selecao_caracteristicas.py

Analise da selecao de caracteristicas (subtopico de resultados):

  Wavelet (db4, nivel 6, 56 caracteristicas)
    - reproduz o filtro de correlacao (0,95) e o Fisher multiclasse do pipeline;
    - Fisher um-contra-todos por classe (quais caracteristicas mais distinguem
      cada disturbio);
    - estabilidade da selecao nos 5 folds;
    - perfil medio de cada classe nas 14 caracteristicas selecionadas.

  FFT (1281 componentes de 0 a 7680 Hz)
    - estabilidade das frequencias selecionadas nos 5 folds;
    - (se o arquivo representacao_dominio_fourier_SVM.csv estiver disponivel)
      Fisher um-contra-todos por classe, espectro medio por classe e
      frequencias selecionadas.

IMPORTANTE: a analise por classe e DESCRITIVA, calculada com todos os sinais,
para interpretar o que os modelos enxergam. A avaliacao dos modelos continua
usando a selecao feita somente no treino de cada fold.

Uso:
    python analise_selecao_caracteristicas.py --entrada <pasta com os CSVs> \
                                              --saida Selecao_Caracteristicas
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LIMITE_CORRELACAO = 0.95
N_FEATURES_FISHER = 14
TOP_N = 5
FS = 15360

ARQ_MATRIZ_WAVELET = "matriz_56_features_completa.csv"
ARQ_SEL_WAVELET = "caracteristicas_Wavelet_selecionadas_por_fold_SVM.csv"
ARQ_SEL_FFT = "frequencias_selecionadas_por_fold.csv"
ARQ_FFT_COMPLETO = "representacao_dominio_fourier_SVM.csv"

# Faixa de frequencia coberta por cada subbanda (db4, nivel 6, fs = 15360 Hz).
FAIXAS_WAVELET = {
    "D1": (FS / 4, FS / 2),
    "D2": (FS / 8, FS / 4),
    "D3": (FS / 16, FS / 8),
    "D4": (FS / 32, FS / 16),
    "D5": (FS / 64, FS / 32),
    "D6": (FS / 128, FS / 64),
    "A6": (0, FS / 128),
}


# ---------------------------------------------------------------------------
# Funcoes do pipeline (reproduzidas) e Fisher um-contra-todos
# ---------------------------------------------------------------------------
def filtro_correlacao(df):
    """Mesma regra do pipeline: mantem a primeira de cada grupo correlacionado."""
    corr = df.corr(method="pearson").abs().to_numpy()
    superior = np.triu(corr, k=1)
    remover = (superior > LIMITE_CORRELACAO).any(axis=0)
    return [c for c, rem in zip(df.columns, remover) if not rem]


def fisher_multiclasse(X, y):
    """Mesmo calculo do pipeline: dispersao entre classes / dentro das classes."""
    media_global = X.mean(axis=0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(axis=0) - media_global) ** 2
        den += len(Xc) * Xc.var(axis=0)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def fisher_um_contra_todos(X, y, classe):
    """Fisher de duas classes: a classe contra todas as demais."""
    a, b = X[y == classe], X[y != classe]
    num = (a.mean(axis=0) - b.mean(axis=0)) ** 2
    den = a.var(axis=0) + b.var(axis=0)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def tabela_por_classe(X, y, nomes, classes):
    """DataFrame (caracteristicas x classes) com o Fisher um-contra-todos."""
    dados = {c: fisher_um_contra_todos(X, y, c) for c in classes}
    return pd.DataFrame(dados, index=nomes)


def top_por_classe(tab, n):
    linhas = []
    for c in tab.columns:
        for pos, (nome, valor) in enumerate(
            tab[c].sort_values(ascending=False).head(n).items(), start=1
        ):
            linhas.append(
                {"Classe": c, "Posicao": pos, "Caracteristica": nome,
                 "Fisher_um_contra_todos": valor}
            )
    return pd.DataFrame(linhas)


def heatmap_fisher(ax, matriz, linhas, colunas, titulo):
    """
    Mapa de calor do Fisher um-contra-todos. A COR e normalizada pelo maximo de
    cada classe (linha), para que classes com Fisher baixo tambem fiquem
    legiveis; os numeros nas celulas sao os valores reais.
    """
    maximos = matriz.max(axis=1, keepdims=True)
    relativo = np.divide(matriz, maximos, out=np.zeros_like(matriz), where=maximos > 0)
    im = ax.imshow(relativo, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(colunas)))
    ax.set_xticklabels(colunas, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(linhas)))
    ax.set_yticklabels(linhas, fontsize=10)
    ax.set_title(titulo, fontsize=12)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = matriz[i, j]
            texto = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
            ax.text(j, i, texto, ha="center", va="center", fontsize=7,
                    color="black" if relativo[i, j] > 0.5 else "white")
    return im


# ---------------------------------------------------------------------------
# Wavelet
# ---------------------------------------------------------------------------
def analisar_wavelet(entrada, saida):
    caminho = entrada / ARQ_MATRIZ_WAVELET
    if not caminho.exists():
        print(f"[Wavelet] ignorado: {caminho} nao encontrado.")
        return

    df = pd.read_csv(caminho)
    y = df["classe"].to_numpy()
    X_df = df.drop(columns=["classe", "id"], errors="ignore")
    classes = sorted(np.unique(y))

    mantidas = filtro_correlacao(X_df)
    Xk = X_df[mantidas]
    print(f"[Wavelet] {X_df.shape[1]} caracteristicas -> {len(mantidas)} apos filtro de correlacao")

    # Fisher multiclasse (o mesmo usado na selecao do pipeline).
    scores = fisher_multiclasse(Xk.to_numpy(), y)
    ranking = pd.Series(scores, index=mantidas).sort_values(ascending=False)
    top14 = list(ranking.head(N_FEATURES_FISHER).index)
    ranking.rename("Fisher_multiclasse").to_csv(
        saida / "wavelet_ranking_fisher_multiclasse.csv", encoding="utf-8-sig"
    )

    # Fisher um-contra-todos por classe.
    tab = tabela_por_classe(Xk.to_numpy(), y, mantidas, classes)
    tab.to_csv(saida / "wavelet_fisher_um_contra_todos_completo.csv",
               encoding="utf-8-sig")
    top_por_classe(tab, TOP_N).to_csv(
        saida / "wavelet_top_caracteristicas_por_classe.csv",
        index=False, encoding="utf-8-sig",
    )

    # Figura: Fisher um-contra-todos (uniao das top-N de cada classe).
    uniao = []
    for c in classes:
        for nome in tab[c].sort_values(ascending=False).head(TOP_N).index:
            if nome not in uniao:
                uniao.append(nome)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(uniao), 5.2))
    im = heatmap_fisher(ax, tab.loc[uniao].T.to_numpy(), classes, uniao,
                        f"Wavelet: Fisher um-contra-todos (as {TOP_N} melhores de cada classe)")
    fig.colorbar(im, ax=ax, label="Fisher relativo ao máximo da classe")
    plt.tight_layout()
    plt.savefig(saida / "wavelet_fisher_por_classe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figura: perfil medio (z-score) de cada classe nas 14 selecionadas.
    z = (Xk[top14] - Xk[top14].mean()) / Xk[top14].std(ddof=0)
    perfil = z.groupby(y).mean().loc[classes]
    perfil.T.to_csv(saida / "wavelet_perfil_medio_z_14_caracteristicas.csv",
                    encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    lim = np.abs(perfil.to_numpy()).max()
    im = ax.imshow(perfil.to_numpy(), aspect="auto", cmap="coolwarm",
                   vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(top14)))
    ax.set_xticklabels(top14, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(perfil.shape[0]):
        for j in range(perfil.shape[1]):
            ax.text(j, i, f"{perfil.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Wavelet: perfil médio de cada classe nas 14 características selecionadas (z-score)")
    fig.colorbar(im, ax=ax, label="média do z-score")
    plt.tight_layout()
    plt.savefig(saida / "wavelet_perfil_classes_14_caracteristicas.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Estabilidade da selecao nos folds (CSV dos pipelines).
    arq = entrada / ARQ_SEL_WAVELET
    if arq.exists():
        sel = pd.read_csv(arq)
        n_folds = sel["Fold"].nunique()
        est = (sel.groupby("Caracteristica")
                  .agg(Folds_selecionada=("Fold", "nunique"),
                       Fisher_medio=("Fisher_Score", "mean"),
                       Fisher_desvio=("Fisher_Score", "std"))
                  .sort_values(["Folds_selecionada", "Fisher_medio"], ascending=False))
        est.to_csv(saida / "wavelet_estabilidade_selecao_folds.csv", encoding="utf-8-sig")
        print(f"[Wavelet] {len(est)} caracteristicas distintas nos {n_folds} folds; "
              f"{(est['Folds_selecionada'] == n_folds).sum()} em todos")

        fig, ax = plt.subplots(figsize=(9, 5.2))
        ordem = est.sort_values("Fisher_medio")
        ax.barh(ordem.index, ordem["Fisher_medio"],
                xerr=ordem["Fisher_desvio"].fillna(0), color="#1f6feb", alpha=0.85)
        ax.set_xlabel("Fisher multiclasse médio nos 5 folds (± desvio)")
        ax.set_title("Wavelet: características selecionadas pelo Fisher nos folds")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(saida / "wavelet_ranking_selecionadas_folds.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Conferencia com a selecao exploratoria salva pelo pipeline, se existir.
    arq14 = entrada / "matriz_14_features_exploratoria.csv"
    if arq14.exists():
        cols14 = [c for c in pd.read_csv(arq14, nrows=1).columns
                  if c not in ("classe", "id")]
        igual = set(cols14) == set(top14)
        print(f"[Wavelet] top-14 reproduzido == selecao exploratoria do pipeline: {igual}")


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------
def frequencias_dos_folds(entrada):
    arq = entrada / ARQ_SEL_FFT
    if not arq.exists():
        return None
    df = pd.read_csv(arq, encoding="utf-8-sig")
    por_fold = {
        int(r["Fold"]): [int(float(f)) for f in str(r["Frequencias_Hz"]).split(",")]
        for _, r in df.iterrows()
    }
    return por_fold


def analisar_fft(entrada, saida):
    por_fold = frequencias_dos_folds(entrada)
    if por_fold is None:
        print(f"[FFT] ignorado: {entrada / ARQ_SEL_FFT} nao encontrado.")
        return

    # Estabilidade das frequencias selecionadas.
    contagem = pd.Series([f for fs_ in por_fold.values() for f in fs_]).value_counts()
    est = contagem.rename("Folds_selecionada").sort_index().rename_axis("Frequencia_Hz").reset_index()
    est["Multiplo_de_60Hz"] = est["Frequencia_Hz"] % 60 == 0
    est["Ordem_harmonica"] = np.where(est["Multiplo_de_60Hz"], est["Frequencia_Hz"] // 60, np.nan)
    est.to_csv(saida / "fft_estabilidade_frequencias_folds.csv", index=False,
               encoding="utf-8-sig")
    n_folds = len(por_fold)
    print(f"[FFT] {len(est)} frequencias distintas; "
          f"{(est['Folds_selecionada'] == n_folds).sum()} em todos os {n_folds} folds")

    fig, ax = plt.subplots(figsize=(11, 4.6))
    cores = ["#c0392b" if n == n_folds else "#7f8c8d" for n in est["Folds_selecionada"]]
    ax.bar(est["Frequencia_Hz"].astype(str), est["Folds_selecionada"], color=cores)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel(f"Nº de folds em que foi selecionada (de {n_folds})")
    ax.set_title("FFT: frequências selecionadas pelo Fisher nos folds "
                 "(em vermelho, as escolhidas em todos)")
    ax.set_yticks(range(0, n_folds + 1))
    plt.xticks(rotation=70, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(saida / "fft_estabilidade_frequencias.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Analise por classe: precisa do espectro completo salvo pelo pipeline.
    caminho = entrada / ARQ_FFT_COMPLETO
    if not caminho.exists():
        print(f"[FFT] analise por classe pulada: {caminho} nao encontrado "
              f"(arquivo gerado pelo FFT_SVM_v3 com o espectro completo dos sinais).")
        return

    df = pd.read_csv(caminho)
    y = df["classe"].to_numpy()
    X_df = df.drop(columns=["classe", "ID_sinal"], errors="ignore")
    freqs = np.array([float(c.replace("FFT_", "").replace("Hz", "")) for c in X_df.columns])
    classes = sorted(np.unique(y))

    mantidas = filtro_correlacao(X_df)
    idx = [X_df.columns.get_loc(c) for c in mantidas]
    Xk = X_df[mantidas].to_numpy()
    fk = freqs[idx]
    print(f"[FFT] {X_df.shape[1]} componentes -> {len(mantidas)} apos filtro de correlacao")

    scores = fisher_multiclasse(Xk, y)
    ordem = np.argsort(scores)[::-1][:N_FEATURES_FISHER]
    top14_hz = np.sort(fk[ordem])
    pd.DataFrame({"Frequencia_Hz": fk, "Fisher_multiclasse": scores}).sort_values(
        "Fisher_multiclasse", ascending=False
    ).to_csv(saida / "fft_ranking_fisher_multiclasse.csv", index=False, encoding="utf-8-sig")

    tab = pd.DataFrame({c: fisher_um_contra_todos(Xk, y, c) for c in classes},
                       index=[f"{f:.0f} Hz" for f in fk])
    tab.to_csv(saida / "fft_fisher_um_contra_todos_completo.csv", encoding="utf-8-sig")
    top_por_classe(tab, TOP_N).to_csv(saida / "fft_top_frequencias_por_classe.csv",
                                      index=False, encoding="utf-8-sig")

    uniao = []
    for c in classes:
        for nome in tab[c].sort_values(ascending=False).head(TOP_N).index:
            if nome not in uniao:
                uniao.append(nome)
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(uniao), 5.2))
    im = heatmap_fisher(ax, tab.loc[uniao].T.to_numpy(), classes, uniao,
                        f"FFT: Fisher um-contra-todos (as {TOP_N} melhores frequências de cada classe)")
    fig.colorbar(im, ax=ax, label="Fisher relativo ao máximo da classe")
    plt.tight_layout()
    plt.savefig(saida / "fft_fisher_por_classe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Espectro medio por classe + frequencias selecionadas.
    fig, axes = plt.subplots(len(classes), 1, figsize=(11, 2.0 * len(classes)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), classes):
        media = X_df[y == c].to_numpy().mean(axis=0)
        ax.plot(freqs, media, linewidth=1.0)
        for f in top14_hz:
            ax.axvline(f, color="#c0392b", alpha=0.35, linewidth=0.8)
        ax.set_ylabel(c)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[-1].set_xlabel("Frequência (Hz)")
    np.atleast_1d(axes)[0].set_title(
        "FFT: espectro médio de cada classe (linhas vermelhas: 14 frequências selecionadas)")
    plt.tight_layout()
    plt.savefig(saida / "fft_espectro_medio_por_classe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=".", help="pasta com os CSVs de entrada")
    ap.add_argument("--saida", default="Selecao_Caracteristicas")
    args = ap.parse_args()

    entrada, saida = Path(args.entrada), Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    analisar_wavelet(entrada, saida)
    analisar_fft(entrada, saida)
    print(f"\nResultados salvos em: {saida.resolve()}")


if __name__ == "__main__":
    main()