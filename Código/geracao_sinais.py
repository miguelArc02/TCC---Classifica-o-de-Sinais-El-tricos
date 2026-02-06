import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parâmetros base do sinal
# =========================
fs = 10000          # taxa de amostragem (Hz)
f = 60              # frequência da rede
t_final = 0.3       # duração (s)
t = np.arange(0, t_final, 1/fs)

V_nominal = 127     # tensão RMS nominal


# =========================
# Funções de geração
# =========================

def seno_puro():
    return V_nominal * np.sqrt(2) * np.sin(2*np.pi*f*t)


def sag(profundidade=0.5, inicio=0.1, fim=0.18):
    v = seno_puro()
    mask = (t >= inicio) & (t <= fim)
    v[mask] *= profundidade
    return v


def swell(aumento=1.5, inicio=0.1, fim=0.18):
    v = seno_puro()
    mask = (t >= inicio) & (t <= fim)
    v[mask] *= aumento
    return v


def interrupcao(inicio=0.1, fim=0.18):
    v = seno_puro()
    mask = (t >= inicio) & (t <= fim)
    v[mask] = 0
    return v


def harmonicas(ordens=[3,5], amplitudes=[0.2, 0.1]):
    v = seno_puro()
    for h, a in zip(ordens, amplitudes):
        v += a * V_nominal * np.sqrt(2) * np.sin(2*np.pi*f*h*t)
    return v


def transitorio(amplitude=3, tempo=0.12, largura=0.002):
    v = seno_puro()
    pulso = amplitude * np.exp(-((t-tempo)**2)/(2*largura**2))
    return v + pulso


def ruido(nivel=0.05):
    v = seno_puro()
    noise = nivel * V_nominal * np.random.randn(len(t))
    return v + noise


# =========================
# Escolher tipo de distúrbio
# =========================

sinais = {
    "Normal": seno_puro(),
    "Sag": sag(),
    "Swell": swell(),
    "Interrupção": interrupcao(),
    "Harmônicas": harmonicas(),
    "Transitório": transitorio(),
    "Ruído": ruido()
}


# =========================
# Plotagem
# =========================

plt.figure(figsize=(12, 8))

for i, (nome, sinal) in enumerate(sinais.items(), 1):
    plt.subplot(len(sinais), 1, i)
    plt.plot(t, sinal)
    plt.title(nome)
    plt.ylabel("V")
    plt.grid()

plt.xlabel("Tempo (s)")
plt.tight_layout()
plt.show()