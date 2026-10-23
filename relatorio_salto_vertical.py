import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Análise cinemática 2D de um salto vertical

    **Ponto de interesse:** ponto médio entre os quadris esquerdo e direito.

    **Objetivo:** obter a posição, a velocidade e a aceleração do ponto a
    partir do vídeo, documentando coleta, digitalização, processamento e
    limitações da medida.

    > Resultado importante: o vídeo não contém uma régua, marcador de
    > calibração ou outra dimensão física. Por isso, a análise
    > primária é apresentada em pixels; a versão em metros abaixo é uma
    > calibração dinâmica, não uma escala medida no cenário.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter

    return Path, mo, np, pd, plt, savgol_filter


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Coleta e qualidade do vídeo

    Foram utilizados dois vídeos frontais do salto (um em gravação normal e outro em timelapse).

    O arquivo normal tem 80 quadros, com taxa nominal de 30 Hz, e foi escolhido para a análise
    temporal. O arquivo em câmera lenta foi usado apenas para inspeção
    visual: sua duração codificada é maior e não foi adotado para derivar
    velocidade/aceleração.

    **Avaliação da coleta atual.** A câmera permaneceu fixa, o corpo ficou
    inteiro no campo visual e o contraste com o fundo permite localizar os
    quadris. As limitações são ausência de marcador físico contrastante,
    ausência de escala conhecida e perspectiva por uma câmera posicionada
    fora do plano estritamente perpendicular ao movimento.

    **Para uma coleta futura:** câmera fixa e perpendicular ao plano
    sagital/frontal escolhido, lente aproximadamente na altura do quadril,
    120 Hz ou mais, boa iluminação, marcador circular de alto contraste
    sobre a espinha ilíaca e uma régua/haste de dimensão conhecida no mesmo
    plano do marcador antes da tentativa. Registrar uma tentativa por vez e
    manter todo o corpo no enquadramento.
    """)
    return


@app.cell
def _(Path):
    ROOT = Path(__file__).resolve().parent
    CSV_RAW = ROOT / "dados" / "trajetoria_quadril_bruta.csv"
    FIG_QC = ROOT / "figuras" / "controle_digitalizacao.png"
    FPS = 30.0  # taxa nominal do arquivo Salto_camera-normal.mp4
    G = 9.80665
    # Intervalo que apresentou trajetória aproximadamente parabólica.
    FLIGHT_FIRST_FRAME = 27
    FLIGHT_LAST_FRAME = 42
    return CSV_RAW, FIG_QC, FLIGHT_FIRST_FRAME, FLIGHT_LAST_FRAME, FPS, G


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Digitalização

    Em cada quadro, foram obtidas as coordenadas dos quadris esquerdo e
    direito por detector de pose baseado nos 17 pontos corporais COCO. O
    ponto analisado foi definido como

    \[
    (x_h,y_h)=\frac{1}{2}\left[(x_{QE},y_{QE})+(x_{QD},y_{QD})\right].
    \]

    A marcação em vermelho abaixo é o ponto médio; os círculos amarelos
    são os dois quadris. A inspeção cobre a postura inicial, agachamento,
    início da fase aérea, ápice, descida e amortecimento. A tabela CSV
    contém todos os 80 quadros, inclusive as coordenadas brutas.
    """)
    return


@app.cell
def _(FIG_QC, plt):
    image_qc = plt.imread(FIG_QC)
    fig_qc, ax_qc = plt.subplots(figsize=(8, 10))
    ax_qc.imshow(image_qc)
    ax_qc.set_axis_off()
    fig_qc
    return


@app.cell
def _(CSV_RAW, FPS, pd):
    raw = pd.read_csv(CSV_RAW).sort_values("frame").reset_index(drop=True)
    time_s = raw["frame"].to_numpy() / FPS
    x_px = raw["x_hip_px"].to_numpy()
    y_px = raw["y_hip_px"].to_numpy()

    # Referencial 2D do laboratório no plano da imagem:
    # origem = ponto do primeiro quadro; x positivo para a direita;
    # y positivo para cima (logo, inverte-se o eixo vertical da imagem).
    x_rel_px = x_px - x_px[0]
    y_rel_px = y_px[0] - y_px
    raw
    return raw, time_s, x_rel_px, y_rel_px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Processamento cinemático

    A trajetória medida é discreta, com \(f_s=30\) Hz. Para reduzir o
    ruído que seria amplificado ao diferenciar a posição, foi aplicado um
    filtro de Savitzky–Golay de ordem 3 e janela de 11 amostras (0,367 s):

    \[
    \mathbf{v}(t)=\frac{d\mathbf{r}}{dt},\qquad
    \mathbf{a}(t)=\frac{d^2\mathbf{r}}{dt^2}.
    \]

    Isso segue a cautela sobre amostragem, ruído e diferenciação discutida
    em **Signal Basic Properties**. A representação vetorial e a leitura
    de velocidade/aceleração seguem **Path Frame**. **Polar Basis** foi
    consultado para a escolha de referencial; como o salto não gira em
    torno de um centro fixo, o referencial cartesiano 2D é mais adequado do
    que coordenadas polares para apresentar os resultados.

    Referências utilizadas:

    - [PathFrame.py](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks_marimo/PathFrame.py)
    - [SignalBasicProperties.py](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks_marimo/SignalBasicProperties.py)
    - [PolarBasis.py](https://molab.marimo.io/github/BMClab/BMC/blob/master/notebooks_marimo/PolarBasis.py)
    """)
    return


@app.cell
def _(
    FLIGHT_FIRST_FRAME,
    FLIGHT_LAST_FRAME,
    FPS,
    G,
    np,
    raw,
    savgol_filter,
    time_s,
    x_rel_px,
    y_rel_px,
):
    # Ajuste da fase aérea: y(t) = c2*t² + c1*t + c0.
    in_flight = raw["frame"].between(FLIGHT_FIRST_FRAME, FLIGHT_LAST_FRAME).to_numpy()
    parabola = np.polyfit(time_s[in_flight], y_rel_px[in_flight], deg=2)
    y_parabola_px = np.polyval(parabola, time_s)
    ay_fit_px_s2 = 2 * parabola[0]
    rmse_px = float(np.sqrt(np.mean((y_rel_px[in_flight] - y_parabola_px[in_flight]) ** 2)))
    r2 = float(
        1
        - np.sum((y_rel_px[in_flight] - y_parabola_px[in_flight]) ** 2)
        / np.sum((y_rel_px[in_flight] - y_rel_px[in_flight].mean()) ** 2)
    )

    # Como não há escala no cenário, esta é apenas uma calibração dinâmica
    # OPCIONAL: impõe |a_y|=g na fase aérea. Não a usar como validação
    # independente de aceleração em SI.
    dynamic_scale_m_per_px = G / abs(ay_fit_px_s2)

    window, order = 11, 3
    x_m = savgol_filter(x_rel_px, window, order) * dynamic_scale_m_per_px
    y_m = savgol_filter(y_rel_px, window, order) * dynamic_scale_m_per_px
    vx_m_s = (
        savgol_filter(x_rel_px, window, order, deriv=1, delta=1 / FPS)
        * dynamic_scale_m_per_px
    )
    vy_m_s = (
        savgol_filter(y_rel_px, window, order, deriv=1, delta=1 / FPS)
        * dynamic_scale_m_per_px
    )
    ax_m_s2 = (
        savgol_filter(x_rel_px, window, order, deriv=2, delta=1 / FPS)
        * dynamic_scale_m_per_px
    )
    ay_m_s2 = (
        savgol_filter(y_rel_px, window, order, deriv=2, delta=1 / FPS)
        * dynamic_scale_m_per_px
    )
    return (
        ax_m_s2,
        ay_fit_px_s2,
        ay_m_s2,
        dynamic_scale_m_per_px,
        parabola,
        r2,
        rmse_px,
        vx_m_s,
        vy_m_s,
        x_m,
        y_m,
        y_parabola_px,
    )


@app.cell
def _(
    FLIGHT_FIRST_FRAME,
    FLIGHT_LAST_FRAME,
    FPS,
    G,
    ay_fit_px_s2,
    dynamic_scale_m_per_px,
    np,
    parabola,
    r2,
    rmse_px,
    time_s,
    x_m,
    y_m,
):
    apex_time_s = float(-parabola[1] / (2 * parabola[0]))
    apex_y_m = float(np.polyval(parabola, apex_time_s) * dynamic_scale_m_per_px)
    crouch_idx = int(np.argmin(y_m))
    apex_idx = int(np.argmax(y_m))
    summary = {
        "quadros": len(time_s),
        "fps": FPS,
        "flight_interval": f"{FLIGHT_FIRST_FRAME}–{FLIGHT_LAST_FRAME}",
        "ay_fit_px_s2": ay_fit_px_s2,
        "r2": r2,
        "rmse_px": rmse_px,
        "scale_mm_per_px": 1000 * dynamic_scale_m_per_px,
        "apex_t_s": apex_time_s,
        "apex_y_m": apex_y_m,
        "crouch_t_s": float(time_s[crouch_idx]),
        "crouch_y_m": float(y_m[crouch_idx]),
        "rise_crouch_to_apex_m": float(y_m[apex_idx] - y_m[crouch_idx]),
        "horizontal_excursion_m": float(x_m.max() - x_m.min()),
        "g": G,
    }
    return (summary,)


@app.cell(hide_code=True)
def _(mo, summary):
    mo.md(rf"""
    ## 4. Resultados

    - A trajetória na janela de quadros {summary['flight_interval']} é
      fortemente parabólica: \(R^2={summary['r2']:.4f}\) e
      RMSE = {summary['rmse_px']:.2f} px.
    - Resultado diretamente observado: \(a_y\) do ajuste na fase aérea =
      **{summary['ay_fit_px_s2']:.0f} px/s²**; o ápice ocorre em
      **{summary['apex_t_s']:.2f} s**.
    - A calibração dinâmica opcional produz
      **{summary['scale_mm_per_px']:.3f} mm/px** e pico vertical de
      **{summary['apex_y_m']:.3f} m** acima da posição inicial.
    - O agachamento mínimo é **{summary['crouch_y_m']:.3f} m** e a
      excursão quadril–ápice é **{summary['rise_crouch_to_apex_m']:.3f} m**.
    - A excursão horizontal estimada é de apenas
      **{summary['horizontal_excursion_m']:.3f} m**, coerente com um salto
      predominantemente vertical.

    A escala em metros foi obtida impondo \(|a_y|=g\) ao ajuste parabólico.
    Portanto, ela é útil para leitura aproximada dos gráficos, mas não
    transforma esta gravação em uma medida independente de \(g\). Uma
    régua no plano do movimento deve substituir essa etapa em uma outra coleta.
    """)
    return


@app.cell
def _(
    FLIGHT_FIRST_FRAME,
    FLIGHT_LAST_FRAME,
    FPS,
    G,
    ax_m_s2,
    ay_m_s2,
    plt,
    time_s,
    vx_m_s,
    vy_m_s,
    y_parabola_px,
    y_rel_px,
):
    fig_kin, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    axes[0].plot(time_s, y_rel_px, "o", ms=3.5, color="#9aa5b1", label="posição bruta")
    axes[0].plot(time_s, y_parabola_px, "--", lw=2, color="#e76f51", label="ajuste parabólico")
    axes[0].set_ylabel("posição vertical (px; + para cima)")
    axes[0].set_title("Posição, velocidade e aceleração do ponto médio dos quadris")
    axes[0].legend(loc="upper left")

    axes[1].plot(time_s, vx_m_s, lw=2, label="$v_x$")
    axes[1].plot(time_s, vy_m_s, lw=2, label="$v_y$")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].set_ylabel("velocidade (m/s)*")
    axes[1].legend()

    axes[2].plot(time_s, ax_m_s2, lw=2, label="$a_x$")
    axes[2].plot(time_s, ay_m_s2, lw=2, label="$a_y$")
    axes[2].axhline(-G, color="black", ls="--", lw=1, label="$-g$")
    axes[2].axhline(0, color="black", lw=.7)
    axes[2].set_ylabel("aceleração (m/s²)*")
    axes[2].set_xlabel("tempo (s)")
    axes[2].legend(ncols=3, fontsize=8)

    for axis in axes:
        axis.axvspan(FLIGHT_FIRST_FRAME / FPS, FLIGHT_LAST_FRAME / FPS, color="#f4a261", alpha=.14)
        axis.grid(alpha=.25)
    fig_kin.text(.5, -.01, "* Escala métrica por calibração dinâmica; ver limitações.", ha="center")
    fig_kin
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Limitações e conclusão

    A trajetória horizontal pequena e o excelente ajuste parabólico da
    fase aérea indicam que a digitalização representa adequadamente o
    salto no plano da câmera. Entretanto, os picos de aceleração na fase
    de impulso e de aterrissagem devem ser interpretados com cautela: a
    frequência de 30 Hz e a diferenciação tornam esses trechos sensíveis ao
    filtro e à resolução espacial.

    Assim, o notebook fornece a trajetória, velocidade e aceleração
    solicitadas, além dos CSVs reproduzíveis. Para uma análise biomecânica
    final em SI, a próxima coleta deve acrescentar marcador físico, escala
    conhecida e taxa de quadros maior.
    """)
    return


if __name__ == "__main__":
    app.run()
