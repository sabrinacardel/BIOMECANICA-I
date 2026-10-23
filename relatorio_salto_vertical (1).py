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
    from io import StringIO

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter

    return StringIO, mo, np, pd, plt, savgol_filter


@app.cell(hide_code=True)
def _():
    # Dados brutos incorporados:
    RAW_TRAJECTORY_CSV = r"""frame,x_left_px,y_left_px,score_left,x_right_px,y_right_px,score_right,x_hip_px,y_hip_px,person_score
0,262.97845458984375,576.6099243164062,1.0,202.73727416992188,578.0465087890625,1.0,232.8578643798828,577.3282470703125,0.9998155236244202
1,263.59173583984375,576.5865478515625,1.0,203.250244140625,580.8976440429688,1.0,233.42098999023438,578.7420654296875,0.9998169541358948
2,262.36859130859375,579.8690795898438,1.0,203.5505828857422,578.429931640625,1.0,232.9595947265625,579.1495361328125,0.999816358089447
3,262.7527160644531,580.5098266601562,1.0,202.3390655517578,581.9475708007812,1.0,232.5458984375,581.2286987304688,0.9998286962509155
4,262.60858154296875,576.7583618164062,1.0,202.14675903320312,581.0751342773438,1.0,232.37767028808594,578.916748046875,0.9998231530189514
5,263.5489501953125,576.9058837890625,1.0,203.13607788085938,576.9058837890625,1.0,233.34251403808594,576.9058837890625,0.9998323917388916
6,261.55841064453125,576.6539306640625,1.0,202.76199340820312,576.6539306640625,1.0,232.1602020263672,576.6539306640625,0.9998359680175781
7,263.21295166015625,577.1615600585938,1.0,202.8397979736328,578.6000366210938,1.0,233.0263671875,577.8807983398438,0.9998538494110107
8,262.3766784667969,577.8836669921875,1.0,203.7217559814453,577.8836669921875,1.0,233.04922485351562,577.8836669921875,0.9998598098754883
9,264.57305908203125,580.7214965820312,1.0,202.97903442382812,580.7214965820312,1.0,233.7760467529297,580.7214965820312,0.9998164772987366
10,263.161376953125,582.9733276367188,1.0,202.97364807128906,581.5335083007812,1.0,233.0675048828125,582.25341796875,0.9999535083770752
11,264.7725524902344,584.4725952148438,1.0,203.17015075683594,583.0331420898438,1.0,233.97134399414062,583.7528686523438,0.9998247027397156
12,262.9297790527344,590.1968994140625,1.0,201.127685546875,588.7574462890625,1.0,232.0287322998047,589.4771728515625,0.9999382495880127
13,261.996337890625,592.4927978515625,1.0,201.58331298828125,592.4927978515625,1.0,231.78982543945312,592.4927978515625,0.9999469518661499
14,262.35369873046875,603.114501953125,1.0,200.71389770507812,603.114501953125,1.0,231.53379821777344,603.114501953125,0.9999322891235352
15,260.1956481933594,614.4650268554688,1.0,200.0664520263672,613.0298461914062,1.0,230.13104248046875,613.7474365234375,0.9999151229858398
16,260.91485595703125,626.1849365234375,1.0,199.06179809570312,623.3060913085938,1.0,229.9883270263672,624.7454833984375,0.9999096393585205
17,260.0002136230469,632.141357421875,1.0,198.2034149169922,630.702880859375,1.0,229.101806640625,631.422119140625,0.9999041557312012
18,259.0824890136719,642.1445922851562,1.0,195.7227325439453,639.2748413085938,1.0,227.40261840820312,640.709716796875,0.9999021291732788
19,260.0455322265625,644.4178466796875,1.0,195.30545043945312,642.9832763671875,1.0,227.6754913330078,643.7005615234375,0.9998846054077148
20,258.2593688964844,646.1522216796875,1.0,196.53411865234375,644.71728515625,1.0,227.39674377441406,645.4347534179688,0.9999052286148071
21,257.7890625,646.1779174804688,1.0,196.0059051513672,643.3026123046875,1.0,226.89749145507812,644.740234375,0.9999270439147949
22,257.0335998535156,641.0151977539062,1.0,195.27651977539062,639.5755004882812,1.0,226.15505981445312,640.2953491210938,0.999935507774353
23,259.0854187011719,634.034912109375,1.0,194.55038452148438,632.595703125,1.0,226.81790161132812,633.3153076171875,0.999943733215332
24,257.826904296875,628.4820556640625,1.0,194.50257873535156,622.7229614257812,1.0,226.16473388671875,625.6025390625,0.9999414682388306
25,257.36639404296875,618.0553588867188,1.0,194.31967163085938,612.2993774414062,1.0,225.84303283691406,615.1773681640625,0.9999428987503052
26,255.99072265625,605.7655639648438,1.0,195.6393585205078,600.0136108398438,1.0,225.81503295898438,602.8895874023438,0.9997573494911194
27,255.46096801757812,589.8651123046875,1.0,196.6405792236328,582.6790161132812,1.0,226.05078125,586.2720947265625,0.9998338222503662
28,257.35748291015625,565.3576049804688,1.0,197.1254119873047,561.0405883789062,1.0,227.241455078125,563.1990966796875,0.9998385906219482
29,256.5755615234375,545.3714599609375,1.0,197.74530029296875,542.4915771484375,1.0,227.16043090820312,543.9315185546875,0.9998294115066528
30,257.9112548828125,528.7484741210938,1.0,200.49893188476562,522.9877319335938,1.0,229.20509338378906,525.8681030273438,0.9998515844345093
31,258.8939514160156,516.3616943359375,1.0,198.68206787109375,510.6008605957031,1.0,228.7880096435547,513.4812622070312,0.9998642206192017
32,259.406494140625,502.99072265625,1.0,197.8687286376953,497.23443603515625,1.0,228.63760375976562,500.1125793457031,0.9998019337654114
33,259.31317138671875,497.7368469238281,1.0,200.36395263671875,491.9809265136719,1.0,229.83856201171875,494.85888671875,0.9998167157173157
34,260.6334228515625,486.3401794433594,1.0,198.92013549804688,486.3401794433594,1.0,229.7767791748047,486.3401794433594,0.9998416900634766
35,261.024658203125,485.7621154785156,1.0,200.92388916015625,484.3255920410156,1.0,230.97427368164062,485.0438537597656,0.9997925162315369
36,262.166748046875,485.3384704589844,1.0,200.36053466796875,485.3384704589844,1.0,231.26364135742188,485.3384704589844,0.9997956156730652
37,261.8055725097656,492.2261657714844,1.0,200.00253295898438,487.9107971191406,1.0,230.904052734375,490.0684814453125,0.9997947812080383
38,261.18048095703125,499.4032897949219,1.0,200.8507843017578,497.96429443359375,1.0,231.015625,498.68377685546875,0.9998548030853271
39,262.7232971191406,512.1820678710938,1.0,199.714111328125,510.74237060546875,1.0,231.2187042236328,511.46221923828125,0.9998512268066406
40,262.94732666015625,522.3572387695312,1.0,199.89401245117188,522.3572387695312,1.0,231.42066955566406,522.3572387695312,0.9998067021369934
41,262.6343078613281,541.9896240234375,1.0,199.37057495117188,540.55224609375,1.0,231.00244140625,541.2709350585938,0.9998562335968018
42,262.0350036621094,561.4135131835938,1.0,198.95907592773438,557.0931396484375,1.0,230.49703979492188,559.2532958984375,0.9997860789299011
43,260.7930603027344,581.7709350585938,1.0,197.80198669433594,580.3336791992188,1.0,229.29751586914062,581.0523071289062,0.9997451901435852
44,261.73291015625,598.64306640625,1.0,198.49017333984375,597.20703125,1.0,230.11154174804688,597.925048828125,0.9998037219047546
45,262.3050231933594,617.3662719726562,1.0,197.80426025390625,611.6123046875,1.0,230.0546417236328,614.4892578125,0.9998989105224609
46,259.4259033203125,625.0237426757812,1.0,197.69906616210938,622.1480102539062,1.0,228.56248474121094,623.5858764648438,0.9998879432678223
47,260.6748046875,631.947265625,1.0,195.8887481689453,631.947265625,1.0,228.28176879882812,631.947265625,0.9998373985290527
48,262.5149841308594,638.2702026367188,1.0,196.2781219482422,636.8302001953125,1.0,229.39654541015625,637.5501708984375,0.9999063014984131
49,262.2089538574219,641.8900146484375,1.0,197.5825653076172,640.4502563476562,1.0,229.895751953125,641.170166015625,0.9999027252197266
50,262.1329650878906,644.08984375,1.0,198.80029296875,642.652587890625,1.0,230.4666290283203,643.3712158203125,0.9998891353607178
51,262.3663635253906,644.813720703125,1.0,199.2119598388672,643.376708984375,1.0,230.78915405273438,644.09521484375,0.9998880624771118
52,263.01458740234375,644.6010131835938,1.0,198.5307159423828,643.1656494140625,1.0,230.77264404296875,643.88330078125,0.999885082244873
53,262.9591979980469,641.0958862304688,1.0,198.47523498535156,638.2215576171875,1.0,230.71722412109375,639.65869140625,0.999902606010437
54,263.32550048828125,635.8846435546875,1.0,198.6016845703125,634.4475708007812,1.0,230.96359252929688,635.1661376953125,0.9999157190322876
55,263.8769836425781,631.6487426757812,1.0,199.13296508789062,630.2105102539062,1.0,231.50497436523438,630.9296264648438,0.9999048709869385
56,263.81146240234375,627.8110961914062,1.0,197.9334716796875,626.3741455078125,1.0,230.87246704101562,627.0926513671875,0.9999011754989624
57,262.4561462402344,622.4658813476562,1.0,199.16244506835938,621.0267944335938,1.0,230.80929565429688,621.746337890625,0.9999048709869385
58,261.49517822265625,615.6492309570312,1.0,199.712158203125,615.6492309570312,1.0,230.60366821289062,615.6492309570312,0.9999247789382935
59,261.05352783203125,611.6629638671875,1.0,199.4610137939453,610.2264404296875,1.0,230.25726318359375,610.9447021484375,0.9998376369476318
60,262.7440185546875,604.84814453125,1.0,198.29173278808594,603.4102172851562,1.0,230.51788330078125,604.129150390625,0.9997978806495667
61,262.04058837890625,601.0794677734375,1.0,199.03399658203125,599.642822265625,1.0,230.53729248046875,600.3611450195312,0.9998908042907715
62,261.79736328125,599.6323852539062,1.0,198.56787109375,598.193115234375,1.0,230.1826171875,598.9127197265625,0.999890923500061
63,262.6344299316406,593.5335693359375,1.0,200.94058227539062,590.6590576171875,1.0,231.78750610351562,592.0963134765625,0.9998983144760132
64,261.61212158203125,590.107666015625,1.0,199.83908081054688,590.107666015625,1.0,230.72560119628906,590.107666015625,0.9998887777328491
65,261.3598937988281,589.1454467773438,1.0,201.08653259277344,587.7088012695312,1.0,231.22320556640625,588.4271240234375,0.9998306035995483
66,261.2069396972656,587.1177978515625,1.0,201.0773162841797,587.1177978515625,1.0,231.14212036132812,587.1177978515625,0.9998487234115601
67,261.24847412109375,586.360595703125,1.0,200.9015350341797,584.923095703125,1.0,231.07501220703125,585.641845703125,0.9998563528060913
68,262.76202392578125,590.0457153320312,1.0,202.32980346679688,590.0457153320312,1.0,232.54591369628906,590.0457153320312,0.9998464584350586
69,263.2068176269531,589.9970703125,1.0,202.8903350830078,589.9970703125,1.0,233.048583984375,589.9970703125,0.9998452663421631
70,263.478515625,589.7485961914062,1.0,201.78782653808594,589.7485961914062,1.0,232.6331787109375,589.7485961914062,0.9998760223388672
71,263.5687561035156,591.0855102539062,1.0,201.986328125,591.0855102539062,1.0,232.7775421142578,591.0855102539062,0.9998804330825806
72,262.7925720214844,590.442138671875,1.0,202.60250854492188,590.442138671875,1.0,232.69754028320312,590.442138671875,0.9998868703842163
73,263.82757568359375,589.7786254882812,1.0,203.6105499267578,589.7786254882812,1.0,233.71905517578125,589.7786254882812,0.9998804330825806
74,262.9482116699219,589.7476196289062,1.0,202.73873901367188,589.7476196289062,1.0,232.84347534179688,589.7476196289062,0.9998751878738403
75,261.9625244140625,588.616455078125,1.0,202.95541381835938,588.616455078125,1.0,232.45896911621094,588.616455078125,0.9998843669891357
76,262.5502624511719,588.2538452148438,1.0,202.28700256347656,589.6929931640625,1.0,232.41864013671875,588.973388671875,0.9998879432678223
77,262.78692626953125,589.836669921875,1.0,202.61923217773438,589.836669921875,1.0,232.7030792236328,589.836669921875,0.9998795986175537
78,263.23272705078125,589.834228515625,1.0,202.81004333496094,589.834228515625,1.0,233.02139282226562,589.834228515625,0.9998757839202881
79,261.6823425292969,587.712890625,1.0,201.56044006347656,589.1489868164062,1.0,231.62139892578125,588.430908203125,0.9998779296875
"""
    return (RAW_TRAJECTORY_CSV,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Coleta e qualidade do vídeo

    Foram fornecidos dois vídeos frontais do salto (um normal e outro em câmera lenta). 
    
    O arquivo normal tem 80 quadros, com taxa nominal de 30 Hz, e foi escolhido para a análise
    temporal. O arquivo em câmera lenta foi usado apenas para inspeção
    visual: sua duração é maior e não foi adotado para derivar
    velocidade/aceleração.

    **Avaliação da coleta atual.** A câmera permaneceu fixa, o corpo ficou
    inteiro no campo visual. As limitações são ausência de marcador físico contrastante,
    ausência de escala conhecida e perspectiva por uma câmera posicionada
    fora do plano estritamente perpendicular ao movimento.

    **Para uma coleta futura:** câmera fixa e perpendicular ao plano
    sagital/frontal escolhido, lente aproximadamente na altura do quadril,
    120 Hz ou mais, boa iluminação, marcador circular de alto contraste
    sobre a espinha ilíaca e uma régua/haste de dimensão conhecida no mesmo
    plano do marcador antes da tentativa.
    """)
    return


@app.cell
def _():
    FPS = 30.0  # taxa nominal do arquivo Salto_camera-normal.mp4
    G = 9.80665
    # Intervalo que apresentou trajetória aproximadamente parabólica.
    FLIGHT_FIRST_FRAME = 27
    FLIGHT_LAST_FRAME = 42
    return FLIGHT_FIRST_FRAME, FLIGHT_LAST_FRAME, FPS, G


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

    O gráfico abaixo é o controle da digitalização: mostra as trajetórias
    dos dois quadris e de seu ponto médio para todos os 80 quadros. Ele
    permite verificar a continuidade da marcação antes do cálculo das
    derivadas. Os dados brutos estão incorporados neste próprio arquivo
    `.py`, para que o notebook seja reproduzível mesmo quando enviado
    sozinho.
    """)
    return


@app.cell
def _(plt, raw):
    fig_qc, (ax_xy, ax_t) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")

    ax_xy.plot(
        raw["x_left_px"], raw["y_left_px"], "o-", ms=2.8, lw=1,
        color="#4c78a8", label="quadril esquerdo"
    )
    ax_xy.plot(
        raw["x_right_px"], raw["y_right_px"], "s-", ms=2.4, lw=1,
        color="#f58518", label="quadril direito"
    )
    ax_xy.plot(
        raw["x_hip_px"], raw["y_hip_px"], "^-", ms=2.6, lw=1.4,
        color="#54a24b", label="ponto médio"
    )
    ax_xy.set_title("Trajetória digitalizada no plano da imagem")
    ax_xy.set_xlabel("x (px)")
    ax_xy.set_ylabel("y (px; + para baixo)")
    ax_xy.invert_yaxis()
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(alpha=.25)
    ax_xy.legend(fontsize=8)

    ax_t.plot(raw["frame"], raw["y_left_px"], lw=1.3, color="#4c78a8", label="esquerdo")
    ax_t.plot(raw["frame"], raw["y_right_px"], lw=1.3, color="#f58518", label="direito")
    ax_t.plot(raw["frame"], raw["y_hip_px"], lw=1.8, color="#54a24b", label="médio")
    ax_t.set_title("Continuidade vertical da digitalização")
    ax_t.set_xlabel("quadro")
    ax_t.set_ylabel("y (px; + para baixo)")
    ax_t.grid(alpha=.25)
    ax_t.legend(fontsize=8)

    fig_qc
    return


@app.cell
def _(FPS, RAW_TRAJECTORY_CSV, StringIO, pd):
    raw = pd.read_csv(StringIO(RAW_TRAJECTORY_CSV)).sort_values("frame").reset_index(drop=True)
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
    mo.md(
        rf"""
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
        régua no plano do movimento deve substituir essa etapa em uma coleta
        final.
        """
    )
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
