"""
Campos Potenciais Artificiais - VSSS (APF, paradigma SENSE-ACT)

O robô não planeja rota: a cada passo lê as posições e segue o vetor resultante.
O atrator é o PONTO DE ABORDAGEM (atrás da bola, alinhado ao gol adversário):
    1. robô
    2. ponto de abordagem
    3. bola
    4. Gol Inimigo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button

# =============================================================================
# PARÂMETROS AJUSTÁVEIS
# =============================================================================

CENARIO = "normal"  # "normal" | "minimo_local"

# --- Ganhos do campo potencial ---
# Modelo CÔNICO para F_att: magnitude constante = k_att, sem crescer com a distância.
# Isso mantém F_att e F_rep na mesma escala.
k_att = 1.0      # magnitude da força atrativa (constante)
k_rep = 16000.0  # ganho repulsivo (≈ F_att em d = rho_0/2)
rho_0 = 22.0     # raio de influência dos obstáculos (cm)

# --- Tática de abordagem ---
# Distância atrás da bola onde o robô se posiciona antes do chute.
# Quanto maior, mais "certo" o ângulo de ataque, mas maior o desvio de rota.
d_offset = 28.0  # cm

# --- Movimento ---
v_max = 10.0  # velocidade máxima (cm/frame)
dt    = 1.0   # passo de tempo

# Limiar de chegada ao ponto de abordagem / à bola (cm)
limiar_posicionamento = 7.0  # chega ao ponto de abordagem
limiar_contato        = 8.0  # chega à bola

# --- Campo VSSS ---
CAMPO_W = 150
CAMPO_H = 130
MARGEM_X = 18
GOL_PROF = 14
GOL_ALT  = 30
GRADE_NX = 22
GRADE_NY = 18

# Centro da boca do gol inimigo (direita) — referência para calcular alinhamento
POS_GOL_INIMIGO = np.array([float(CAMPO_W), CAMPO_H / 2.0])

# --- Posições iniciais ---
if CENARIO == "normal":
    POS_ROBO_INICIAL = np.array([10.0, 10.0])
    POS_BOLA_INICIAL = np.array([110.0, 75.0])
    POS_OBS_INICIAIS = [
        np.array([55.0,  65.0]),
        np.array([95.0,  30.0]),
        np.array([90.0, 105.0]),
    ]

elif CENARIO == "minimo_local":
    POS_ROBO_INICIAL = np.array([20.0, 65.0])
    POS_BOLA_INICIAL = np.array([130.0, 65.0])
    POS_OBS_INICIAIS = [
        np.array([62.0, 48.0]),
        np.array([62.0, 82.0]),
        np.array([78.0, 65.0]),
    ]

# =============================================================================
# GEOMETRIA TÁTICA
# =============================================================================

def ponto_atrator(pos_bola: np.ndarray) -> np.ndarray:
    """
    Calcula o ponto de abordagem: atrás da bola, alinhado com o gol inimigo.

    Geometria:
        gol_inimigo -─ bola -─ [ATRATOR] -─ robô

    A direção do chute é (bola → gol). O atrator fica na direção oposta,
    d_offset cm atrás da bola. Assim o robô chega sempre pelo lado certo.
    """
    dir_chute = POS_GOL_INIMIGO - pos_bola  # aponta do robô p/ o gol
    d = np.linalg.norm(dir_chute)
    if d < 1e-6:
        return pos_bola.copy()
    # Ponto oposto: atrás da bola em relação ao gol
    atrator = pos_bola - d_offset * (dir_chute / d)
    # Garante que o atrator fica dentro do campo (com margem)
    atrator[0] = np.clip(atrator[0], 3.0, CAMPO_W - 3.0)
    atrator[1] = np.clip(atrator[1], 3.0, CAMPO_H - 3.0)
    return atrator


# =============================================================================
# CAMPO POTENCIAL
# =============================================================================

def forca_atrativa(pos_robo: np.ndarray, alvo: np.ndarray) -> np.ndarray:
    """
    Modelo CÔNICO: F_att = k_att * (alvo - pos_robo) / d
    Magnitude constante = k_att, independente da distância.

    Por que não o modelo parabólico?
    Com F = k_att * d e d_bola ≈ 70 cm → |F_att| ≈ 56, que suprime F_rep ≈ 0.5.
    O modelo cônico mantém ambos na mesma escala.
    """
    diff = alvo - pos_robo
    d = np.linalg.norm(diff)
    if d < 1e-6:
        return np.zeros(2)
    return k_att * (diff / d)


def forca_repulsiva(pos_robo: np.ndarray, pos_obs: np.ndarray) -> np.ndarray:
    """
    F_rep de Khatib (1986):
      d < rho_0: k_rep * (1/d - 1/rho_0) * (1/d²) * (pos_robo - pos_obs)/d
      d ≥ rho_0: 0
    """
    diff = pos_robo - pos_obs
    d = np.linalg.norm(diff)
    if d < 1e-6:
        d = 1e-6
    if d >= rho_0:
        return np.zeros(2)
    mag = k_rep * (1.0 / d - 1.0 / rho_0) * (1.0 / d ** 2)
    return mag * (diff / d)


def forca_total(pos_robo: np.ndarray,
                alvo: np.ndarray,
                obstaculos: list) -> np.ndarray:
    """Força resultante = atrativa (em direção ao alvo) + soma das repulsivas."""
    f = forca_atrativa(pos_robo, alvo)
    for obs in obstaculos:
        f += forca_repulsiva(pos_robo, obs)
    return f


def calcular_campo_vetorial(pos_bola: np.ndarray, obstaculos: list):
    """
    Calcula U, V normalizados na grade — usando o PONTO DE ABORDAGEM como alvo,
    não a bola. O campo mostra a intenção tática: a região entre bola e atrator
    tem vetores "para trás" (o robô não deve entrar por aí, ou ficará mal posicionado).
    """
    atrator = ponto_atrator(pos_bola)
    xs = np.linspace(0, CAMPO_W, GRADE_NX)
    ys = np.linspace(0, CAMPO_H, GRADE_NY)
    X, Y = np.meshgrid(xs, ys)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(GRADE_NY):
        for j in range(GRADE_NX):
            pos = np.array([X[i, j], Y[i, j]])
            f   = forca_total(pos, atrator, obstaculos)
            n   = np.linalg.norm(f)
            if n > 1e-6:
                U[i, j] = f[0] / n
                V[i, j] = f[1] / n
    return X, Y, U, V


# =============================================================================
# CAMPO DE FUTEBOL
# =============================================================================

def desenhar_campo(ax: plt.Axes) -> None:
    """Campo VSSS limpo: grama uniforme, marcações brancas, gols coloridos sem excesso."""
    lw  = 1.8
    cor = 'white'
    mg_gol = (CAMPO_H - GOL_ALT) / 2

    # Grama — cor única e suave
    ax.add_patch(patches.Rectangle(
        (0, 0), CAMPO_W, CAMPO_H,
        facecolor='#4a7c35', edgecolor=cor, linewidth=2.2, zorder=0
    ))

    # Linha central
    ax.plot([CAMPO_W / 2, CAMPO_W / 2], [0, CAMPO_H],
            color=cor, linewidth=lw, zorder=1)

    # Círculo central + ponto
    ax.add_patch(plt.Circle((CAMPO_W / 2, CAMPO_H / 2), radius=20,
                             color=cor, fill=False, linewidth=lw, zorder=1))
    ax.add_patch(plt.Circle((CAMPO_W / 2, CAMPO_H / 2), radius=2,
                             color=cor, zorder=1))

    # Arcos de canto
    for cx, cy, a1, a2 in [(0,0,0,90),(CAMPO_W,0,90,180),(CAMPO_W,CAMPO_H,180,270),(0,CAMPO_H,270,360)]:
        ax.add_patch(patches.Arc((cx, cy), 12, 12, theta1=a1, theta2=a2,
                                  color=cor, linewidth=lw * 0.8, zorder=1))

    # Áreas de gol
    prof_area, alt_area = 25, 60
    my = (CAMPO_H - alt_area) / 2
    for x0 in [0, CAMPO_W - prof_area]:
        ax.add_patch(patches.Rectangle((x0, my), prof_area, alt_area,
                                        linewidth=lw, edgecolor=cor,
                                        facecolor='none', zorder=1))

    # Pontos de pênalti
    for px in [34, CAMPO_W - 46]:
        ax.add_patch(plt.Circle((px, CAMPO_H / 2), radius=1.2, color=cor, zorder=1))

    # Gols — borda colorida, fill suave, sem rede (mais limpo e mais rápido)
    def _gol(x0, cor_gol, rotulo):
        ax.add_patch(patches.Rectangle(
            (x0, mg_gol), GOL_PROF * (1 if x0 < 0 else -1) * -1, GOL_ALT,
            linewidth=2, edgecolor=cor_gol, facecolor=cor_gol + '18', zorder=1
        ))
        ax.text(x0 + GOL_PROF * (0.5 if x0 >= 0 else -0.5),
                CAMPO_H / 2, rotulo,
                color=cor_gol, fontsize=8, fontweight='bold',
                ha='center', va='center', zorder=2,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.6, ec='none'))

    # Gol ALIADO (esquerda) — teal discreto
    ax.add_patch(patches.Rectangle((-GOL_PROF, mg_gol), GOL_PROF, GOL_ALT,
                                    lw=2, edgecolor='#1abc9c', facecolor='#1abc9c18', zorder=1))
    ax.text(-GOL_PROF / 2, CAMPO_H / 2, 'GOL\nALIADO',
            color='#148f77', fontsize=8, fontweight='bold',
            ha='center', va='center', zorder=2,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.65, ec='none'))

    # Gol INIMIGO (direita) — vermelho discreto
    ax.add_patch(patches.Rectangle((CAMPO_W, mg_gol), GOL_PROF, GOL_ALT,
                                    lw=2, edgecolor='#e74c3c', facecolor='#e74c3c18', zorder=1))
    ax.text(CAMPO_W + GOL_PROF / 2, CAMPO_H / 2, 'GOL\nINIMIGO',
            color='#c0392b', fontsize=8, fontweight='bold',
            ha='center', va='center', zorder=2,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.65, ec='none'))

    # Traves
    ax.plot([0, 0], [mg_gol, mg_gol + GOL_ALT],
            color=cor, lw=3, solid_capstyle='round', zorder=2)
    ax.plot([CAMPO_W, CAMPO_W], [mg_gol, mg_gol + GOL_ALT],
            color=cor, lw=3, solid_capstyle='round', zorder=2)
    ax.plot([-GOL_PROF, -GOL_PROF], [mg_gol, mg_gol + GOL_ALT],
            color='#1abc9c', lw=2, solid_capstyle='round', zorder=2)
    ax.plot([CAMPO_W + GOL_PROF, CAMPO_W + GOL_PROF], [mg_gol, mg_gol + GOL_ALT],
            color='#e74c3c', lw=2, solid_capstyle='round', zorder=2)


# =============================================================================
# SIMULAÇÃO E ANIMAÇÃO
# =============================================================================

def rodar_simulacao() -> None:
    fig = plt.figure(figsize=(12, 9.4))
    fig.patch.set_facecolor('#f5f5f5')
    ax = fig.add_axes([0, 0.06, 1, 0.94])
    ax.set_facecolor('#d8e8d0')   # cinza-esverdeado neutro fora do campo
    ax.set_xlim(-MARGEM_X, CAMPO_W + MARGEM_X)
    ax.set_ylim(-6, CAMPO_H + 6)
    ax.set_aspect('equal')
    ax.axis('off')

    titulo_base = (
        f"Campos Potenciais — VSSS v2  |  Cenário: {CENARIO}\n"
        f"k_att={k_att}  k_rep={k_rep}  ρ₀={rho_0}cm  d_offset={d_offset}cm  "
        f"[arraste bola/adversários • R = reiniciar]"
    )
    ax.set_title(titulo_base, fontsize=10, pad=10, color='#222')
    desenhar_campo(ax)

    # ----------------------------------------------------------------- Estado
    estado = {
        'pos':      POS_ROBO_INICIAL.copy().astype(float),
        'pos_bola': POS_BOLA_INICIAL.copy().astype(float),
        'obs':      [o.copy().astype(float) for o in POS_OBS_INICIAIS],
        'parado':   False,
        'fase':     'posicionamento',  # 'posicionamento' → 'contato' → parado
        'hist':     [],
    }

    # --------------------------------------------------------- Quiver inicial
    atrator_init = ponto_atrator(estado['pos_bola'])
    X, Y, U, V = calcular_campo_vetorial(estado['pos_bola'], estado['obs'])
    quiver_obj = ax.quiver(
        X, Y, U, V,
        alpha=0.45, color='white',
        scale=30, width=0.003, headwidth=4, zorder=2
    )

    # ----------------------------------------- Linha de alinhamento tático
    # Mostra a reta: ponto_atrator - bola - gol_inimigo
    # Essa linha deixa claro para a plateia qual é a "intenção" do robô.
    def _pts_linha(pos_bola):
        """Retorna os dois extremos da linha de alinhamento para o plot."""
        at = ponto_atrator(pos_bola)
        # Extrapola a linha até a borda direita do eixo
        return [at[0], CAMPO_W + MARGEM_X], [at[1], pos_bola[1] + (CAMPO_W + MARGEM_X - at[0]) * (POS_GOL_INIMIGO[1] - at[1]) / max(POS_GOL_INIMIGO[0] - at[0], 1)]

    lx, ly = _pts_linha(estado['pos_bola'])
    linha_alinhamento, = ax.plot(
        lx, ly,
        color='gold', linewidth=1.2, linestyle='--', alpha=0.55,
        zorder=3, label='Linha de chute'
    )

    # ------------------------------------------ Marcador: ponto de abordagem
    # Estrela dourada — é AQUI que o robô quer chegar primeiro.
    atrator_mark, = ax.plot(
        *atrator_init, marker='*', markersize=14,
        color='gold', markeredgecolor='darkorange', markeredgewidth=0.8,
        zorder=7, linestyle='none'
    )
    # -------------------------------------------------------------- Bola
    bola_patch = plt.Circle(estado['pos_bola'], radius=5,
                             color='darkorange', zorder=5)
    ax.add_patch(bola_patch)

    # --------------------------------------------------------- Adversários
    obs_patches, inf_patches = [], []
    for obs in estado['obs']:
        p = plt.Circle(obs, radius=7, color='crimson', zorder=5)
        c = plt.Circle(obs, radius=rho_0, color='crimson', fill=False,
                       linestyle='--', linewidth=0.8, alpha=0.35, zorder=3)
        ax.add_patch(p); ax.add_patch(c)
        obs_patches.append(p); inf_patches.append(c)

    # --------------------------------------------------------- Robô atacante
    robo_patch = plt.Circle(estado['pos'], radius=7, color='limegreen',
                             zorder=6, ec='darkgreen', linewidth=1.5)
    ax.add_patch(robo_patch)

    rastro_x = [estado['pos'][0]]
    rastro_y = [estado['pos'][1]]
    rastro_linha, = ax.plot(rastro_x, rastro_y,
                             color='limegreen', linewidth=2.2, alpha=0.95, zorder=4)

    # -------------------------------------------------------------- Legenda
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
               markersize=10, label='Robô atacante'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
               markersize=10, label='Bola  [arraste]'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
               markersize=10, label='Adversário  [arraste]'),
        Line2D([0], [0], marker='*', color='gold', markersize=12,
               label=f'Ponto de abordagem  (−{d_offset}cm)'),
        Line2D([0], [0], color='steelblue', linewidth=1.5, label='Campo de forças'),
        Line2D([0], [0], color='gold', linewidth=1.2, linestyle='--',
               label='Linha de chute'),
    ], loc='upper left', fontsize=8.5, framealpha=0.88)

    # ==========================================================================
    # Auxiliares de atualização visual
    # ==========================================================================

    def _atualizar_visuais_taticos(pos_bola):
        """Recalcula e reposiciona o ponto de abordagem e a linha de alinhamento."""
        at = ponto_atrator(pos_bola)
        atrator_mark.set_data([at[0]], [at[1]])

        # Linha do ponto de abordagem até o gol inimigo, passando pela bola
        # Usamos dois pontos: atrator e um ponto além do gol à direita
        dx = POS_GOL_INIMIGO[0] - at[0]
        dy = POS_GOL_INIMIGO[1] - at[1]
        if abs(dx) > 1e-6:
            x_fim = CAMPO_W + MARGEM_X
            y_fim = at[1] + dy * (x_fim - at[0]) / dx
        else:
            x_fim, y_fim = at[0], CAMPO_H + 6
        linha_alinhamento.set_data([at[0], x_fim], [at[1], y_fim])

    def _atualizar_quiver(pos_bola, obs):
        _, _, U2, V2 = calcular_campo_vetorial(pos_bola, obs)
        quiver_obj.set_UVC(U2, V2)

    def _resetar_titulo():
        ax.set_title(titulo_base, fontsize=10, color='#222')

    def _resetar_robo():
        estado['pos'][:]  = POS_ROBO_INICIAL.copy()
        estado['parado']  = False
        estado['fase']    = 'posicionamento'
        estado['hist'].clear()
        robo_patch.center = estado['pos']
        rastro_x.clear(); rastro_x.append(estado['pos'][0])
        rastro_y.clear(); rastro_y.append(estado['pos'][1])
        rastro_linha.set_data(rastro_x, rastro_y)
        _resetar_titulo()

    # ==========================================================================
    # Interatividade: drag com mouse
    # ==========================================================================

    drag = {'ativo': False, 'alvo': None}
    RAIO_CAPTURA = 14.0

    def on_press(event):
        if event.inaxes != ax or event.button != 1:
            return
        click = np.array([event.xdata, event.ydata])
        if np.linalg.norm(click - estado['pos_bola']) < RAIO_CAPTURA:
            drag['ativo'] = True; drag['alvo'] = 'bola'; return
        for i, obs in enumerate(estado['obs']):
            if np.linalg.norm(click - obs) < RAIO_CAPTURA:
                drag['ativo'] = True; drag['alvo'] = i; return

    def on_motion(event):
        if not drag['ativo'] or event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        nova_pos = np.array([
            np.clip(event.xdata, 0, CAMPO_W),
            np.clip(event.ydata, 0, CAMPO_H),
        ])
        if drag['alvo'] == 'bola':
            estado['pos_bola'][:] = nova_pos
            bola_patch.center     = nova_pos
            estado['parado'] = False
            estado['fase']   = 'posicionamento'
            estado['hist'].clear()
            rastro_x.clear(); rastro_x.append(estado['pos'][0])
            rastro_y.clear(); rastro_y.append(estado['pos'][1])
            rastro_linha.set_data(rastro_x, rastro_y)
            _resetar_titulo()
            _atualizar_visuais_taticos(nova_pos)
        else:
            i = drag['alvo']
            estado['obs'][i][:]   = nova_pos
            obs_patches[i].center = nova_pos
            inf_patches[i].center = nova_pos
            estado['parado'] = False
            estado['fase']   = 'posicionamento'
            estado['hist'].clear()
            _resetar_titulo()
        _atualizar_quiver(estado['pos_bola'], estado['obs'])
        fig.canvas.draw_idle()

    def on_release(event):
        drag['ativo'] = False; drag['alvo'] = None

    def on_key(event):
        if event.key in ('r', 'R'):
            _resetar_robo()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('motion_notify_event',  on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event',      on_key)

    # -------------------------------------------------------------- Botão Pause
    ax_btn = fig.add_axes([0.42, 0.01, 0.16, 0.04])
    btn_pause = Button(ax_btn, 'Pausar', color='#ddeeff', hovercolor='#bbddff')
    pausado = {'v': False}

    def on_pause(_event):
        pausado['v'] = not pausado['v']
        btn_pause.label.set_text('Continuar' if pausado['v'] else 'Pausar')
        fig.canvas.draw_idle()

    btn_pause.on_clicked(on_pause)

    # ==========================================================================
    # Animação: máquina de estados SENSE-ACT de duas fases
    # ==========================================================================

    def atualizar(frame: int):
        """
        SENSE-ACT com duas fases táticas:

        Fase 1 — POSICIONAMENTO:
          O alvo é o ponto de abordagem (atrás da bola, alinhado ao gol).
          O robô desvia dos adversários e se aproxima da posição de chute.

        Fase 2 — CONTATO:
          Ao chegar ao ponto de abordagem, o alvo muda para a BOLA.
          O robô empurra a bola na direção exata do gol inimigo.
        """
        if estado['parado'] or pausado['v']:
            return

        pos  = estado['pos']
        fase = estado['fase']
        atrator = ponto_atrator(estado['pos_bola'])

        # --- Seleciona alvo conforme a fase ---
        if fase == 'posicionamento':
            alvo = atrator
            if np.linalg.norm(pos - atrator) < limiar_posicionamento:
                estado['fase'] = 'contato'
                ax.set_title(titulo_base + "\n▶ Em posição! Avançando para o chute...",
                             fontsize=10, color='steelblue')
                return
        else:  # contato
            alvo = estado['pos_bola']
            if np.linalg.norm(pos - estado['pos_bola']) < limiar_contato:
                estado['parado'] = True
                ax.set_title(titulo_base + "\n⚽ Contato! Bola chutada em direção ao gol inimigo!",
                             fontsize=10, color='darkgreen')
                return

        # --- SENSE → ACT ---
        f = forca_total(pos, alvo, estado['obs'])
        n = np.linalg.norm(f)

        # Detecção de mínimo local: janela de 20 frames, threshold maior para
        # capturar tanto robôs parados (desvio≈0) quanto oscilando no lugar.
        hist = estado['hist']
        hist.append(pos.copy())
        if len(hist) > 20:
            hist.pop(0)
        if len(hist) == 20:
            desvio = np.std([p[0] for p in hist]) + np.std([p[1] for p in hist])
            if desvio < 4.0:
                estado['parado'] = True
                ax.set_title(
                    titulo_base + "\n⚠ MÍNIMO LOCAL — robô preso!  (mova um adversário para continuar)",
                    fontsize=10, color='darkred')
                return

        if n < 1e-9:
            return  # força zero transitória, aguarda próximo frame

        velocidade = (f / n) * min(n, v_max)
        pos_nova   = np.clip(pos + velocidade * dt,
                              [0.0, 0.0], [float(CAMPO_W), float(CAMPO_H)])
        estado['pos'][:] = pos_nova
        robo_patch.center = pos_nova
        rastro_x.append(pos_nova[0]); rastro_y.append(pos_nova[1])
        rastro_linha.set_data(rastro_x, rastro_y)

    anim = FuncAnimation(  # noqa: F841 — referência mantida viva pelo escopo
        fig, atualizar,
        frames=2000, interval=40,
        blit=False, repeat=False,
    )

    plt.tight_layout()
    plt.show()


# =============================================================================
# PONTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  Campos Potenciais — VSSS v2  (abordagem tática)")
    print(f"  Cenário  : {CENARIO}")
    print(f"  d_offset : {d_offset} cm atrás da bola")
    print(f"  k_att={k_att}  k_rep={k_rep}  rho_0={rho_0}cm")
    print("=" * 65)
    print("  A estrela dourada = ponto de abordagem (robô posiciona aí primeiro)")
    print("  A linha dourada   = alinhamento gol ← bola ← robô")
    print("  Arraste a bola para ver o ponto de abordagem mudar dinamicamente.")
    print("  Pressione R para reiniciar o robô.")
    print("=" * 65)
    rodar_simulacao()
