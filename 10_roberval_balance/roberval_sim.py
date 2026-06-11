"""
Roberval Balance — Dynamics & Force Simulation
===============================================
Reference: Park, Choi & Kim (2019)
"What Force and Torque Are Working with the Roberval Balance?"
The Physics Teacher, Vol. 57, March 2019

Panels
------
  Left  : Animated balance diagram with force vectors (N_L/N_R, F1/F2)
  Top-R  : θ(t) — tilt angle time series (ODE integration)
  Mid-R  : N_L, N_R vs object position a_R  (Eqs. 1 & 2)
  Bot-R  : Internal horizontal forces F1, F2 vs a_R  (Eqs. 4 & 5)

Sliders
-------
  W_L (g), W_R (g) — object masses
  a_R (cm)         — right-object position along platform
  Damping          — rotational damping coefficient
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# ── Physical constants & geometry ────────────────────────────────────────────

G      = 9.81          # m/s²
L      = 0.150         # beam half-length (pivot → arm attachment), m
D      = 0.060         # vertical gap between upper / lower beam, m
B      = 0.100         # half base-width (for N_L/N_R formula), m
PLAT   = 0.065         # half-width of arm platform, m

M_BEAM = 0.40          # total double-beam mass, kg
M_ARM  = 0.15          # one arm + platform mass, kg
M_POST = 0.20          # central post mass, kg
W0     = (M_BEAM + 2*M_ARM + M_POST) * G   # balance's own weight, N

C_DAMP     = 0.05      # default rotational damping, N·m·s/rad
THETA_STOP = np.radians(38)   # mechanical stop, rad

# ── Physics ───────────────────────────────────────────────────────────────────

def I_eff(WL: float, WR: float) -> float:
    """Effective rotational inertia about the central pivot axis (kg·m²).

    Contributions:
      - Double beam (two rods, total mass M_BEAM, each length 2L, rotating about centre)
      - Two arms (translating at radius L)
      - Two objects (translating at radius L)
    """
    return ((1/3)*M_BEAM + 2*M_ARM + (WL + WR)/G) * L**2


def ode(t, y, WL, WR, c):
    """Equation of motion:  I·θ̈ = L·cos(θ)·(W_R − W_L) − c·θ̇

    Key result: object position (a_L, a_R) does NOT appear in the torque.
    Only the weight DIFFERENCE drives tilt — this is the Roberval principle.
    """
    theta, dtheta = y
    tau = L * np.cos(theta) * (WR - WL) - c * dtheta
    # Soft mechanical stops (penalty spring)
    if theta >  THETA_STOP: tau -= 600 * (theta -  THETA_STOP)
    if theta < -THETA_STOP: tau -= 600 * (theta + THETA_STOP)
    return [dtheta, tau / I_eff(WL, WR)]


def simulate(WL, WR, c=C_DAMP, theta0=0.03, t_end=6.0, dt=0.025):
    """Integrate EOM; return (t, theta) numpy arrays."""
    sol = solve_ivp(
        ode, (0, t_end), [theta0, 0.0],
        args=(WL, WR, c),
        t_eval=np.arange(0, t_end, dt),
        method='RK45', max_step=dt
    )
    return sol.t, sol.y[0]


def normal_forces_gf(aR, W, aL=0.0):
    """Ground normal forces from paper Eqs. (1)–(2).

    Assumes both objects have the same weight W (position-variation test).
    Returns (N_L, N_R) in gram-force (gf).
    """
    N_sum  = W0 + 2*W                       # Eq.(1): NR + NL = W0 + 2W
    N_diff = ((aR - aL) / B) * W            # Eq.(2): b(NR - NL) = Δa·W
    NL = (N_sum - N_diff) / G * 1000
    NR = (N_sum + N_diff) / G * 1000
    return NL, NR


def horiz_forces(aR, W, aL=0.0):
    """Internal horizontal forces F1 (left arm), F2 (right arm) in N.

    From torque equilibrium on each arm — paper Eqs.(4) & (5):
        d·F1 = a_L·W   →  F1 = a_L·W / D
        d·F2 = a_R·W   →  F2 = a_R·W / D
    """
    return aL*W/D, aR*W/D


def delta_a_max(W):
    """Maximum object displacement before tipping — paper Eq.(3)."""
    return (2 + W0/W) * B


# ── Geometry ──────────────────────────────────────────────────────────────────

def key_pts(theta):
    """Six key connection points for beam angle θ.

    Convention: θ > 0  →  right side down.
    Upper beam pivots at (0, +D/2); lower beam pivots at (0, -D/2).
    Arms stay vertical (parallelogram constraint).
    """
    ct, st = np.cos(theta), np.sin(theta)
    return {
        'pu': np.array([0.,  D/2]),                    # upper pivot on post
        'pl': np.array([0., -D/2]),                    # lower pivot on post
        'ru': np.array([ L*ct,  D/2 - L*st]),          # upper beam, right end
        'rl': np.array([ L*ct, -D/2 - L*st]),          # lower beam, right end
        'lu': np.array([-L*ct,  D/2 + L*st]),          # upper beam, left end
        'll': np.array([-L*ct, -D/2 + L*st]),          # lower beam, left end
    }


# ── Drawing ───────────────────────────────────────────────────────────────────

C_STRUCT = '#2b2d42'
C_ARM    = '#1e6091'
C_OBJ_R  = '#e63946'
C_OBJ_L  = '#457b9d'
C_NF     = '#2a9d8f'
C_HF     = '#f4a261'


def draw_balance(ax, theta, aR, WL, WR, aL=0.0):
    ax.clear()
    ax.set_xlim(-0.33, 0.33)
    ax.set_ylim(-0.30, 0.28)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')
    ax.axis('off')

    p = key_pts(theta)
    ru, rl, lu, ll = p['ru'], p['rl'], p['lu'], p['ll']

    # Beams
    for a, b in [(lu, ru), (ll, rl)]:
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=C_STRUCT, lw=3.5, solid_capstyle='round', zorder=3)

    # Central post (fixed)
    ax.plot([0, 0], [p['pl'][1]-0.01, p['pu'][1]+0.01],
            color=C_STRUCT, lw=6, zorder=4)

    # Vertical arms
    for top, bot in [(ru, rl), (lu, ll)]:
        ax.plot([top[0], bot[0]], [top[1], bot[1]],
                color=C_ARM, lw=3, zorder=3)
        ax.plot([bot[0], bot[0]], [bot[1], bot[1]-0.022],
                color=C_ARM, lw=3, zorder=3)

    # Horizontal platforms
    for con in [ru, lu]:
        ax.plot([con[0]-PLAT, con[0]+PLAT], [con[1], con[1]],
                color=C_ARM, lw=5, solid_capstyle='round', zorder=5)

    # Axle circles
    for pt in p.values():
        ax.add_patch(plt.Circle(pt, 0.006, color='#777', zorder=9,
                                ec=C_STRUCT, lw=0.8))

    # Objects
    for arm_x, arm_y, a_obj, W_obj, col in [
        (ru[0], ru[1], aR, WR, C_OBJ_R),
        (lu[0], lu[1], aL, WL, C_OBJ_L),
    ]:
        s  = 0.015 + 0.007*(W_obj/G)
        xc = arm_x + a_obj
        ax.add_patch(plt.Rectangle((xc-s, arm_y), 2*s, 2*s,
                                    color=col, zorder=7, alpha=0.92,
                                    ec='white', lw=0.8))
        ax.text(xc, arm_y + 2*s + 0.012, f'{W_obj/G*1000:.0f} g',
                ha='center', fontsize=8.5, color=col, fontweight='bold')

    # Ground reference
    gy = -0.24
    ax.axhline(gy, color='#bbb', lw=1.5, ls='--', alpha=0.7)

    # Normal-force arrows (N_L, N_R)
    NL, NR = normal_forces_gf(aR, WR, aL)
    ref_gf = (W0 + 2*WR) / G * 1000 / 2
    sc_n   = 0.09 / ref_gf
    for x, N, lbl in [(-B, NL, f'N_L\n{NL:.0f} gf'), (B, NR, f'N_R\n{NR:.0f} gf')]:
        ax.annotate('', xy=(x, gy + N*sc_n), xytext=(x, gy),
                    arrowprops=dict(arrowstyle='->', color=C_NF, lw=2.2), zorder=6)
        ax.text(x, gy + N*sc_n + 0.012, lbl,
                ha='center', va='bottom', fontsize=8, color=C_NF)

    # Internal horizontal-force arrows (F1, F2)
    F1, F2 = horiz_forces(aR, WR, aL)
    F_max  = max(abs(F1), abs(F2), 0.05)
    sc_f   = 0.055 / F_max
    mid_R  = np.array([ru[0], (ru[1]+rl[1])/2])
    mid_L  = np.array([lu[0], (lu[1]+ll[1])/2])
    for mid, F, direction, lbl in [
        (mid_R, F2, -1, f'F₂={F2:.2f} N'),
        (mid_L, F1,  1, f'F₁={F1:.2f} N'),
    ]:
        if abs(F) > 0.05:
            ax.annotate('', xy=(mid[0] + direction*F*sc_f, mid[1]), xytext=mid,
                        arrowprops=dict(arrowstyle='->', color=C_HF, lw=2.0), zorder=6)
            ha = 'right' if direction < 0 else 'left'
            ax.text(mid[0] + direction*F*sc_f + direction*0.008, mid[1],
                    lbl, ha=ha, fontsize=7.5, color=C_HF)

    # Angle readout
    ax.text(0.30, 0.24, f'θ = {np.degrees(theta):+.1f}°',
            fontsize=11, ha='right', color=C_STRUCT,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.88, ec='#ccc'))
    ax.set_title('Roberval Balance — Live Simulation', fontsize=11, pad=5)


# ── Static plots ──────────────────────────────────────────────────────────────

def plot_theta(ax, t_arr, th_arr, frame):
    ax.clear()
    ax.plot(t_arr, np.degrees(th_arr), color=C_STRUCT, lw=1.8)
    ax.axhline(0, color='#aaa', lw=0.8, ls='--')
    fi = min(frame, len(t_arr)-1)
    ax.axvline(t_arr[fi], color=C_OBJ_R, lw=1.3, ls='--', alpha=0.7)
    ax.scatter([t_arr[fi]], [np.degrees(th_arr[fi])],
               color=C_OBJ_R, s=40, zorder=5)
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.set_ylabel('θ (°)', fontsize=9)
    ax.set_title('Tilt angle θ(t)', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def plot_nf(ax, W, aR_cur, aL=0.0):
    ax.clear()
    a_arr = np.linspace(-PLAT, PLAT, 300)
    NL_arr = [normal_forces_gf(a, W, aL)[0] for a in a_arr]
    NR_arr = [normal_forces_gf(a, W, aL)[1] for a in a_arr]
    ax.plot(a_arr*100, NL_arr, color=C_OBJ_L, lw=2, label='N_L')
    ax.plot(a_arr*100, NR_arr, color=C_OBJ_R, lw=2, label='N_R')
    NL0, NR0 = normal_forces_gf(aR_cur, W, aL)
    ax.axvline(aR_cur*100, color='#888', lw=1, ls='--', alpha=0.6)
    ax.scatter([aR_cur*100]*2, [NL0, NR0], color=[C_OBJ_L, C_OBJ_R], s=45, zorder=5)
    am = delta_a_max(W)*100
    ax.axvline( am, color=C_OBJ_R, lw=1, ls=':', alpha=0.5, label=f'Δa_max={am:.1f} cm')
    ax.axvline(-am, color=C_OBJ_L, lw=1, ls=':', alpha=0.5)
    ax.set_xlabel('a_R (cm)', fontsize=9)
    ax.set_ylabel('Normal force (gf)', fontsize=9)
    ax.set_title('Ground normal forces (Eqs. 1 & 2)', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


def plot_hf(ax, W, aR_cur, aL=0.0):
    ax.clear()
    a_arr = np.linspace(-PLAT, PLAT, 300)
    F1_arr = [horiz_forces(a, W, aL)[0] for a in a_arr]
    F2_arr = [horiz_forces(a, W, aL)[1] for a in a_arr]
    ax.plot(a_arr*100, F1_arr, color=C_OBJ_L, lw=2, label='F₁ (left arm)')
    ax.plot(a_arr*100, F2_arr, color=C_OBJ_R, lw=2, label='F₂ (right arm)')
    F1_0, F2_0 = horiz_forces(aR_cur, W, aL)
    ax.axvline(aR_cur*100, color='#888', lw=1, ls='--', alpha=0.6)
    ax.scatter([aR_cur*100]*2, [F1_0, F2_0], color=[C_OBJ_L, C_OBJ_R], s=45, zorder=5)
    ax.set_xlabel('a_R (cm)', fontsize=9)
    ax.set_ylabel('Force (N)', fontsize=9)
    ax.set_title('Internal horizontal forces (Eqs. 4 & 5)', fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#ffffff')
    plt.suptitle(
        'Roberval Balance Dynamics  |  Park, Choi & Kim (2019) The Physics Teacher',
        fontsize=10, color='#555', y=0.98, style='italic'
    )

    gs = GridSpec(3, 2, figure=fig,
                  left=0.06, right=0.97, top=0.95, bottom=0.22,
                  hspace=0.50, wspace=0.30)

    ax_bal = fig.add_subplot(gs[0:3, 0])   # balance (spans all 3 rows, left col)
    ax_th  = fig.add_subplot(gs[0, 1])
    ax_nf  = fig.add_subplot(gs[1, 1])
    ax_hf  = fig.add_subplot(gs[2, 1])

    # ── Sliders ────────────────────────────────────────────────────────────
    sl_defs = [
        ('W_L (g)',   100,  10, 400),
        ('W_R (g)',   100,  10, 400),
        ('a_R (cm)',    0,  -6,   6),
        ('Damping',    50,   5, 200),
    ]
    sliders = {}
    for i, (lbl, val, vmin, vmax) in enumerate(sl_defs):
        sax = fig.add_axes([0.15, 0.155 - i*0.040, 0.72, 0.025])
        sl  = Slider(sax, lbl, vmin, vmax, valinit=val,
                     color='#4a90d9', track_color='#e0e0e0')
        sl.label.set_fontsize(9)
        sl.valtext.set_fontsize(9)
        sliders[lbl] = sl

    # ── Shared simulation state ────────────────────────────────────────────
    state = {
        'WL': 1.0*G, 'WR': 1.0*G, 'aR': 0.0, 'c': C_DAMP,
        't': None, 'th': None, 'frame': 0,
    }

    def recompute():
        state['t'], state['th'] = simulate(
            state['WL'], state['WR'], c=state['c'])
        state['frame'] = 0

    recompute()
    plot_nf(ax_nf, state['WR'], state['aR'])
    plot_hf(ax_hf, state['WR'], state['aR'])

    # ── Animation ──────────────────────────────────────────────────────────
    def animate(frame):
        state['frame'] = frame % len(state['t'])
        fi = state['frame']
        draw_balance(ax_bal, state['th'][fi], state['aR'],
                     state['WL'], state['WR'])
        plot_theta(ax_th, state['t'], state['th'], fi)
        fig.canvas.draw_idle()

    ani = animation.FuncAnimation(
        fig, animate, interval=25, blit=False, cache_frame_data=False
    )

    # ── Slider callbacks ───────────────────────────────────────────────────
    def on_change(_):
        state['WL'] = sliders['W_L (g)'].val / 1000 * G
        state['WR'] = sliders['W_R (g)'].val / 1000 * G
        state['aR'] = sliders['a_R (cm)'].val / 100
        state['c']  = sliders['Damping'].val  / 1000
        recompute()
        plot_nf(ax_nf, state['WR'], state['aR'])
        plot_hf(ax_hf, state['WR'], state['aR'])

    for sl in sliders.values():
        sl.on_changed(on_change)

    plt.show()
    return ani   # keep reference to prevent GC


if __name__ == '__main__':
    _ani = main()
