#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correlation_dashboard_v15.py

- Uses user's robust exact() that takes (n_max, beta, coeffs, order_i, order_j, t_end, delta_t).
- Coefficient sliders (c1..c4) affect the exact curve after clicking Apply.
- Initial polymer configuration (cosine ring + small noise) shown at launch (no Reset needed).
- Single Start/Pause toggle + Reset.
- N shown on the correlation plot is the total number of samples in the history mean:
    N = history_count * len(history_mean)   (0 if no history yet)
"""

import threading, time, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ---- Polynomial potential and its derivative ----
def V_poly(x, c1, c2, c3, c4):
    return c1*x + c2*(x**2) + c3*(x**3) + c4*(x**4)

def dV_poly(x, c1, c2, c3, c4):
    return c1 + 2*c2*x + 3*c3*(x**2) + 4*c4*(x**3)

# ===== Exact quantum correlation (robust, coefficient-based) =====
import numpy as _np
from numpy.linalg import eigh
hbar = 1.0
m = 1.0  # mass for exact()

def exact(n_max, beta, coeffs, order_i=1, order_j=1, t_end=20, delta_t=0.02): 
    max_order = int(max(order_i, order_j, len(coeffs)))
    dim = n_max + max_order
    a = _np.zeros((dim, dim))
    for n in range(1, dim):
        a[n-1, n] = _np.sqrt(n)
    a_dag = a.T
    x = _np.sqrt(hbar/(2*m)) * (a + a_dag)
    x_powers = [x]
    for i in range(0, max_order-1):
        x_powers.append(x_powers[i] @ x)
    V = _np.zeros((dim, dim))
    for i in range(0, len(coeffs)):
        V += coeffs[i] * x_powers[i]
    P = _np.sqrt(m * hbar / 2) * (a_dag - a)
    T = -(P @ P) / (2 * m)
    H = (T + V)[:n_max, :n_max]
    E, psi = eigh(H)
    Boltz = _np.exp(-beta * E); Z = _np.sum(Boltz)
    A = x_powers[order_i-1][:n_max, :n_max]; A_mel = psi.T @ A @ psi
    B = x_powers[order_j-1][:n_max, :n_max]; B_mel = psi.T @ B @ psi
    Em, En = _np.meshgrid(E, E, indexing='ij'); dE = Em - En; eps = 1e-10
    with _np.errstate(divide='ignore', invalid='ignore'):
        W = _np.where(_np.abs(dE) < eps, Boltz[:, None], (Boltz[None, :] - Boltz[:, None]) / (beta * dE))
    AB = A_mel * B_mel.T
    times = _np.arange(0, t_end, delta_t)
    phases = _np.exp(1j * dE[None, :, :] * times[:, None, None] / hbar)
    C_t = _np.einsum('ijk,jk,jk->i', phases, W, AB).real / Z
    return times, C_t


# -------------- Normal-mode helpers (exact propagation of internal modes) -------------
def force_value(coeffs, x):
    # Force = -dV/dx for V(x) = c1 x + c2 x^2 + c3 x^3 + c4 x^4
    c1, c2, c3, c4 = coeffs
    return -(c1 + 2.0*c2*x + 3.0*c3*(x**2) + 4.0*c4*(x**3))

def normal_mode_frequencies(k_eff, n, m):
    # omega_k = sqrt( (2 - 2 cos(2πk/n)) * k_eff / m )
    k = np.arange(n, dtype=float)
    return np.sqrt((2.0 - 2.0*np.cos(2.0*np.pi*k/n)) * (k_eff / m))

def mode_propagation(x, v, omega, delta_t):
    # Transform to normal coordinates (complex arrays from FFT)
    Xk = np.fft.fft(x)
    Vk = np.fft.fft(v)

    Xk_next = np.empty_like(Xk)
    Vk_next = np.empty_like(Vk)

    # k=0 mode (omega=0)
    Xk_next[0] = Xk[0] + Vk[0] * delta_t
    Vk_next[0] = Vk[0]

    # k>0 modes
    omega_pos = omega[1:]
    coswt = np.cos(omega_pos * delta_t)
    sinwt = np.sin(omega_pos * delta_t)
    Xk_pos = Xk[1:]
    Vk_pos = Vk[1:]
    # Avoid division by zero for any accidental zeros (shouldn't happen except k=0 handled above)
    with np.errstate(divide='ignore', invalid='ignore'):
        Xk_next[1:] = Xk_pos * coswt + Vk_pos * (sinwt / np.where(omega_pos==0.0, 1.0, omega_pos))
        Vk_next[1:] = Vk_pos * coswt - Xk_pos * (omega_pos * sinwt)

    # Back to real space
    x_next = np.fft.ifft(Xk_next).real
    v_next = np.fft.ifft(Vk_next).real
    return x_next, v_next

# -------------- Ring-polymer integrator -------------
def polymer_verlet_core(q, v, beta_n, n_steps, dt, n, force_func, m, c_params):
    """Velocity-Verlet for ring polymer (ω_P = 1/beta_n). Returns series for n_steps."""
    omegaP = 1.0 / max(1e-16, beta_n)
    q_series = np.empty((n_steps, n), dtype=float)
    v_series = np.empty((n_steps, n), dtype=float)
    qk = q.copy(); vk = v.copy()
    for t in range(n_steps):
        q_plus = np.roll(qk, -1); q_minus = np.roll(qk, 1)
        F_spring = -m*(omegaP**2) * (2.0*qk - q_plus - q_minus)
        F_pot = -force_func(qk, *c_params)
        F = F_spring + F_pot
        v_half = vk + 0.5*dt*F/m
        q_new = qk + dt*v_half
        q_plus2 = np.roll(q_new, -1); q_minus2 = np.roll(q_new, 1)
        F_spring_new = -m*(omegaP**2) * (2.0*q_new - q_plus2 - q_minus2)
        F_pot_new = -force_func(q_new, *c_params)
        F_new = F_spring_new + F_pot_new
        v_new = v_half + 0.5*dt*F_new/m
        q_series[t] = q_new; v_series[t] = v_new
        qk, vk = q_new, v_new
    return q_series, v_series

# ----------------- Worker state ---------------------
class Runner:
    def __init__(self):
        # parameters
        self.beta = 1.0
        self.N = 80
        self.c1, self.c2, self.c3, self.c4 = 0.0, 0.5, 0.0, 0.0
        self.i_order, self.j_order = 1, 1
        self.dt = 0.05
        self.t_end = 20.0
        self.m = 1.0
        self.delay_ms = 4     # visual pacing (0.5–5 ms slider)
        self.n_max = 64       # for exact()

        # internal
        self._lock = threading.Lock()
        self._stop = False
        self.running = False   # start stopped
        self._needs_reset = False  # initial config already generated below

        # initial configuration: cosine ring + noise
        n0 = self.N
        k0 = np.arange(n0)
        self.q = 0.1 + np.cos(2.0*np.pi*k0/n0)
        beta_n0 = self.beta / n0
        self.v = np.random.default_rng().normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n0*self.m)), size=n0)
        self.latest_q = self.q.copy()

        # CF histories
        self.history_mean = None
        self.history_count = 0

        # current run curves
        self.cur_t = None
        self.cur_cf = None

        # exact curve (compute for initial params so line shows at launch)
        tt, Ct = exact(self.n_max, self.beta, [self.c1, self.c2, self.c3, self.c4],
                       order_i=self.i_order, order_j=self.j_order,
                       t_end=self.t_end, delta_t=self.dt)
        self.exact_t = tt
        self.exact_curve = Ct

    def set_params(self, beta, N, c1, c2, c3, c4, i_ord, j_ord, t_end, dt, delay_ms, n_max):
        with self._lock:
            self.beta = float(beta)
            self.N = int(max(4, 2*(N//2)))
            self.c1, self.c2, self.c3, self.c4 = float(c1), float(c2), float(c3), float(c4)
            self.i_order, self.j_order = int(i_ord), int(j_ord)
            self.t_end = float(t_end)
            self.dt = float(dt)
            self.delay_ms = int(delay_ms)
            self.n_max = int(n_max)
            # do not force reset here; user controls Reset explicitly

    def stop(self):
        with self._lock:
            self._stop = True

    def run(self):
        rng = np.random.default_rng()
        while True:
            with self._lock:
                if self._stop:
                    return
                if not self.running:
                    pass_flag = True
                else:
                    pass_flag = False
                paused = not self.running
                needs_reset = self._needs_reset
                beta = self.beta; N = self.N; mloc = self.m
                c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4
                i_ord, j_ord = self.i_order, self.j_order
                t_end = self.t_end; dt = self.dt; delay_ms = self.delay_ms
                n_max = self.n_max

            if paused:
                time.sleep(0.05)
                continue

            # Prepare state if needed
            if needs_reset or self.q is None or self.v is None or len(self.q) != N:
                n = N; beta_n = beta / n
                k = np.arange(n)
                self.q = np.cos(2.0*np.pi*k/n) + rng.normal(0.0, 0.1, size=n)
                self.v = rng.normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n*mloc)), size=n)
                self.latest_q = self.q.copy()
                self.cur_t = None; self.cur_cf = None
                self._needs_reset = False

            # precompute exact each run with current parameters
            n = len(self.q); beta_n = beta / n
            n_steps = int(max(1, round(t_end / dt)))
            coeffs = [c1, c2, c3, c4]
            tt, Ct = exact(n_max, beta, coeffs, order_i=i_ord, order_j=j_ord, t_end=t_end, delta_t=dt)
            self.exact_t = tt
            self.exact_curve = Ct

            # start a run
            x0 = float(np.mean(self.q)); x0i = x0**i_ord
            t_list = [0.0]; cf_list = [x0i * (x0**j_ord)]

            for step in range(1, n_steps+1):
                # exact-internal-mode propagation (normal modes), with external-force half-kicks
                # spring constant for inter-bead coupling: k_eff = m / beta_n^2
                k_eff = mloc / max(1e-16, beta_n*beta_n)
                omega = normal_mode_frequencies(k_eff, n, mloc)
                # half-kick with external force
                f = force_value([c1, c2, c3, c4], self.q)
                v_half = self.v + 0.5 * dt * f / mloc
                # exact propagation of internal modes for dt
                x_new, v_mode = mode_propagation(self.q, v_half, omega, dt)
                # external force at new positions
                f_new = force_value([c1, c2, c3, c4], x_new)
                # second half-kick
                v_new = v_mode + 0.5 * dt * f_new / mloc
                self.q, self.v = x_new, v_new
                self.latest_q = self.q.copy()

                xc = float(np.mean(self.q))
                cf_list.append(x0i * (xc**j_ord))
                t_list.append(step*dt)

                self.cur_t = np.array(t_list, float)
                self.cur_cf = np.array(cf_list, float)

                time.sleep(max(0.0, 0.001*delay_ms))

                with self._lock:
                    if not self.running or self._stop:
                        break

            # update running mean at end of run
            L = len(cf_list)
            cf_arr = np.array(cf_list, float)
            if self.history_mean is None or self.history_mean.size != L:
                self.history_mean = cf_arr.copy()
                self.history_count = 1
            else:
                self.history_count += 1
                w = 1.0 / self.history_count
                self.history_mean = (1.0 - w) * self.history_mean + w * cf_arr

            # keep end config; redraw velocities for next run
            self.v = rng.normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n*mloc)), size=n)

def main():
    r = Runner()

    # Figure 1: polymer configuration + potential
    fig1, ax1 = plt.subplots(figsize=(7,4))
    line_V, = ax1.plot([], [], lw=1.5, label=r"$V(x)$")
    ring_line, = ax1.plot([], [], "--", lw=1.2, label="bead ring")
    ring_scatter = ax1.plot([], [], "o", ms=4)[0]
    vline = ax1.axvline(0.0, color="k", lw=1.0)
    ax1.set_xlabel(r"$x$"); ax1.set_ylabel(r"$V(x)$"); ax1.set_title("Potential with bead ring")
    ax1.legend(loc="best"); ax1.grid(True)

    # Figure 2: correlation curves
    fig2, ax2 = plt.subplots(figsize=(7,4))
    line_exact, = ax2.plot([], [], lw=1.5, label="Exact")
    line_mean,  = ax2.plot([], [], lw=1.5, label="History mean")
    line_cur,   = ax2.plot([], [], lw=1.5, label="Current run")
    samples_text = ax2.text(0.02, 0.95, r"$N = 0$", transform=ax2.transAxes, va='top')
    ax2.set_xlabel(r"$t$"); ax2.set_ylabel(r"$C_{ij}(t)$"); ax2.set_title("Correlation function")
    ax2.legend(loc="best"); ax2.grid(True)

    # Figure 3: controls
    fig3, ax3 = plt.subplots(figsize=(10,6)); ax3.axis('off'); axcolor='lightgoldenrodyellow'
    ax_beta=fig3.add_axes([0.10,0.80,0.35,0.05], facecolor=axcolor)
    ax_N   =fig3.add_axes([0.10,0.72,0.35,0.05], facecolor=axcolor)
    ax_c1  =fig3.add_axes([0.10,0.64,0.35,0.05], facecolor=axcolor)
    ax_c2  =fig3.add_axes([0.10,0.56,0.35,0.05], facecolor=axcolor)
    ax_c3  =fig3.add_axes([0.10,0.48,0.35,0.05], facecolor=axcolor)
    ax_c4  =fig3.add_axes([0.10,0.40,0.35,0.05], facecolor=axcolor)
    ax_i   =fig3.add_axes([0.55,0.80,0.35,0.05], facecolor=axcolor)
    ax_j   =fig3.add_axes([0.55,0.72,0.35,0.05], facecolor=axcolor)
    ax_tend=fig3.add_axes([0.55,0.64,0.35,0.05], facecolor=axcolor)
    ax_dt  =fig3.add_axes([0.55,0.56,0.35,0.05], facecolor=axcolor)
    ax_delay=fig3.add_axes([0.55,0.48,0.35,0.05], facecolor=axcolor)
    ax_nmax=fig3.add_axes([0.10,0.32,0.35,0.05], facecolor=axcolor)

    s_beta = Slider(ax_beta, r"$\beta$",                    0.1,    20.0,   valinit=r.beta,     valstep=0.1)
    s_N    = Slider(ax_N,    r"$N$",                        4,      256,    valinit=r.N,        valstep=2)
    s_c1   = Slider(ax_c1,   r"$c_1$ ($x$)",                -2.0,   2.0,    valinit=r.c1,       valstep=0.01)
    s_c2   = Slider(ax_c2,   r"$c_2$ ($x^2$)",              -2.0,   2.0,    valinit=r.c2,       valstep=0.01)
    s_c3   = Slider(ax_c3,   r"$c_3$ ($x^3$)",              -2.0,   2.0,    valinit=r.c3,       valstep=0.01)
    s_c4   = Slider(ax_c4,   r"$c_4$ ($x^4$)",              0.0,    2.0,    valinit=r.c4,       valstep=0.01)
    s_i    = Slider(ax_i,    r"$i$ order",                  0,      4,      valinit=r.i_order,  valstep=1)
    s_j    = Slider(ax_j,    r"$j$ order",                  0,      4,      valinit=r.j_order,  valstep=1)
    s_tend = Slider(ax_tend, r"$t_{\mathrm{end}}$",         0.1,    200.0,  valinit=r.t_end,    valstep=0.1)
    s_dt   = Slider(ax_dt,   r"$\delta t$ (verlet)",        1e-4,   0.2,    valinit=r.dt,       valstep=1e-4)
    s_delay= Slider(ax_delay,r"Delay (ms)",                 0.5,    10,     valinit=r.delay_ms, valstep=0.1)
    s_nmax = Slider(ax_nmax, r"$n_{\mathrm{max}}$ (exact)", 8,    256,    valinit=r.n_max,    valstep=1)

    ax_apply=fig3.add_axes([0.10,0.22,0.18,0.06]); b_apply=Button(ax_apply,"Apply Params")
    ax_toggle=fig3.add_axes([0.32,0.22,0.18,0.06]); b_toggle=Button(ax_toggle,"Start")
    ax_reset=fig3.add_axes([0.54,0.22,0.18,0.06]); b_reset=Button(ax_reset,"Reset (new x_init)")

    def on_apply(evt):
        r.set_params(s_beta.val, int(s_N.val), s_c1.val, s_c2.val, s_c3.val, s_c4.val,
                     int(s_i.val), int(s_j.val), s_tend.val, s_dt.val, int(s_delay.val), int(s_nmax.val))
        # recompute exact with current coefficients and settings
        coeffs = [r.c1, r.c2, r.c3, r.c4]
        tt, Ct = exact(r.n_max, r.beta, coeffs, order_i=r.i_order, order_j=r.j_order, t_end=r.t_end, delta_t=r.dt)
        r.exact_t = tt
        r.exact_curve = Ct

    def on_toggle(evt):
        with r._lock:
            r.running = not r.running
        b_toggle.label.set_text("Pause" if r.running else "Start")

    def on_reset(evt):
        with r._lock:
            # Clear histories & current curve; recompute exact; request new init config
            r.history_mean = None
            r.history_count = 0
            r.cur_t = None
            r.cur_cf = None
            # recompute exact on current settings
            coeffs = [r.c1, r.c2, r.c3, r.c4]
            tt, Ct = exact(r.n_max, r.beta, coeffs, order_i=r.i_order, order_j=r.j_order, t_end=r.t_end, delta_t=r.dt)
            r.exact_t = tt
            r.exact_curve = Ct
            # ask worker to regenerate cosine+noise configuration
            r._needs_reset = True

    b_apply.on_clicked(on_apply); b_toggle.on_clicked(on_toggle); b_reset.on_clicked(on_reset)

    # Worker
    th = threading.Thread(target=r.run, daemon=True); th.start()

    # Refresh
    def refresh(_evt):
        # Figure 1
        xmin,xmax=-4.0,4.0
        xs=np.linspace(xmin,xmax,800)
        Vx=V_poly(xs, r.c1, r.c2, r.c3, r.c4); line_V.set_data(xs,Vx); ax1.set_xlim(xmin,xmax)
        q=r.latest_q
        if q is not None:
            n=q.size; k=np.arange(n)
            y0=float(np.nanmax(Vx))*1.05 if np.any(np.isfinite(Vx)) else 0.0
            amp=0.20*(np.nanmax(Vx)-np.nanmin(Vx)+1e-12)
            xk=q.astype(float); yk=y0+amp*np.sin(2.0*np.pi*k/n)
            ring_line.set_data(np.r_[xk,xk[0]], np.r_[yk,yk[0]]); ring_scatter.set_data(xk, yk)
            vline.set_xdata([float(np.mean(q))])
            y_min=min(np.nanmin(Vx), y0-2.2*amp); y_max=max(np.nanmax(Vx), y0+2.2*amp)
            if np.isfinite(y_min) and np.isfinite(y_max): ax1.set_ylim(y_min,y_max)

        # Figure 2
        if r.exact_t is not None and r.exact_curve is not None:
            line_exact.set_data(r.exact_t, r.exact_curve)
        else:
            line_exact.set_data([], [])
        if r.history_mean is not None:
            tt = np.linspace(0.0, r.dt*(len(r.history_mean)-1), len(r.history_mean))
            line_mean.set_data(tt, r.history_mean)
            N_hist = r.history_count  # number of CF samples (runs) accumulated
        else:
            line_mean.set_data([], [])
            N_hist = 0
        if r.cur_t is not None and r.cur_cf is not None:
            line_cur.set_data(r.cur_t, r.cur_cf)
        else:
            line_cur.set_data([], [])

        # Limits
        xs_all=[]; ys_all=[]
        for ln in (line_exact, line_mean, line_cur):
            xd=np.asarray(ln.get_xdata()); yd=np.asarray(ln.get_ydata())
            if xd.size>0: xs_all.append(xd)
            if yd.size>0: ys_all.append(yd)
        if xs_all:
            xa=np.concatenate(xs_all); xmin2=float(np.nanmin(xa)); xmax2=float(np.nanmax(xa))
        else:
            xmin2,xmax2=0.0, r.t_end
        if ys_all:
            ya=np.concatenate(ys_all); ymin2=float(np.nanmin(ya)); ymax2=float(np.nanmax(ya))
        else:
            ymin2,ymax2=-1.0,1.0
        if xmin2==xmax2: xmax2=xmin2+1e-6
        if ymin2==ymax2: ymax2=ymin2+1e-6
        pad=0.05*(abs(ymin2)+abs(ymax2)); ax2.set_xlim(xmin2,xmax2); ax2.set_ylim(ymin2-pad, ymax2+pad)

        samples_text.set_text(fr"$N = {N_hist}$")

        fig1.canvas.draw_idle(); fig2.canvas.draw_idle()

    t1=fig1.canvas.new_timer(interval=60); t1.add_callback(refresh,None); t1.start()
    t2=fig2.canvas.new_timer(interval=80); t2.add_callback(refresh,None); t2.start()

    def on_close(_e): r.stop()
    for fig in (fig1,fig2,fig3):
        fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

if __name__ == "__main__":
    main()
