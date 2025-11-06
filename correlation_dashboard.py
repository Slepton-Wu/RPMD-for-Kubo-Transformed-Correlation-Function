import threading, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from numpy.linalg import eigh
from numba import njit

# need to install rocket fft (via pip install rocket-fft) so that fft can be performed in numba. Otherwise, you need to remove all the numba stuffs -- but the code will run slower.

# ---------------- Polynomial potential ----------------
@njit(nopython=True, fastmath=True)
def V_poly(x, c1, c2, c3, c4):
    return c1*x + c2*(x**2) + c3*(x**3) + c4*(x**4)

@njit(nopython=True, fastmath=True)
def force_value(coeffs, x):
    c1, c2, c3, c4 = coeffs
    return -(c1 + 2.0*c2*x + 3.0*c3*(x**2) + 4.0*c4*(x**3))

# ---------------- Exact quantum correlation ----------------
hbar = 1.0
m_exact = 1.0  # mass used inside exact()

def exact(n_max, beta, coeffs, order_i=1, order_j=1, t_end=20, delta_t=0.02):
    max_order = int(max(order_i, order_j, len(coeffs)))
    dim = n_max + max_order
    a = np.zeros((dim, dim))
    for n in range(1, dim):
        a[n-1, n] = np.sqrt(n)
    a_dag = a.T
    x = np.sqrt(hbar/(2*m_exact)) * (a + a_dag)
    x_powers = [x]
    for i in range(0, max_order-1):
        x_powers.append(x_powers[i] @ x)
    V = np.zeros((dim, dim))
    for i in range(0, len(coeffs)):
        V += coeffs[i] * x_powers[i]
    P = np.sqrt(m_exact * hbar / 2) * (a_dag - a)
    T = -(P @ P) / (2 * m_exact)
    H = (T + V)[:n_max, :n_max]
    E, psi = eigh(H)
    Boltz = np.exp(-beta * E); Z = np.sum(Boltz)
    A = x_powers[order_i-1][:n_max, :n_max]; A_mel = psi.T @ A @ psi
    B = x_powers[order_j-1][:n_max, :n_max]; B_mel = psi.T @ B @ psi
    Em, En = np.meshgrid(E, E, indexing='ij'); dE = Em - En; eps = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        W = np.where(np.abs(dE) < eps, Boltz[:, None], (Boltz[None, :] - Boltz[:, None]) / (beta * dE))
    AB = A_mel * B_mel.T
    times = np.arange(0, t_end, delta_t)
    phases = np.exp(1j * dE[None, :, :] * times[:, None, None] / hbar)
    C_t = np.einsum('ijk,jk,jk->i', phases, W, AB).real / Z
    return times, C_t

# ---------------- Normal-mode helpers ----------------
@njit(nopython=True, fastmath=True)
def normal_mode_frequencies(k_eff, n, m):
    k = np.arange(n)
    return np.sqrt((2.0 - 2.0*np.cos(2.0*np.pi*k/n)) * (k_eff / m))

@njit(nopython=True, fastmath=True)
def mode_propagation(x, v, omega, delta_t):
    #Transform to normal coordinates
    Xk = np.fft.fft(x)
    Vk = np.fft.fft(v)

    Xk_next = np.empty_like(Xk)
    Vk_next = np.empty_like(Vk)

    #Evolve the omega=0 mode
    Xk_next[0] = Xk[0] + Vk[0] * delta_t
    Vk_next[0] = Vk[0]

    # Evolve the omgea>0 mode
    omega_pos = omega[1:]
    coswt = np.cos(omega_pos*delta_t)
    sinwt = np.sin(omega_pos*delta_t)
    Xk_pos = Xk[1:]
    Vk_pos = Vk[1:]
    Xk_next[1:] = Xk_pos * coswt + Vk_pos * (sinwt / omega_pos)
    Vk_next[1:] = Vk_pos * coswt - Xk_pos * (omega_pos * sinwt)
    
    # Transform back to real coordinates
    x_next = np.fft.ifft(Xk_next, axis=0).real
    v_next = np.fft.ifft(Vk_next, axis=0).real
    return x_next, v_next

@njit(nopython=True, fastmath=True)
def propagate(q, v, coeffs, dt, m, omega):
    f = force_value(coeffs, q)
    v_half = v + 0.5 * dt * f / m
    x_new, v_mode = mode_propagation(q, v_half, omega, dt)
    f_new = force_value(coeffs, x_new)
    v = v_mode + 0.5 * dt * f_new / m
    q = x_new
    return q, v

# ---------------- Runner ----------------
class Runner:
    def __init__(self):
        # Params
        self.beta = 1.0
        self.N = 80
        self.c1, self.c2, self.c3, self.c4 = 0.0, 0.5, 0.0, 0.0
        self.i_order, self.j_order = 1, 1
        self.dt = 0.05
        self.t_end = 20.0
        self.m = 1.0
        self.delay_ms = 4
        self.n_max = 64

        # State
        self._lock = threading.Lock()
        self._stop = False
        self.running = False
        self._needs_reset = False
        self._thermalising = False

        # Initial config: cosine ring
        n0 = self.N
        k0 = np.arange(n0)
        self.q = np.cos(2.0*np.pi*k0/n0)
        beta_n0 = self.beta / n0
        self.v = np.random.default_rng().normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n0*self.m)), size=n0)
        self.latest_q = self.q.copy()

        # Correlators
        self.history_mean = None
        self.history_count = 0
        self.cur_t = None
        self.cur_cf = None

        # Exact curve for initial params
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
            self.delay_ms = float(delay_ms)
            self.n_max = int(n_max)

    def stop(self):
        with self._lock:
            self._stop = True

    def thermalise(self, cycles=10):
        # Run several fast cycles without recording correlation or updating plots live.
        with self._lock:
            if self._thermalising:
                return
            self._thermalising = True
            was_running = self.running
            self.running = False
            beta = self.beta; mloc = self.m
            c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4
            t_end = self.t_end; dt = self.dt
            if self.q is None or self.v is None or len(self.q) != self.N:
                n = self.N; beta_n = beta / n
                k = np.arange(n)
                self.q = np.cos(2.0*np.pi*k/n)
                self.v = np.random.default_rng().normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n*mloc)), size=n)
            q = self.q.copy(); v = self.v.copy()
            n = len(q); beta_n = beta / n
        rng = np.random.default_rng()
        n_steps = int(max(1, round(t_end / dt)))
        k_eff = mloc / max(1e-16, beta_n*beta_n)
        omega = normal_mode_frequencies(k_eff, n, mloc)
        coeffs = [c1, c2, c3, c4]

        for cyc in range(cycles):
            v = rng.normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n*mloc)), size=n)
            for _ in range(n_steps):
                q, v = propagate(q, v, coeffs, dt, mloc, omega)
        with self._lock:
            self.q = q; self.v = v; self.latest_q = q.copy()
            self._thermalising = False
            self.running = was_running

    def run(self):
        rng = np.random.default_rng()
        while True:
            with self._lock:
                if self._stop:
                    return
                if not self.running:
                    time.sleep(0.05)
                    continue
                needs_reset = self._needs_reset
                beta = self.beta; N = self.N; mloc = self.m
                c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4
                i_ord, j_ord = self.i_order, self.j_order
                t_end = self.t_end; dt = self.dt; delay_ms = self.delay_ms
                n_max = self.n_max

            if needs_reset or self.q is None or self.v is None or len(self.q) != N:
                n = N; beta_n = beta / n
                k = np.arange(n)
                self.q = np.cos(2.0*np.pi*k/n)
                self.v = rng.normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n*mloc)), size=n)
                self.latest_q = self.q.copy()
                self.cur_t = None; self.cur_cf = None
                self._needs_reset = False

            # Precompute exact each run
            coeffs = [c1, c2, c3, c4]
            tt, Ct = exact(n_max, beta, coeffs, order_i=i_ord, order_j=j_ord, t_end=t_end, delta_t=dt)
            self.exact_t = tt
            self.exact_curve = Ct

            # One run
            n = len(self.q); beta_n = beta / n
            n_steps = int(max(1, round(t_end / dt)))
            x0 = float(np.mean(self.q)); x0i = x0**i_ord
            t_list = [0.0]; cf_list = [x0i * (x0**j_ord)]
            k_eff = mloc / max(1e-16, beta_n*beta_n)
            omega = normal_mode_frequencies(k_eff, n, mloc)

            for step in range(1, n_steps+1):
                self.q, self.v = propagate(self.q, self.v, coeffs, dt, mloc, omega)
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

            # Update history mean
            L = len(cf_list)
            cf_arr = np.array(cf_list, float)
            if self.history_mean is None or self.history_mean.size != L:
                self.history_mean = cf_arr.copy()
                self.history_count = 1
            else:
                self.history_count += 1
                w = 1.0 / self.history_count
                self.history_mean = (1.0 - w) * self.history_mean + w * cf_arr

            # Refresh velocities for next run
            self.v = rng.normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n*mloc)), size=n)

# ---------------- Main / UI ----------------
def main():
    r = Runner()

    # Figure 1: polymer config + potential
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
    s_delay= Slider(ax_delay,r"Delay (ms)",                 0,      10,     valinit=r.delay_ms, valstep=0.1)
    s_nmax = Slider(ax_nmax, r"$n_{\mathrm{max}}$ (exact)", 8,      256,    valinit=r.n_max,    valstep=1)

    ax_apply=fig3.add_axes([0.08,0.22,0.16,0.06]); b_apply=Button(ax_apply,"Apply")
    ax_toggle=fig3.add_axes([0.26,0.22,0.16,0.06]); b_toggle=Button(ax_toggle,"Start")
    ax_reset=fig3.add_axes([0.44,0.22,0.18,0.06]); b_reset=Button(ax_reset,"Reset")
    ax_therm=fig3.add_axes([0.64,0.22,0.20,0.06]); b_therm=Button(ax_therm,"Thermalise")

    def on_apply(evt):
        r.set_params(s_beta.val, int(s_N.val), s_c1.val, s_c2.val, s_c3.val, s_c4.val,
                     int(s_i.val), int(s_j.val), s_tend.val, s_dt.val, s_delay.val, int(s_nmax.val))
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
            r.history_mean = None
            r.history_count = 0
            r.cur_t = None
            r.cur_cf = None
            coeffs = [r.c1, r.c2, r.c3, r.c4]
            tt, Ct = exact(r.n_max, r.beta, coeffs, order_i=r.i_order, order_j=r.j_order, t_end=r.t_end, delta_t=r.dt)
            r.exact_t = tt
            r.exact_curve = Ct
            # regenerate a fresh cosine+noise configuration immediately
            n = r.N
            k = np.arange(n)
            r.q = np.cos(2.0*np.pi*k/n)
            beta_n = r.beta / n
            r.v = np.random.default_rng().normal(0.0, 1.0/np.sqrt(max(1e-16, beta_n*r.m)), size=n)
            r.latest_q = r.q.copy()
            r._needs_reset = False

    def on_thermalise(evt):
        def _run():
            r.thermalise(cycles=10)
        th2 = threading.Thread(target=_run, daemon=True)
        th2.start()

    b_apply.on_clicked(on_apply); b_toggle.on_clicked(on_toggle); b_reset.on_clicked(on_reset); b_therm.on_clicked(on_thermalise)

    # Worker
    th = threading.Thread(target=r.run, daemon=True); th.start()

    # Refresh
    def refresh(_evt):
        # Figure 1: potential + ring
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

        # Figure 2: correlation
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
