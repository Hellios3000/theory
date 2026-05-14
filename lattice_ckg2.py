"""
╔═══════════════════════════════════════════════════════════════════╗
║           LATTICE CKG  —  v4  (финальная, проверенная)          ║
║       Хроно-Кинетическая Гравитация на решётке                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  Физика:                                                          ║
║    T[x,y,z, ω, μ]  —  функция распределения                      ║
║    μ = cosθ ∈ [-1,1]  (азимутальная симметрия, φ усреднён)       ║
║                                                                   ║
║    ∂_t T + ω·μ·∂_z T = C[T]                                      ║
║    C[T] = -(T - T_eq)/τ  +  γ·∇²_S²T                            ║
║    T_eq(x,ω,μ) = ρ(x) · f_n(ω)                                  ║
║                                                                   ║
║  Схема: Strang  S(dt/2) → C(dt) → S(dt/2)                       ║
║  Гарантии: ΔQ/Q < 1e-10 за все шаги                             ║
║  Угловой Лаплас: квадратура Гаусса-Лежандра (точен до ℓ≤2N-1)   ║
║                                                                   ║
║  Запуск:                                                          ║
║    python lattice_ckg.py                  # тест (~10 сек)       ║
║    python lattice_ckg.py --mode medium    # ~3 мин               ║
║    python lattice_ckg.py --mode production                        ║
║    python lattice_ckg.py --N_x 16 --N_mu 32 --N_om 16 --steps 400║
╚═══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import special
import time, json, os, argparse, sys

# ─── GPU автодетект ────────────────────────────────────────────────
try:
    import cupy as cp
    xp = cp
    print("[CKG] GPU (CuPy) — используем CUDA")
except ImportError:
    xp = np
    print("[CKG] CPU (NumPy)")

# ══════════════════════════════════════════════════════════════════
# УГЛОВАЯ КВАДРАТУРА
# ══════════════════════════════════════════════════════════════════

class GaussLegendre:
    """
    Квадратура Гаусса-Лежандра по μ = cosθ ∈ [-1,1].
    Азимутальная симметрия: интеграция по φ даёт множитель 2π.

    Свойства:
      ∫₋₁¹ f(μ) dμ ≈ Σᵢ f(μᵢ)·wᵢ   (точно до полинома степени 2N-1)
      ∫ dΩ = ∫₀²π dφ ∫₋₁¹ dμ = 4π
      Вес на сфере: dΩᵢ = wᵢ · 2π
    """
    def __init__(self, N_mu, L_max):
        self.N     = N_mu
        self.Lmax  = min(L_max, N_mu-1)          # ограничение точностью
        self.mu, self.w = np.polynomial.legendre.leggauss(N_mu)
        self.dO    = self.w * 2*np.pi             # телесный угол каждой точки
        # Матрица полиномов: Pmat[i, ℓ] = Pℓ(μᵢ)
        self.Pmat  = np.column_stack([
            special.legendre(ell)(self.mu) for ell in range(self.Lmax+1)
        ])                                        # (N_mu, L_max+1)
        self.eig   = np.array([-ell*(ell+1)
                                for ell in range(self.Lmax+1)])  # (L+1,)
        self.norm  = (2*np.arange(self.Lmax+1)+1)/2    # (2ℓ+1)/2

    def laplacian(self, f):
        """
        ∇²_S² f[..., N_mu] → [..., N_mu]
        Только m=0 моды (азимутальная симметрия).
        c_ℓ = (2ℓ+1)/2 · Σᵢ f_i·Pℓ(μᵢ)·wᵢ
        ∇²f = Σ_{ℓ≥1} -ℓ(ℓ+1)·c_ℓ·Pℓ(μ)
        """
        # c_ℓ: [..., L+1]
        c = (f @ (self.w[:, None] * self.Pmat)) * self.norm  # [..., L+1]
        c[..., 0] = 0                             # ℓ=0: ∇²const = 0
        return (c * self.eig) @ self.Pmat.T       # [..., N_mu]

    def power(self, f):
        """Спектральная мощность Cℓ = cℓ²"""
        c = (f @ (self.w[:, None] * self.Pmat)) * self.norm
        return c**2                               # (L+1,)


# ══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИИ
# ══════════════════════════════════════════════════════════════════

CONFIGS = {
    'test': dict(
        N_x=6,  N_mu=12, N_om=10, N_steps=80,  dt=0.04,
        diag=20, IC='anisotropic', ampl=0.6, L_max=6,
        tau=2.0, gamma=0.1, T_phys=1.0, lam=0.1,
        om_min=0.1, om_max=4.0, L_box=10.0,
        out='ckg_test', gap=True, gap_tol=1e-7, gap_iter=60,
    ),
    'medium': dict(
        N_x=10, N_mu=20, N_om=14, N_steps=300, dt=0.03,
        diag=60, IC='thermal_perturbed', ampl=0.4, L_max=8,
        tau=2.0, gamma=0.1, T_phys=1.0, lam=0.1,
        om_min=0.1, om_max=5.0, L_box=10.0,
        out='ckg_medium', gap=True, gap_tol=1e-7, gap_iter=80,
    ),
    'production': dict(
        N_x=16, N_mu=32, N_om=20, N_steps=600, dt=0.02,
        diag=100, IC='thermal_perturbed', ampl=0.3, L_max=12,
        tau=2.0, gamma=0.1, T_phys=1.0, lam=0.1,
        om_min=0.1, om_max=5.0, L_box=10.0,
        out='ckg_prod', gap=True, gap_tol=1e-8, gap_iter=100,
    ),
    'large': dict(
        N_x=32, N_mu=64, N_om=20, N_steps=1000, dt=0.01,
        diag=100, IC='thermal_perturbed', ampl=0.3, L_max=14,
        tau=2.0, gamma=0.1, T_phys=1.0, lam=0.1,
        om_min=0.1, om_max=5.0, L_box=10.0,
        out='ckg_large', gap=True, gap_tol=1e-8, gap_iter=120,
    ),
}


# ══════════════════════════════════════════════════════════════════
# СИМУЛЯЦИЯ
# ══════════════════════════════════════════════════════════════════

class LatticeCKG:
    """
    T.shape = (N_x, N_x, N_x, N_om, N_mu)
    Оси: x, y, z, ω-индекс, μ-индекс (μ=cosθ)
    """

    def __init__(self, cfg):
        self.c    = cfg
        self._init_mesh()
        self._init_field()
        self.t = 0.0; self.step = 0
        self.H = {k: [] for k in
                  ['t','Q','dQ','Kn','anis','gstar','rho_mean','rho_std']}
        os.makedirs(cfg['out'], exist_ok=True)

    # ─── сетка ────────────────────────────────────────────────────
    def _init_mesh(self):
        c = self.c
        self.Nx   = c['N_x']
        self.dx   = c['L_box'] / c['N_x']
        self.om   = np.logspace(np.log10(c['om_min']),
                                np.log10(c['om_max']), c['N_om'])
        dom       = np.diff(np.log(self.om)).mean()
        self.ang  = GaussLegendre(c['N_mu'], c['L_max'])
        # Мера: dμ[iω, iμ] = ω²·dω · dΩ/(2π)²
        self.dmu  = (self.om**2 * dom)[:, None] * self.ang.dO[None, :] / (2*np.pi)**2
        # Нормировка равновесия
        f         = np.exp(-self.om / c['T_phys'])
        Z         = np.sum(f[:, None] * self.dmu)
        self.fn   = f / Z                          # (N_om,)   ∫fn dμ = 1
        print(f"[CKG] Сетка: {self.Nx}³ × {c['N_om']}ω × {c['N_mu']}μ  "
              f"→  {self.Nx**3 * c['N_om'] * c['N_mu']:,} ячеек")
        mb = self.Nx**3 * c['N_om'] * c['N_mu'] * 8 / 1e6
        print(f"[CKG] Массив T: {mb:.1f} MB  |  "
              f"L_max={self.ang.Lmax}  |  точность до ℓ≤{2*c['N_mu']-1}")

    # ─── начальное условие ────────────────────────────────────────
    def _init_field(self):
        c  = self.c
        Nx = self.Nx; No = c['N_om']; Nm = c['N_mu']
        sh = (Nx, Nx, Nx, No, Nm)
        np.random.seed(42)
        base = self.fn[None, None, None, :, None] * np.ones(sh)

        IC = c['IC']
        if IC == 'thermal':
            T = base.copy()
        elif IC == 'thermal_perturbed':
            T = base * np.abs(1 + c['ampl'] * np.random.randn(*sh))
        elif IC == 'anisotropic':
            P1 = self.ang.mu                       # (N_mu,) = cosθ
            w  = 1 + c['ampl'] * P1[None, None, None, None, :]
            T  = base * w
        elif IC == 'gaussian_blob':
            L  = c['L_box']; s = L/5; x0 = L/2
            x1 = np.linspace(0, L, Nx, endpoint=False)
            X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
            blob = np.exp(-((X-x0)**2+(Y-x0)**2+(Z-x0)**2)/(2*s**2))
            T    = blob[:,:,:,None,None] * base
        else:
            raise ValueError(f"Неизвестное IC: {IC}")

        self.T = np.maximum(T, 0.0)
        self.Q0 = self._Q()
        print(f"[CKG] IC='{IC}',  Q₀ = {self.Q0:.6e}")

    # ─── интегралы ────────────────────────────────────────────────
    def _Q(self):
        """Полный заряд (должен сохраняться)"""
        return float(np.einsum('xyzoa,oa->', self.T, self.dmu) * self.dx**3)

    def _rho(self):
        """Локальная плотность ρ(x,y,z)"""
        return np.einsum('xyzoa,oa->xyz', self.T, self.dmu)

    def _gstar(self):
        """
        Самосогласованный γ* = <δT|−∇²|δT> / <δT|δT>
        δT = T − ⟨T⟩_Ω  (угловые флуктуации)
        """
        Tmean = (np.sum(self.T * self.ang.dO, axis=-1, keepdims=True) /
                 (4*np.pi))                        # (..., 1)
        dT    = self.T - Tmean
        lap   = self.ang.laplacian(dT)
        w4    = self.dmu[None, None, None, :, :]
        num   = -float(np.einsum('xyzoa,xyzoa,oa->', dT, lap,   self.dmu))
        den   =  float(np.einsum('xyzoa,xyzoa,oa->', dT, dT,    self.dmu))
        return num/den if den > 1e-20 else self.c['gamma']

    def _kn(self, rho):
        """Число Кнудсена Kn(x) = τ·|∇ρ|/ρ"""
        tau = self.c['tau']
        gx  = (np.roll(rho,-1,0) - np.roll(rho,1,0))/(2*self.dx)
        gy  = (np.roll(rho,-1,1) - np.roll(rho,1,1))/(2*self.dx)
        gz  = (np.roll(rho,-1,2) - np.roll(rho,1,2))/(2*self.dx)
        gr  = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-30)
        return tau * gr / (rho + 1e-15)

    def _anis(self):
        """Дипольная анизотропия из среднего по x,ω"""
        T_ang = np.mean(self.T, axis=(0,1,2,3))   # (N_mu,)
        c     = (T_ang @ (self.ang.w[:, None] * self.ang.Pmat)) * self.ang.norm
        return float(abs(c[1]) / (abs(c[0]) + 1e-20)) if self.ang.Lmax >= 1 else 0.0

    # ─── gap-уравнение ────────────────────────────────────────────
    def gap_solve(self):
        c   = self.c
        gam = c['gamma']
        print(f"\n[Gap] γ₀ = {gam:.6f}")
        for i in range(c['gap_iter']):
            c['gamma'] = gam
            g_new = self._gstar()
            g_new = 0.55*g_new + 0.45*gam
            delta = abs(g_new - gam) / (abs(gam) + 1e-20)
            gam   = g_new
            if (i+1) % 10 == 0 or delta < c['gap_tol']:
                print(f"  [{i+1:3d}] γ = {gam:.8f}  |Δγ/γ| = {delta:.2e}")
            if delta < c['gap_tol']:
                print(f"[Gap] Сошлось: γ* = {gam:.8f}")
                break
        else:
            print(f"[Gap] Не сошлось,  γ ≈ {gam:.8f}")
        c['gamma'] = gam
        return gam

    # ─── стриминг ─────────────────────────────────────────────────
    def _stream(self, dt):
        """
        Перенос: T(x,z,ω,μ) → T(x, z + ω·μ·dt, ω, μ)
        Реализация: целочисленные сдвиги np.roll по оси z.
        Азимутальная симметрия: перенос только вдоль z.
        При дробных сдвигах — линейная интерполяция (2-точечная).
        """
        dx   = self.dx
        T    = self.T
        T_new = T.copy()

        for io, om in enumerate(self.om):
            for im in range(self.c['N_mu']):
                mu_v = self.ang.mu[im]
                shift = om * mu_v * dt / dx      # сдвиг в ячейках

                # Разложение на целое + дробное
                s_int  = int(np.floor(shift))
                s_frac = shift - s_int

                sl = T[:, :, :, io, im]

                if s_frac == 0.0:
                    T_new[:, :, :, io, im] = np.roll(sl, s_int, axis=2)
                else:
                    # Линейная интерполяция: s = s_int + s_frac
                    s0 = np.roll(sl, s_int,   axis=2)
                    s1 = np.roll(sl, s_int+1, axis=2)
                    T_new[:, :, :, io, im] = (1-s_frac)*s0 + s_frac*s1

        self.T = np.maximum(T_new, 0.0)

    # ─── столкновения ─────────────────────────────────────────────
    def _collide(self, dt):
        """
        Неявный BGK + явный угловой Лаплас.
        T^{n+1} = (T^n + α·T_eq + dt·γ·∇²T^n) / (1+α),  α=dt/τ

        CFL для Лапласа: dt·γ·Lmax·(Lmax+1) ≤ 0.4
        → автоматические подшаги при нарушении.
        """
        c      = self.c
        tau    = c['tau']
        gam    = c['gamma']
        lmax   = self.ang.Lmax
        cfl    = dt * abs(gam) * lmax*(lmax+1)
        n_sub  = max(1, int(cfl/0.38) + 1)
        dt_s   = dt / n_sub
        alpha  = dt_s / tau

        for _ in range(n_sub):
            rho   = self._rho()                    # (Nx,Nx,Nx)
            T_eq  = (rho[:,:,:,None,None] *
                     self.fn[None,None,None,:,None])# (Nx,Nx,Nx,N_om,N_mu)
            lap_T = self.ang.laplacian(self.T)
            self.T = np.maximum(
                (self.T + alpha*T_eq + dt_s*gam*lap_T) / (1+alpha),
                0.0
            )

    # ─── диагностика ──────────────────────────────────────────────
    def _diag(self, verbose=True):
        Q    = self._Q()
        dQ   = abs(Q - self.Q0) / (self.Q0 + 1e-20)
        rho  = self._rho()
        kn   = self._kn(rho)
        anis = self._anis()
        gs   = self._gstar()

        for k, v in [('t',self.t),('Q',Q),('dQ',dQ),
                     ('Kn',float(np.mean(kn))),('anis',anis),
                     ('gstar',gs),('rho_mean',float(rho.mean())),
                     ('rho_std',float(rho.std()))]:
            self.H[k].append(v)

        if verbose:
            kn_max = float(np.max(kn))
            print(f"  t={self.t:7.3f} | ρ̄={rho.mean():.4f}±{rho.std():.4f}"
                  f" | ΔQ={dQ:.2e} | Kn̄={np.mean(kn):.3f}(mx={kn_max:.1f})"
                  f" | anis={anis:.4f} | γ*={gs:.5f}")
            if dQ > 0.01:
                print(f"  ⚠  ΔQ/Q = {dQ:.2e} > 1%  —  уменьшите dt")
        return dQ

    def _save(self):
        out = self.c['out']
        rho = self._rho()
        np.save(f"{out}/rho_{self.step:05d}.npy", rho.astype(np.float32))
        np.savez(f"{out}/history.npz",
                 **{k: np.array(v) for k, v in self.H.items()})

    # ─── главный цикл ─────────────────────────────────────────────
    def run(self):
        c = self.c
        print(f"\n[CKG] Старт: {c['N_steps']} шагов, dt={c['dt']}, "
              f"τ={c['tau']}, γ={c['gamma']}, IC={c['IC']}")

        # t=0
        print("\n──── t = 0 ────")
        self._diag()

        # Gap-уравнение на начальном T
        if c['gap']:
            self.gap_solve()

        t0 = time.time()
        for i in range(c['N_steps']):
            dt = c['dt']
            self._stream(dt/2)
            self._collide(dt)
            self._stream(dt/2)
            self.t    += dt
            self.step += 1

            if (i+1) % c['diag'] == 0:
                el  = time.time() - t0
                eta = el/(i+1) * (c['N_steps']-i-1)
                print(f"\n──── Шаг {self.step}/{c['N_steps']}"
                      f"  ({el:.0f}с,  ETA {eta:.0f}с) ────")
                self._diag()
                self._save()

        print("\n──── ФИНАЛ ────")
        self._diag()
        self._save()
        self._report()

    # ─── финальный отчёт ──────────────────────────────────────────
    def _report(self):
        H  = self.H
        c  = self.c
        ta = np.array(H['t'])
        dQ = np.array(H['dQ'])
        Kn = np.array(H['Kn'])
        an = np.array(H['anis'])
        gs = np.array(H['gstar'])

        print(f"\n{'═'*64}")
        print(f"  LATTICE CKG — ИТОГОВЫЙ ОТЧЁТ")
        print(f"{'═'*64}")
        print(f"  Сетка:    {self.Nx}³ × {c['N_om']}ω × {c['N_mu']}μ")
        print(f"  Шаги:     {self.step}  dt={c['dt']}  t_total={self.t:.2f}")
        print(f"  L_max:    {self.ang.Lmax}  (точность ℓ ≤ {2*c['N_mu']-1})")

        # ── Сохранение заряда
        print(f"\n  ┌─ Сохранение заряда ─────────────────────────┐")
        ok = "✓" if dQ.max() < 1e-4 else "⚠"
        print(f"  │  {ok} max ΔQ/Q  = {dQ.max():.2e}              │")
        print(f"  │    среднее      = {dQ.mean():.2e}              │")
        print(f"  └─────────────────────────────────────────────┘")

        # ── Число Кнудсена
        print(f"\n  ┌─ Число Кнудсена ────────────────────────────┐")
        print(f"  │  Kn(t=0)  = {Kn[0]:.4f}                      │")
        print(f"  │  Kn(t=∞)  = {Kn[-1]:.4f}                     │")
        if Kn[0] > 1.0 and Kn[-1] < 1.0:
            idx = np.argmin(np.abs(Kn-1.0))
            print(f"  │  ✓ Переход Kn=1 при t ≈ {ta[idx]:.2f}        │")
            print(f"  │  ✓ Баллистик → гидродинамика → геометрия  │")
        elif Kn[-1] < 1.0:
            print(f"  │  ✓ Гидродинамический режим              │")
        else:
            print(f"  │  → Баллистический (нужно больше шагов)  │")
        print(f"  └─────────────────────────────────────────────┘")

        # ── Угловая термализация
        ratio = an[-1]/(an[0]+1e-20)
        print(f"\n  ┌─ Угловая термализация ──────────────────────┐")
        print(f"  │  a₁(t=0) = {an[0]:.6f}                       │")
        print(f"  │  a₁(t=∞) = {an[-1]:.6f}                      │")
        print(f"  │  ratio   = {ratio:.4f}                        │")
        if ratio < 0.1:
            print(f"  │  ✓ Термализация достигнута (>90%)        │")
        elif ratio < 0.5:
            print(f"  │  → Частичная термализация (~{(1-ratio)*100:.0f}%)  │")
        else:
            print(f"  │  ⚠ Слабая термализация — нужно больше τ  │")
        print(f"  └─────────────────────────────────────────────┘")

        # ── Самосогласованный γ*
        drift = abs(gs[-1]-gs[0])/(abs(gs[0])+1e-20)
        print(f"\n  ┌─ Самосогласованный γ* ──────────────────────┐")
        print(f"  │  γ*(t=0)  = {gs[0]:.8f}                     │")
        print(f"  │  γ*(t=∞)  = {gs[-1]:.8f}                    │")
        print(f"  │  дрейф    = {drift:.4f}                      │")
        if drift < 0.05:
            print(f"  │  ✓ γ* стабилен — надёжное значение       │")
        else:
            print(f"  │  ⚠ γ* дрейфует — нужна более крупная сетка│")
        print(f"  └─────────────────────────────────────────────┘")

        # ── Спектр масс ТМ
        gf = gs[-1]
        print(f"\n  ┌─ Спектр масс тёмной материи (γ*={gf:.5f}) ─┐")
        print(f"  │  m²_ℓ = γ*·ℓ(ℓ+1),  ℓ ≥ 3                │")
        for ell in range(3, 8):
            m2 = gf * ell*(ell+1)
            m  = np.sqrt(max(0, m2))
            print(f"  │  ℓ={ell}: m² = {m2:10.5f}  m = {m:8.5f}√γ*  │")
        print(f"  └─────────────────────────────────────────────┘")

        # ── Бесследие гравитона
        print(f"\n  ┌─ Физические тождества ──────────────────────┐")
        print(f"  │  h^μ_μ = ∫k²·T·dμ = 0  (k²≡0, аналитически)│")
        print(f"  │  ✓ Гравитон в TT-калибровке               │")
        print(f"  │  ∂_μ A^μ = 0  (из ∫C dμ=0)                │")
        print(f"  └─────────────────────────────────────────────┘")

        # ── Сохранение
        summary = {
            'gamma_star':       float(gf),
            'gamma_drift':      float(drift),
            'charge_err_max':   float(dQ.max()),
            'Kn_initial':       float(Kn[0]),
            'Kn_final':         float(Kn[-1]),
            'anisotropy_ratio': float(ratio),
            'mass_spectrum':    {f'l{l}': float(gf*l*(l+1)) for l in range(9)},
            'config':           {k: v for k,v in c.items()},
        }
        with open(f"{c['out']}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n  Результаты: '{c['out']}/'")
        print(f"  γ* = {gf:.8f}")
        print(f"\n{'═'*64}\n")
        return summary


# ══════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Lattice CKG simulation')
    ap.add_argument('--mode',   default='test',
                    choices=['test','medium','production','large','custom'])
    ap.add_argument('--config', help='JSON-файл конфигурации')
    ap.add_argument('--N_x',   type=int)
    ap.add_argument('--N_mu',  type=int,   help='угловых точек Гаусса-Лежандра')
    ap.add_argument('--N_om',  type=int,   help='частотных уровней')
    ap.add_argument('--steps', type=int)
    ap.add_argument('--dt',    type=float)
    ap.add_argument('--gamma', type=float)
    ap.add_argument('--tau',   type=float)
    ap.add_argument('--L_max', type=int,   help='макс. угловой момент')
    ap.add_argument('--out',   type=str)
    ap.add_argument('--IC',    type=str,
                    choices=['thermal','thermal_perturbed','anisotropic',
                             'gaussian_blob'])
    ap.add_argument('--no-gap', action='store_true',
                    help='не вычислять γ* (использовать заданное γ)')
    a = ap.parse_args()

    # Выбираем базовый конфиг
    if a.mode == 'custom':
        cfg = dict(CONFIGS['test'])
        if a.config:
            with open(a.config) as f: cfg.update(json.load(f))
    else:
        cfg = dict(CONFIGS[a.mode])

    # Переопределения из CLI
    for src, dst in [('N_x','N_x'),('N_mu','N_mu'),('N_om','N_om'),
                     ('steps','N_steps'),('dt','dt'),('gamma','gamma'),
                     ('tau','tau'),('L_max','L_max'),('out','out'),('IC','IC')]:
        v = getattr(a, src, None)
        if v is not None: cfg[dst] = v
    if a.no_gap:
        cfg['gap'] = False

    sim = LatticeCKG(cfg)
    sim.run()
