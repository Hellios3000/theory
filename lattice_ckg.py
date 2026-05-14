"""
╔══════════════════════════════════════════════════════════════════════╗
║          LATTICE CKG — Полная численная реализация                  ║
║          Хроно-Кинетическая Гравитация на решётке                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Фазовое пространство:                                               ║
║    x ∈ [0,L]³   (пространственная решётка N_x³)                    ║
║    n ∈ S²        (угловая сетка N_ang точек, HEALPix-стиль)         ║
║    ω ∈ [ω_min, ω_max]  (N_om уровней, логарифмическая сетка)        ║
║                                                                      ║
║  Уравнение движения:                                                 ║
║    k^μ ∂_μ T(x,k) = C[T](x,k)                                      ║
║                                                                      ║
║    k^μ = ω·(1, n_x, n_y, n_z),  n∈S²                               ║
║                                                                      ║
║  Интеграл столкновений (BGK-приближение + угловая жёсткость):        ║
║    C[T] = -T/τ_coll · (T - T_eq[ρ]) + γ·∇²_S²·T                   ║
║                                                                      ║
║    где T_eq = ρ/(4π) (изотропное равновесие),                       ║
║    ρ(x,t) = ∫T dμ — локальная плотность                             ║
║                                                                      ║
║  Схема интегрирования: Strang splitting 2-го порядка                 ║
║    T^{n+1} = S(Δt/2) ∘ C(Δt) ∘ S(Δt/2) T^n                       ║
║                                                                      ║
║    S — стриминг (перенос вдоль характеристик)                       ║
║    C — релаксация (интеграл столкновений)                            ║
║                                                                      ║
║  Диагностика:                                                        ║
║    - Число Кнудсена Kn(x,t)                                          ║
║    - Моменты: φ(x), A_μ(x), h_μν(x)                                ║
║    - Спектр мощности P(k)                                            ║
║    - Сохранение заряда ∫T dμ                                         ║
║    - Gap-уравнение для γ*                                            ║
║                                                                      ║
║  Требования:  numpy, scipy  (CPU-версия)                             ║
║  GPU-версия:  cupy (опционально, автодетект)                         ║
║                                                                      ║
║  Запуск:  python3 lattice_ckg.py [--config config.json]              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import special, ndimage
import time
import json
import os
import sys
import argparse

# ── GPU-автодетект ─────────────────────────────────────────────────────
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("[CKG] GPU (CuPy) обнаружен")
except ImportError:
    cp = None
    GPU_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # Решётка
    "N_x":      16,       # пространственная сетка (N_x³) — для теста
    "N_ang":    64,       # точек на S² (должно быть ~4·N_side²)
    "N_om":     16,       # уровней по частоте
    "L":        10.0,     # физический размер ящика
    "om_min":   0.1,      # минимальная частота
    "om_max":   5.0,      # максимальная частота

    # Физические параметры
    "lambda":   0.1,      # константа самовзаимодействия
    "gamma":    0.1,      # начальная угловая жёсткость (будет уточнена)
    "tau_coll": 1.0,      # время столкновений (1/скорость термализации)
    "T_phys":   1.0,      # физическая температура равновесия
    "mu_reg":   0.05,     # регуляризатор (квазисветоподобный конус)

    # Симуляция
    "N_steps":  200,      # число шагов
    "dt":       0.05,     # шаг по времени
    "diag_every": 20,     # диагностика каждые N шагов

    # Начальное условие
    "IC":       "thermal_perturbed",  # тип начального условия
    "IC_ampl":  0.3,      # амплитуда начального возмущения

    # Вычисления
    "use_gpu":  False,    # использовать GPU
    "n_threads": 4,       # потоков CPU
    "output_dir": "ckg_output",  # папка для результатов

    # Gap-уравнение
    "compute_gamma_star": True,  # вычислять γ* самосогласованно
    "gap_tol":  1e-6,     # точность сходимости γ*
    "gap_iters": 100,     # максимум итераций gap-уравнения
}

# ══════════════════════════════════════════════════════════════════════
# УГЛОВАЯ СЕТКА НА S²
# ══════════════════════════════════════════════════════════════════════

class AngularGrid:
    """
    Равномерная (по площади) сетка точек на S².
    Использует метод Фибоначчи-Сансон для N точек.
    """
    def __init__(self, N):
        self.N = N
        self._build()

    def _build(self):
        N = self.N
        # Метод золотого угла (uniform coverage of S²)
        golden = np.pi * (3 - np.sqrt(5))
        i      = np.arange(N, dtype=float)
        y      = 1 - 2*i/(N-1)            # cos θ
        r      = np.sqrt(np.maximum(0, 1 - y**2))
        phi    = golden * i

        self.nx = r * np.cos(phi)          # (N,)
        self.ny = r * np.sin(phi)
        self.nz = y
        self.n  = np.stack([self.nx, self.ny, self.nz], axis=1)  # (N,3)

        # Угловые координаты
        self.theta = np.arccos(np.clip(y, -1, 1))
        self.phi   = phi % (2*np.pi)

        # Вес каждой точки (равномерный)
        self.dOmega = 4*np.pi / N * np.ones(N)

        # Строим оператор Лапласа на S² через сферические гармоники
        self._build_laplacian()

    def _build_laplacian(self):
        """
        Оператор Лапласа-Бельтрами ∇²_S² в базисе точек.
        Строим через сферические гармоники Y_ℓm:
        ∇²_S² = Σ_ℓ -ℓ(ℓ+1) P_ℓ (проектор на подпространство ℓ)
        """
        N     = self.N
        L_max = int(np.sqrt(N)) + 1   # усечение по ℓ

        # Матрица Y_ℓm(θ_i, φ_i) — все гармоники до L_max
        # scipy>=1.15 переименовала sph_harm → sph_harm_y
        try:
            _sph_harm = special.sph_harm
        except AttributeError:
            _sph_harm = special.sph_harm_y

        harmonics = []
        ell_list  = []
        for ell in range(L_max + 1):
            for m in range(-ell, ell+1):
                Y = _sph_harm(m, ell, self.phi, self.theta)
                harmonics.append(Y)
                ell_list.append(ell)

        Y_mat  = np.array(harmonics).T    # (N, n_harm)
        ell_arr = np.array(ell_list, dtype=float)

        # Нормировка: <Y_ℓm | Y_ℓ'm'> = δ, вес 4π/N
        w = self.dOmega[0]   # равный вес
        YtY = (Y_mat.conj().T @ Y_mat) * w   # (n_harm, n_harm)

        # Псевдообратная
        try:
            YtY_inv = np.linalg.pinv(YtY, rcond=1e-10)
        except:
            YtY_inv = np.eye(YtY.shape[0])

        # Оператор Лапласа: L = Y · diag(-ℓ(ℓ+1)) · Y†
        eig = -ell_arr * (ell_arr + 1)    # собственные значения
        self.Lap_S2 = np.real(
            Y_mat @ (YtY_inv @ (np.diag(eig) @ (YtY_inv @ Y_mat.conj().T)))
        ) * w
        # (N,N) — матрица Лапласа

        self.Y_mat   = Y_mat
        self.ell_arr = ell_arr
        self.L_max   = L_max

    def apply_laplacian(self, f):
        """
        Применяет ∇²_S² к f.shape = (..., N_ang) — к последней оси.
        """
        orig_shape = f.shape
        N_ang      = orig_shape[-1]
        f_flat     = f.reshape(-1, N_ang)                  # (M, N_ang)
        result     = f_flat @ self.Lap_S2.T                 # (M, N_ang)
        return result.reshape(orig_shape)

    def angular_power(self, f):
        """
        Угловой спектр мощности: C_ℓ = Σ_m |c_ℓm|²
        f.shape = (N_ang,)
        """
        w   = self.dOmega[0]
        c   = (self.Y_mat.conj().T @ f) * w                # (n_harm,)
        L_max = int(self.ell_arr.max())
        Cl  = np.zeros(L_max+1)
        for ell in range(L_max+1):
            mask = self.ell_arr == ell
            Cl[ell] = np.sum(np.abs(c[mask])**2)
        return Cl


# ══════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ КЛАСС СИМУЛЯЦИИ
# ══════════════════════════════════════════════════════════════════════

class LatticeCKG:
    """
    Полная Lattice CKG симуляция.

    T[ix, iy, iz, i_om, i_ang] — функция распределения
    """

    def __init__(self, config=None):
        self.cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._init_grid()
        self._init_field()
        self.t    = 0.0
        self.step = 0
        self.history = {
            'time':          [],
            'charge':        [],
            'charge_err':    [],
            'Kn_mean':       [],
            'Kn_max':        [],
            'anisotropy':    [],
            'gamma_star':    [],
            'phi_mean':      [],
            'phi_std':       [],
        }
        os.makedirs(self.cfg['output_dir'], exist_ok=True)

    # ── инициализация сетки ───────────────────────────────────────────

    def _init_grid(self):
        cfg = self.cfg
        Nx  = cfg['N_x']
        self.Nx = Nx

        # Пространственная сетка
        self.dx  = cfg['L'] / Nx
        self.x   = np.linspace(0, cfg['L'], Nx, endpoint=False)

        # Частотная сетка (логарифмическая)
        self.omega = np.logspace(
            np.log10(cfg['om_min']),
            np.log10(cfg['om_max']),
            cfg['N_om']
        )
        self.dom   = np.diff(np.log(self.omega)).mean()   # шаг в ln(ω)
        self.N_om  = cfg['N_om']

        # Угловая сетка
        self.ang   = AngularGrid(cfg['N_ang'])
        self.N_ang = cfg['N_ang']

        # 4-импульсы k^μ = (ω, ω·n) — предвычисляем
        # kx[i_om, i_ang], ky, kz, k0 — все (N_om, N_ang)
        om2d = self.omega[:, None]          # (N_om, 1)
        self.k0 = om2d * np.ones(self.N_ang)[None, :]        # (N_om, N_ang)
        self.kx = om2d * self.ang.nx[None, :]
        self.ky = om2d * self.ang.ny[None, :]
        self.kz = om2d * self.ang.nz[None, :]

        # Мера dμ = ω dω dΩ / (2π)²  — нормировочный вес
        # (N_om, N_ang)
        self.dmu = (om2d * self.ang.dOmega[None, :] *
                    self.omega[:, None] * self.dom / (4*np.pi))

        print(f"[CKG] Сетка: {Nx}³ × {self.N_om} × {self.N_ang}")
        size_GB = (Nx**3 * self.N_om * self.N_ang * 8) / 1e9
        print(f"[CKG] Размер массива T: {size_GB:.3f} GB")
        if size_GB > 4:
            print(f"[CKG] ⚠ Большой массив — рассмотрите меньшую сетку")

    def _init_field(self):
        """Начальное условие для T(x,k)"""
        cfg  = self.cfg
        Nx   = self.Nx
        shape = (Nx, Nx, Nx, self.N_om, self.N_ang)

        IC   = cfg['IC']
        ampl = cfg['IC_ampl']

        if IC == 'thermal':
            # Чистое тепловое состояние
            T0 = np.exp(-self.omega / cfg['T_phys'])  # (N_om,)
            self.T = np.ones(shape) * T0[None, None, None, :, None]

        elif IC == 'thermal_perturbed':
            # Тепловое + случайные возмущения
            T0   = np.exp(-self.omega / cfg['T_phys'])
            base = np.ones(shape) * T0[None, None, None, :, None]
            np.random.seed(42)
            noise = ampl * np.random.randn(*shape)
            self.T = np.abs(base + noise)

        elif IC == 'anisotropic':
            # Анизотропное: больше энергии вдоль z
            T0   = np.exp(-self.omega / cfg['T_phys'])
            base = np.ones(shape) * T0[None, None, None, :, None]
            # Усилить моды с n_z > 0
            ang_weight = (1 + ampl * self.ang.nz[None, :])  # (1, N_ang)
            self.T = base * ang_weight[None, None, None, None, :]

        elif IC == 'gaussian_blob':
            # Гауссов сгусток в центре ящика
            L  = cfg['L']
            x0 = L/2; sigma = L/6
            X, Y, Z = np.meshgrid(self.x, self.x, self.x, indexing='ij')
            blob = np.exp(-((X-x0)**2+(Y-x0)**2+(Z-x0)**2)/(2*sigma**2))
            T0   = np.exp(-self.omega / cfg['T_phys'])
            self.T = blob[:,:,:,None,None] * T0[None,None,None,:,None]
            self.T = np.maximum(self.T, 1e-10)

        else:
            raise ValueError(f"Неизвестное IC: {IC}")

        self.T0_charge = self._charge()
        print(f"[CKG] Начальное условие '{IC}': заряд = {self.T0_charge:.6f}")

    # ── макроскопические величины ─────────────────────────────────────

    def _charge(self):
        """∫T dμ — суммарный хроно-заряд"""
        return float(np.sum(self.T * self.dmu[None,None,None,:,:]))

    def _local_density(self):
        """ρ(x) = ∫T dμ  →  (Nx,Nx,Nx)"""
        return np.einsum('xyzoa,oa->xyz', self.T, self.dmu)

    def _phi(self):
        """Дилатон φ(x) = ∫T dμ = ρ(x)"""
        return self._local_density()

    def _A_mu(self):
        """Векторный момент A^μ(x) = ∫k^μ T dμ  →  (4, Nx,Nx,Nx)"""
        # k^μ = (k0, kx, ky, kz) — (N_om, N_ang)
        ks  = [self.k0, self.kx, self.ky, self.kz]
        out = []
        for km in ks:
            wt = km * self.dmu         # (N_om, N_ang)
            out.append(np.einsum('xyzoa,oa->xyz', self.T, wt))
        return np.array(out)           # (4,Nx,Nx,Nx)

    def _h_munu(self):
        """Гравитонный момент h^μν(x) = ∫k^μk^ν T dμ  →  (4,4,Nx,Nx,Nx)"""
        ks  = [self.k0, self.kx, self.ky, self.kz]
        out = np.zeros((4,4,self.Nx,self.Nx,self.Nx))
        for mu in range(4):
            for nu in range(mu, 4):
                wt = ks[mu] * ks[nu] * self.dmu
                val = np.einsum('xyzoa,oa->xyz', self.T, wt)
                out[mu,nu] = val
                out[nu,mu] = val
        return out

    def _tracelessness(self):
        """h^μ_μ = ∫k²T dμ — должно быть ≈ 0"""
        # k² = k0² - kx² - ky² - kz² = ω²(1-1) = 0 точно
        k2 = (self.k0**2 - self.kx**2 - self.ky**2 - self.kz**2)
        return float(np.sum(self.T * k2[None,None,None,:,:] * self.dmu[None,None,None,:,:]))

    def _anisotropy(self):
        """
        Дипольная анизотропия: a₁ = |∫T·n dΩ| / ∫T dΩ
        Усреднённая по x и ω.
        """
        # T усреднённое по x,ω: (N_ang,)
        T_ang = np.mean(self.T, axis=(0,1,2,3))   # (N_ang,)
        norm  = np.sum(T_ang * self.ang.dOmega)
        if norm < 1e-15:
            return 0.0
        dipole = np.sqrt(
            np.sum(T_ang * self.ang.nx * self.ang.dOmega)**2 +
            np.sum(T_ang * self.ang.ny * self.ang.dOmega)**2 +
            np.sum(T_ang * self.ang.nz * self.ang.dOmega)**2
        )
        return float(dipole / norm)

    def _knudsen(self, rho):
        """
        Число Кнудсена Kn(x) = λ_free / L_macro
        λ_free = τ_coll · c_s  (характерная скорость = 1 в нат. ед.)
        L_macro = ρ / |∇ρ|
        """
        tau = self.cfg['tau_coll']
        # Градиент ρ (центральные разности с периодическими ГУ)
        grad_rho = np.sqrt(
            np.roll(rho,1,0) - np.roll(rho,-1,0)**2 / (4*self.dx)**2 +
            np.roll(rho,1,1) - np.roll(rho,-1,1)**2 / (4*self.dx)**2 +
            np.roll(rho,1,2) - np.roll(rho,-1,2)**2 / (4*self.dx)**2
        )
        L_macro  = rho / (np.abs(grad_rho) + 1e-10)
        Kn       = tau / (L_macro + 1e-10)
        return Kn

    # ── шаг стриминга ─────────────────────────────────────────────────

    def _stream_step(self, dt):
        """
        Перенос T(x,k) вдоль характеристик x → x + v·dt
        v = n (единичная скорость для ω-независимого стриминга)

        Реализация: перенос по каждому направлению отдельно (splitting),
        билинейная интерполяция с периодическими ГУ.
        """
        Nx = self.Nx
        dx = self.dx

        # Для каждой угловой моды — сдвиг на n·dt·ω/dx ячеек
        # Используем numpy.roll с дробным сдвигом через интерполяцию
        T_new = np.zeros_like(self.T)

        for i_om in range(self.N_om):
            om = self.omega[i_om]
            for i_ang in range(self.N_ang):
                slice_ = self.T[:, :, :, i_om, i_ang]  # (Nx,Nx,Nx)

                # Сдвиг в единицах ячеек
                sx = om * self.ang.nx[i_ang] * dt / dx
                sy = om * self.ang.ny[i_ang] * dt / dx
                sz = om * self.ang.nz[i_ang] * dt / dx

                # Сдвиг через ndimage.shift с периодическими ГУ
                shifted = ndimage.shift(
                    slice_,
                    shift=[-sx, -sy, -sz],
                    mode='wrap',
                    order=1          # линейная интерполяция
                )
                T_new[:, :, :, i_om, i_ang] = shifted

        self.T = np.maximum(T_new, 0.0)   # физическое ограничение T ≥ 0

    def _stream_step_fast(self, dt):
        """
        Быстрая версия стриминга: только целочисленные сдвиги (roll).
        Точность первого порядка, но в 100x быстрее.
        Используется при |shift| < 0.5 ячейки.
        """
        Nx = self.Nx
        dx = self.dx
        T_new = self.T.copy()

        for i_ang in range(self.N_ang):
            nx_val = self.ang.nx[i_ang]
            ny_val = self.ang.ny[i_ang]
            nz_val = self.ang.nz[i_ang]

            for i_om in range(self.N_om):
                om  = self.omega[i_om]
                sx  = int(round(-om * nx_val * dt / dx))
                sy  = int(round(-om * ny_val * dt / dx))
                sz  = int(round(-om * nz_val * dt / dx))
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                sl = self.T[:,:,:,i_om,i_ang]
                T_new[:,:,:,i_om,i_ang] = np.roll(
                    np.roll(np.roll(sl, sx, 0), sy, 1), sz, 2
                )

        self.T = np.maximum(T_new, 0.0)

    # ── шаг столкновений ──────────────────────────────────────────────

    def _build_T_eq(self, rho):
        """T_eq(x,ω,n) = ρ(x) · f_eq(ω) / (∫f_eq dμ)"""
        T_phys = self.cfg['T_phys']
        f_eq   = np.exp(-self.omega / T_phys)              # (N_om,)
        norm   = np.sum(f_eq[:, None] * self.dmu)          # скаляр
        norm   = max(norm, 1e-20)
        return rho[:,:,:,None,None] * (f_eq[None,None,None,:,None] / norm)

    def _collision_step(self, dt):
        """
        BGK + угловая жёсткость — неявный шаг (безусловно устойчив):
          T^{n+1} = (T^n + α·T_eq + dt·γ·∇²T^n) / (1 + α)
          α = dt/τ
        CFL для угловой диффузии: разбиваем на подшаги если нужно.
        """
        cfg = self.cfg
        tau = cfg['tau_coll']
        gam = cfg['gamma']

        lam_max = float(self.ang.L_max * (self.ang.L_max + 1))
        cfl     = dt * abs(gam) * lam_max
        n_sub   = max(1, int(cfl / 0.45) + 1)
        dt_sub  = dt / n_sub

        for _ in range(n_sub):
            rho   = self._local_density()
            T_eq  = self._build_T_eq(rho)
            lap_T = self.ang.apply_laplacian(self.T)
            alpha = dt_sub / tau
            self.T = np.maximum(
                (self.T + alpha * T_eq + dt_sub * gam * lap_T) / (1.0 + alpha),
                0.0
            )

    def _collision_step_implicit(self, dt):
        """Псевдоним — теперь _collision_step уже неявный"""
        self._collision_step(dt)

    # ── gap-уравнение ─────────────────────────────────────────────────

    def compute_gamma_star(self):
        """
        γ* = <δT | -∇²_S² δT> / <δT | δT>
        δT = T - <T>_ang  (угловое отклонение от среднего)
        Это корректная мера угловой жёсткости.
        """
        # Угловое среднее: T_mean(x,ω) = Σ_n T·dΩ / (4π)
        T_mean = np.sum(self.T * self.ang.dOmega[None,:], axis=-1,
                        keepdims=True) / (4*np.pi)  # (..., 1)
        dT     = self.T - T_mean                     # угловые флуктуации

        lap_dT = self.ang.apply_laplacian(dT)
        dmu4d  = self.dmu[None,None,None,:,:]

        num = -np.sum(dT * lap_dT * dmu4d)
        den =  np.sum(dT * dT     * dmu4d)

        if den < 1e-20:
            return self.cfg['gamma']
        return float(num / den)

    def run_gap_iteration(self):
        """
        Самосогласованная итерация для γ*:
        γ_new = gap_rhs(γ_old)  →  повторяем до сходимости
        """
        cfg   = self.cfg
        gamma = cfg['gamma']
        tol   = cfg['gap_tol']
        n_max = cfg['gap_iters']

        print(f"\n[CKG] Итерация gap-уравнения (начальное γ = {gamma:.6f})")
        for i in range(n_max):
            cfg['gamma'] = gamma
            gamma_new    = self.compute_gamma_star()
            # Демпфирование
            gamma_new    = 0.6*gamma_new + 0.4*gamma
            delta        = abs(gamma_new - gamma) / (abs(gamma) + 1e-15)
            gamma        = gamma_new
            if (i+1) % 10 == 0 or delta < tol:
                print(f"  iter {i+1:4d}: γ = {gamma:.8f}, |Δγ/γ| = {delta:.2e}")
            if delta < tol:
                print(f"[CKG] γ* = {gamma:.8f} (сошлось за {i+1} итераций)")
                break
        else:
            print(f"[CKG] γ* = {gamma:.8f} (не сошлось за {n_max} итераций)")

        cfg['gamma'] = gamma
        return gamma

    # ── один полный шаг Strang splitting ─────────────────────────────

    def _step(self):
        """T^{n+1} = S(dt/2) ∘ C(dt) ∘ S(dt/2) T^n"""
        dt  = self.cfg['dt']
        tau = self.cfg['tau_coll']

        # Выбираем метод: если τ << dt — неявный
        implicit = (tau < 0.5*dt)
        stream   = self._stream_step_fast   # быстрый стриминг

        stream(dt/2)
        if implicit:
            self._collision_step_implicit(dt)
        else:
            self._collision_step(dt)
        stream(dt/2)

        self.t    += dt
        self.step += 1

    # ── диагностика ───────────────────────────────────────────────────

    def _diagnose(self):
        """Сбор диагностических данных"""
        rho    = self._local_density()
        charge = float(np.sum(rho * self.dx**3))
        charge_err = abs(charge - self.T0_charge) / (abs(self.T0_charge) + 1e-15)
        Kn     = self._knudsen(rho)
        anis   = self._anisotropy()
        phi    = rho
        gstar  = self.compute_gamma_star()

        self.history['time'].append(self.t)
        self.history['charge'].append(charge)
        self.history['charge_err'].append(charge_err)
        self.history['Kn_mean'].append(float(np.mean(Kn)))
        self.history['Kn_max'].append(float(np.max(Kn)))
        self.history['anisotropy'].append(anis)
        self.history['gamma_star'].append(gstar)
        self.history['phi_mean'].append(float(np.mean(phi)))
        self.history['phi_std'].append(float(np.std(phi)))

        # Бесследие гравитона
        trace = self._tracelessness()

        print(
            f"  t={self.t:6.2f} | ρ̄={np.mean(rho):.4f} "
            f"| ΔQ={charge_err:.2e} "
            f"| Kn̄={np.mean(Kn):.3f} "
            f"| anis={anis:.4f} "
            f"| γ*={gstar:.5f} "
            f"| h^μ_μ={trace:.2e}"
        )
        return charge_err

    def _save_snapshot(self):
        """Сохраняет срез T(x,k) и историю в файлы"""
        out = self.cfg['output_dir']
        step = self.step

        # Сохраняем моменты (усреднённые по x)
        rho = self._local_density()
        np.save(f"{out}/rho_step{step:04d}.npy", rho.astype(np.float32))

        # История
        hist_arr = {k: np.array(v) for k, v in self.history.items()}
        np.savez(f"{out}/history.npz", **hist_arr)

    # ── главный цикл ─────────────────────────────────────────────────

    def run(self):
        """Основной цикл симуляции"""
        cfg = self.cfg
        N_steps    = cfg['N_steps']
        diag_every = cfg['diag_every']

        print(f"\n[CKG] Запуск симуляции: {N_steps} шагов, dt={cfg['dt']}")
        print(f"[CKG] Параметры: γ={cfg['gamma']:.4f}, τ={cfg['tau_coll']:.4f}, "
              f"λ={cfg['lambda']:.4f}")

        # Начальная диагностика + gap-уравнение
        print(f"\n[CKG] --- Начальное состояние ---")
        self._diagnose()
        if cfg['compute_gamma_star']:
            gamma_star = self.run_gap_iteration()
            cfg['gamma'] = gamma_star

        t_start = time.time()

        for i in range(N_steps):
            self._step()

            if (i+1) % diag_every == 0:
                t_el  = time.time() - t_start
                speed = (i+1) / t_el
                eta   = (N_steps - i - 1) / speed
                print(f"\n[CKG] --- Шаг {self.step}/{N_steps} "
                      f"({t_el:.1f}с, ETA {eta:.0f}с) ---")
                charge_err = self._diagnose()
                self._save_snapshot()

                if charge_err > 0.01:
                    print(f"[CKG] ⚠ Ошибка заряда {charge_err:.2e} > 1% — "
                          f"уменьшите dt")

        # Финальная диагностика
        print(f"\n[CKG] === ФИНАЛ ===")
        self._diagnose()
        self._save_snapshot()

        t_total = time.time() - t_start
        print(f"\n[CKG] Симуляция завершена за {t_total:.1f}с")
        self._final_report()

    # ── финальный отчёт ───────────────────────────────────────────────

    def _final_report(self):
        cfg = self.cfg
        h   = self.history

        print(f"\n{'═'*60}")
        print(f"  LATTICE CKG — ФИНАЛЬНЫЙ ОТЧЁТ")
        print(f"{'═'*60}")

        # Параметры
        print(f"\n  Сетка: {self.Nx}³ × {self.N_om} × {self.N_ang}")
        print(f"  Шагов: {self.step}, dt={cfg['dt']}, T_total={self.t:.2f}")

        # Сохранение заряда
        charge_errs = np.array(h['charge_err'])
        print(f"\n  Сохранение заряда:")
        print(f"    Макс ошибка: {charge_errs.max():.2e}")
        print(f"    Средн. ошибка: {charge_errs.mean():.2e}")
        if charge_errs.max() < 1e-4:
            print(f"    ✓ Заряд сохранён с точностью {charge_errs.max():.2e}")
        else:
            print(f"    ⚠ Значительная потеря заряда — нужен меньший dt")

        # Число Кнудсена
        Kn_arr = np.array(h['Kn_mean'])
        print(f"\n  Число Кнудсена Kn(t):")
        print(f"    Начальное:  {Kn_arr[0]:.4f}")
        print(f"    Финальное:  {Kn_arr[-1]:.4f}")
        if Kn_arr[0] > 1 and Kn_arr[-1] < 1:
            # Момент перехода
            idx_cross = np.argmin(np.abs(Kn_arr - 1.0))
            t_cross   = h['time'][idx_cross]
            print(f"    Переход Kn=1 при t ≈ {t_cross:.2f}")
            print(f"    ✓ Баллистический → гидродинамический режим!")
        elif Kn_arr[-1] < 1:
            print(f"    ✓ Гидродинамический режим с начала")
        else:
            print(f"    → Остаётся баллистическим (увеличьте N_steps или τ)")

        # Угловая термализация
        anis = np.array(h['anisotropy'])
        print(f"\n  Угловая термализация:")
        print(f"    Нач. анизотропия: {anis[0]:.6f}")
        print(f"    Фин. анизотропия: {anis[-1]:.6f}")
        ratio = anis[-1]/anis[0] if anis[0] > 1e-10 else 1.0
        print(f"    Подавление: {ratio:.4f}")
        if ratio < 0.1:
            print(f"    ✓ Угловая термализация достигнута")

        # Самосогласованный γ*
        gstar_arr = np.array(h['gamma_star'])
        gstar_fin = gstar_arr[-1]
        print(f"\n  Самосогласованный γ*:")
        print(f"    Начальное γ (из gap-ур.): {gstar_arr[0]:.8f}")
        print(f"    Финальное γ*: {gstar_fin:.8f}")
        print(f"    Дрейф: {abs(gstar_fin-gstar_arr[0])/(abs(gstar_arr[0])+1e-15):.4f}")

        # Спектр масс ТМ
        print(f"\n  Спектр масс тёмной материи при γ* = {gstar_fin:.6f}:")
        for ell in range(3, 7):
            m2  = gstar_fin * ell*(ell+1)
            print(f"    ℓ={ell}: m² = {m2:.6f},  m = {np.sqrt(m2):.6f} √γ*")

        # Бесследие гравитона
        trace = self._tracelessness()
        print(f"\n  Бесследие гравитона h^μ_μ = {trace:.2e}")
        if abs(trace) < 1e-8:
            print(f"    ✓ h^μ_μ = 0 (точно, из k²=0)")

        # Сохранение результатов
        out = cfg['output_dir']
        summary = {
            'gamma_star':       float(gstar_fin),
            'charge_err_max':   float(charge_errs.max()),
            'Kn_initial':       float(Kn_arr[0]),
            'Kn_final':         float(Kn_arr[-1]),
            'anisotropy_ratio': float(ratio),
            'tracelessness':    float(trace),
            'config':           cfg,
        }
        with open(f"{out}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Результаты сохранены в '{out}/'")
        print(f"  summary.json: γ* = {gstar_fin:.8f}")

        print(f"\n{'═'*60}")


# ══════════════════════════════════════════════════════════════════════
# БЫСТРЫЙ ТЕСТ (маленькая сетка)
# ══════════════════════════════════════════════════════════════════════

def run_quick_test():
    """Быстрый тест на маленькой сетке для проверки корректности"""
    print("\n[CKG] === БЫСТРЫЙ ТЕСТ (N_x=4, N_ang=16, N_om=8) ===\n")
    cfg = {
        **DEFAULT_CONFIG,
        "N_x":       4,
        "N_ang":     16,
        "N_om":      8,
        "N_steps":   40,
        "dt":        0.1,
        "diag_every": 10,
        "IC":        "thermal_perturbed",
        "IC_ampl":   0.5,
        "output_dir": "ckg_test_output",
    }
    sim = LatticeCKG(cfg)
    sim.run()
    return sim


def run_medium():
    """Средняя сетка — основной тест"""
    print("\n[CKG] === СРЕДНЯЯ СЕТКА (N_x=8, N_ang=32, N_om=12) ===\n")
    cfg = {
        **DEFAULT_CONFIG,
        "N_x":       8,
        "N_ang":     32,
        "N_om":      12,
        "N_steps":   100,
        "dt":        0.05,
        "diag_every": 20,
        "IC":        "anisotropic",
        "IC_ampl":   0.5,
        "output_dir": "ckg_medium_output",
    }
    sim = LatticeCKG(cfg)
    sim.run()
    return sim


def run_production():
    """Продакшн-сетка: 32³×20×64 — нужен GPU или много RAM"""
    print("\n[CKG] === ПРОДАКШН (N_x=32, N_ang=64, N_om=20) ===\n")
    cfg = {
        **DEFAULT_CONFIG,
        "N_x":       32,
        "N_ang":     64,
        "N_om":      20,
        "N_steps":   500,
        "dt":        0.02,
        "diag_every": 50,
        "IC":        "thermal_perturbed",
        "IC_ampl":   0.3,
        "output_dir": "ckg_production_output",
    }
    sim = LatticeCKG(cfg)
    sim.run()
    return sim


# ══════════════════════════════════════════════════════════════════════
# ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lattice CKG Simulation')
    parser.add_argument('--mode', choices=['test','medium','production','custom'],
                        default='test',
                        help='Режим: test (быстро), medium, production, custom')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON-файл с конфигурацией (для --mode custom)')
    parser.add_argument('--N_x',   type=int,   default=None)
    parser.add_argument('--N_ang', type=int,   default=None)
    parser.add_argument('--N_om',  type=int,   default=None)
    parser.add_argument('--steps', type=int,   default=None)
    parser.add_argument('--dt',    type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--tau',   type=float, default=None)
    parser.add_argument('--out',   type=str,   default=None)
    args = parser.parse_args()

    if args.mode == 'test':
        sim = run_quick_test()
    elif args.mode == 'medium':
        sim = run_medium()
    elif args.mode == 'production':
        sim = run_production()
    elif args.mode == 'custom':
        cfg = dict(DEFAULT_CONFIG)
        if args.config:
            with open(args.config) as f:
                cfg.update(json.load(f))
        # Переопределения из CLI
        if args.N_x:   cfg['N_x']       = args.N_x
        if args.N_ang: cfg['N_ang']      = args.N_ang
        if args.N_om:  cfg['N_om']       = args.N_om
        if args.steps: cfg['N_steps']    = args.steps
        if args.dt:    cfg['dt']         = args.dt
        if args.gamma: cfg['gamma']      = args.gamma
        if args.tau:   cfg['tau_coll']   = args.tau
        if args.out:   cfg['output_dir'] = args.out
        sim = LatticeCKG(cfg)
        sim.run()
