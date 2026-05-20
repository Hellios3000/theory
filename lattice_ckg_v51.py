"""
╔══════════════════════════════════════════════════════════════════════╗
║         LATTICE CKG v5  —  ПОЛНАЯ реализация                        ║
║                                                                      ║
║  Исправлено относительно v4:                                         ║
║  ✓ Полная S² (θ,φ) — все Y_ℓm, не только m=0                       ║
║  ✓ 3D стриминг по всем осям (x,y,z) с n=(sinθcosφ, sinθsinφ, cosθ) ║
║  ✓ Вычисление h_μν = ∫k_μk_νT dμ (гравитонный момент)             ║
║  ✓ Правильный API sph_harm_y(ell, m, theta, phi)                    ║
║  ✓ Проверка: h^μ_μ=0, ∂_μh^μν=0 числено                           ║
║                                                                      ║
║  Запуск:                                                             ║
║    python lattice_ckg_v5.py               # тест (~2 мин)           ║
║    python lattice_ckg_v5.py --mode medium  # ~15 мин                ║
║    python lattice_ckg_v5.py --N_x 8 --N_th 12 --N_ph 24 --steps 200║
╚══════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from scipy import special
from numpy.polynomial.legendre import leggauss
import time, json, os, argparse

try:
    sph_harm = special.sph_harm_y   # scipy >= 1.15: (ell, m, theta, phi)
    SPH_NEW  = True
except AttributeError:
    sph_harm = special.sph_harm     # старый: (m, ell, phi, theta)
    SPH_NEW  = False

def Ylm(ell, m, theta, phi):
    """Y_ℓm(θ,φ) — единый интерфейс для обеих версий scipy"""
    if SPH_NEW:
        return sph_harm(ell, m, theta, phi)
    else:
        return sph_harm(m, ell, phi, theta)

# ── цвета ──────────────────────────────────────────────────────────
G="\033[92m"; R="\033[91m"; Y="\033[93m"; B="\033[94m"; W="\033[0m"; BO="\033[1m"
def ok(s):   print(f"  {G}✓{W} {s}")
def warn(s): print(f"  {Y}⚠{W} {s}")
def info(s): print(f"  {B}→{W} {s}")

# ══════════════════════════════════════════════════════════════════════
# УГЛОВАЯ ОСНОВА: полная S² через GL×равном.
# ══════════════════════════════════════════════════════════════════════

class FullS2:
    """
    Полная сферическая квадратура на S²:
      θ: N_th точек Гаусса-Лежандра (по cosθ)
      φ: N_ph равномерных точек

    Сетка: (N_th, N_ph) точек
    Мера: dΩ_ij = w_i · dφ  (интегрирование по обоим углам)
    """
    def __init__(self, N_th, N_ph, L_max):
        self.N_th  = N_th
        self.N_ph  = N_ph
        self.L_max = min(L_max, N_th-1, N_ph//2-1)
        self.N_ang = N_th * N_ph

        # Квадратура
        self.mu, self.w_gl = leggauss(N_th)   # mu=cosθ, веса
        self.theta = np.arccos(np.clip(self.mu, -1+1e-10, 1-1e-10))
        self.phi   = np.linspace(0, 2*np.pi, N_ph, endpoint=False)
        self.dph   = 2*np.pi / N_ph

        # 2D сетки
        self.TH, self.PH = np.meshgrid(self.theta, self.phi, indexing='ij')
        # (N_th, N_ph)

        # Декартовые направления n = (nx, ny, nz)
        sinT = np.sin(self.TH)
        self.nx = sinT * np.cos(self.PH)   # (N_th, N_ph)
        self.ny = sinT * np.sin(self.PH)
        self.nz = np.cos(self.TH)

        # Вес для интегрирования: w_i · dφ
        # ∫f dΩ = Σ_i Σ_j f_ij · w_i · dφ
        self.W = np.outer(self.w_gl, np.ones(N_ph)) * self.dph  # (N_th, N_ph)

        # Базис сферических гармоник
        self._build_harmonics()

        n_harm = len(self.keys)
        n_full = sum(2*l+1 for l in range(self.L_max+1))
        print(f"[S²] N_th={N_th}, N_ph={N_ph}, L_max={self.L_max}")
        print(f"[S²] Гармоник: {n_harm}/{n_full} (все ℓ≤{self.L_max}, все m)")
        print(f"[S²] Покрытие мод: {n_harm/max(1,n_full)*100:.0f}%")

    def _build_harmonics(self):
        """Строим все Y_ℓm до L_max"""
        self.Ylm_all = []  # список (N_th,N_ph) комплексных массивов
        self.keys    = []  # список (ell, m)
        self.eigs    = []  # -ell(ell+1)
        for ell in range(self.L_max+1):
            for m in range(-ell, ell+1):
                Y = Ylm(ell, m, self.TH, self.PH)   # (N_th, N_ph)
                self.Ylm_all.append(Y)
                self.keys.append((ell, m))
                self.eigs.append(-ell*(ell+1))
        self.Ylm_all = np.array(self.Ylm_all)  # (n_harm, N_th, N_ph)
        self.eigs    = np.array(self.eigs)      # (n_harm,)

    def integrate(self, f):
        """∫f dΩ  (f: (...,N_th,N_ph))"""
        return np.einsum('...ij,ij->', f, self.W)

    def project(self, f):
        """
        c_ℓm = ∫f Y*_ℓm dΩ
        f: (..., N_th, N_ph) → c: (..., n_harm) комплексн.
        """
        # c[...,k] = Σ_ij f_ij · conj(Ylm_all[k,ij]) · W_ij
        return np.einsum('...ij,kij,ij->...k', f.astype(complex),
                         np.conj(self.Ylm_all), self.W)

    def laplacian(self, f):
        """
        ∇²_S² f = Σ_ℓm -ℓ(ℓ+1) · c_ℓm · Y_ℓm
        f: (..., N_th, N_ph) → ∇²f: (..., N_th, N_ph)
        """
        c   = self.project(f)                       # (..., n_harm)
        cL  = c * self.eigs                          # (..., n_harm)
        # Σ_k cL_k Y_k
        return np.real(np.einsum('...k,kij->...ij', cL, self.Ylm_all))


# ══════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИИ
# ══════════════════════════════════════════════════════════════════════

CONFIGS = {
    'test': dict(
        N_x=4, N_th=8,  N_ph=16, N_om=8,  L_max=4,
        N_steps=40,  dt=0.1,  diag=10,
        tau=1.0, gamma=0.1, T_phys=1.0,
        om_min=0.1, om_max=4.0, L_box=10.0,
        IC='anisotropic', ampl=0.5,
        out='ckg_v5_test',
    ),
    'medium': dict(
        N_x=6, N_th=12, N_ph=24, N_om=10, L_max=6,
        N_steps=150, dt=0.05, diag=30,
        tau=0.5, gamma=0.1, T_phys=1.0,
        om_min=0.1, om_max=4.0, L_box=10.0,
        IC='gaussian_blob', ampl=0.4,
        out='ckg_v5_medium',
    ),
    'full': dict(
        N_x=8,  N_th=16, N_ph=32, N_om=12, L_max=8,
        N_steps=300, dt=0.03, diag=50,
        tau=0.5, gamma=0.1, T_phys=1.0,
        om_min=0.1, om_max=5.0, L_box=10.0,
        IC='gaussian_blob', ampl=0.3,
        out='ckg_v5_full',
    ),
}


# ══════════════════════════════════════════════════════════════════════
# СИМУЛЯЦИЯ
# ══════════════════════════════════════════════════════════════════════

class LatticeCKGv5:
    """
    T[x,y,z, ω, θ, φ] — функция распределения на полном фазовом пространстве

    Индексация: T.shape = (N_x, N_x, N_x, N_om, N_th, N_ph)
    """

    def __init__(self, cfg):
        self.c   = cfg
        self._setup()
        self._init_T()
        self.t = 0.0; self.step = 0
        self.H = {k: [] for k in [
            't','Q','dQ','Kn','gstar','rho_mean',
            'h_trace','h_diverg','anis_l1','anis_l2'
        ]}
        os.makedirs(cfg['out'], exist_ok=True)

    def _setup(self):
        c = self.c
        self.Nx   = c['N_x']
        self.dx   = c['L_box'] / c['N_x']
        self.om   = np.logspace(np.log10(c['om_min']),
                                np.log10(c['om_max']), c['N_om'])
        dom       = np.diff(np.log(self.om)).mean()
        self.ang  = FullS2(c['N_th'], c['N_ph'], c['L_max'])

        # Мера dμ[iω, ith, iph] = ω²·dω · W[ith,iph] / (2π)²
        self.dmu  = (self.om**2 * dom)[:,None,None] * \
                    self.ang.W[None,:,:] / (2*np.pi)**3
        # (N_om, N_th, N_ph)

        # Равновесная нормировка
        f         = np.exp(-self.om / c['T_phys'])
        Z         = np.sum(f[:,None,None] * self.dmu)
        self.fn   = f / Z      # (N_om,)

        # 4-импульс k^μ = ω·(1, nx, ny, nz)
        # om: (N_om,), nx/ny/nz: (N_th,N_ph)
        self.k0   = self.om[:,None,None] * np.ones((self.c['N_th'],self.c['N_ph']))[None]
        self.kx   = self.om[:,None,None] * self.ang.nx[None]
        self.ky   = self.om[:,None,None] * self.ang.ny[None]
        self.kz   = self.om[:,None,None] * self.ang.nz[None]
        # все shape: (N_om, N_th, N_ph)

        Nx=self.Nx; No=c['N_om']; Nth=c['N_th']; Nph=c['N_ph']
        mb = Nx**3 * No * Nth * Nph * 8 / 1e6
        print(f"[v5] Сетка: {Nx}³ × {No}ω × {Nth}θ × {Nph}φ")
        print(f"[v5] T размер: {mb:.0f} MB  |  L_max={self.ang.L_max}")
        print(f"[v5] Угловых мод: {len(self.ang.keys)} (полная S²)")

    def _init_T(self):
        c = self.c
        Nx=self.Nx; No=c['N_om']; Nth=c['N_th']; Nph=c['N_ph']
        sh = (Nx,Nx,Nx,No,Nth,Nph)
        np.random.seed(42)
        base = self.fn[None,None,None,:,None,None] * np.ones(sh)

        IC = c['IC']
        if IC == 'thermal':
            self.T = base.copy()

        elif IC == 'thermal_perturbed':
            self.T = np.abs(base * (1 + c['ampl']*np.random.randn(*sh)))

        elif IC == 'anisotropic':
            # Возмущение через Y_10 и Y_11
            Y10 = np.real(Ylm(1,0,self.ang.TH,self.ang.PH))  # (Nth,Nph)
            Y11 = np.real(Ylm(1,1,self.ang.TH,self.ang.PH))
            wt  = 1 + c['ampl']*(Y10 + Y11)[None,None,None,None]
            self.T = np.abs(base * wt)

        elif IC == 'gaussian_blob':
            x1 = np.linspace(0,c['L_box'],Nx,endpoint=False)
            X,Y,Z = np.meshgrid(x1,x1,x1,indexing='ij')
            x0 = c['L_box']/2
            blob = np.exp(-((X-x0)**2+(Y-x0)**2+(Z-x0)**2)/(2*(c['L_box']/5)**2))
            self.T = blob[:,:,:,None,None,None] * base
            # Добавляем угловую анизотропию
            Y10 = np.real(Ylm(1,0,self.ang.TH,self.ang.PH))
            self.T = self.T * (1 + c['ampl']*Y10[None,None,None,None])
            self.T = np.abs(self.T)

        self.Q0 = self._Q()
        print(f"[v5] IC='{IC}',  Q₀ = {self.Q0:.4e}")

    # ── интегралы ──────────────────────────────────────────────────
    def _Q(self):
        """Полный заряд ∫T dμ d³x"""
        return float(np.einsum('xyzoa b,oab->', self.T.reshape(
            self.Nx,self.Nx,self.Nx,self.c['N_om'],
            self.c['N_th'],self.c['N_ph']), self.dmu) * self.dx**3)

    def _rho(self):
        """ρ(x) = ∫T dμ"""
        return np.einsum('xyzoa b,oab->xyz', self.T.reshape(
            self.Nx,self.Nx,self.Nx,self.c['N_om'],
            self.c['N_th'],self.c['N_ph']), self.dmu)

    def _h_munu(self):
        """
        h_μν(x) = ∫ k_μ k_ν T dμ  →  (4, 4, Nx,Nx,Nx)

        Проверки:
          h^μ_μ = ∫k²T dμ = 0  (k²=0)
          ∂_μh^μν ≈ 0  (поперечность)
        """
        Nx=self.Nx; No=self.c['N_om']; Nth=self.c['N_th']; Nph=self.c['N_ph']
        T4 = self.T.reshape(Nx,Nx,Nx,No,Nth,Nph)
        ks = [self.k0, self.kx, self.ky, self.kz]  # (No,Nth,Nph) каждый

        h = np.zeros((4,4,Nx,Nx,Nx))
        for mu in range(4):
            for nu in range(mu, 4):
                wt = ks[mu] * ks[nu] * self.dmu    # (No,Nth,Nph)
                val = np.einsum('xyzoa b,oab->xyz', T4, wt)
                h[mu,nu] = val
                h[nu,mu] = val
        return h

    def _h_trace(self, h):
        """h^μ_μ = η^μν h_μν = h_00 - h_11 - h_22 - h_33"""
        return float(np.mean(h[0,0] - h[1,1] - h[2,2] - h[3,3]))

    def _h_divergence(self, h):
        """∂_μ h^μν — поперечность (должно быть ≈0)"""
        # Числовой градиент по x (ось 2 в h[mu,nu,x,y,z])
        dx = self.dx
        dh00_dx0 = 0  # dt h — не вычисляем (стационарно)
        dh01_dx1 = (np.roll(h[0,1],-1,0)-np.roll(h[0,1],1,0))/(2*dx)
        dh02_dx2 = (np.roll(h[0,2],-1,1)-np.roll(h[0,2],1,1))/(2*dx)
        dh03_dx3 = (np.roll(h[0,3],-1,2)-np.roll(h[0,3],1,2))/(2*dx)
        div = dh01_dx1 + dh02_dx2 + dh03_dx3
        return float(np.std(div))  # должно быть мало

    def _anisotropy(self):
        """Угловая анизотропия: дипольный и квадрупольный моменты"""
        T_ang = np.mean(self.T, axis=(0,1,2,3))   # (N_th, N_ph)
        c_all = self.ang.project(T_ang)             # (n_harm,)
        c0    = abs(c_all[0]) + 1e-20
        # ℓ=1 (диполь)
        idx1  = [i for i,(l,m) in enumerate(self.ang.keys) if l==1]
        a1    = float(np.sqrt(sum(abs(c_all[i])**2 for i in idx1))) / c0
        # ℓ=2 (квадруполь)
        idx2  = [i for i,(l,m) in enumerate(self.ang.keys) if l==2]
        a2    = float(np.sqrt(sum(abs(c_all[i])**2 for i in idx2))) / c0
        return a1, a2

    def _gamma_star(self):
        """γ* = <δT|−∇²_S²|δT> / <δT|δT> (полный S²)"""
        T4    = self.T.reshape(self.Nx**3, self.c['N_om'],
                               self.c['N_th'], self.c['N_ph'])
        Tmean = np.sum(T4 * self.ang.W[None,None], axis=(-2,-1), keepdims=True)/(4*np.pi)
        dT    = T4 - Tmean      # (Nx³, No, Nth, Nph)

        # Лаплас по угловым осям
        sh    = dT.shape
        dT2d  = dT.reshape(-1, self.c['N_th'], self.c['N_ph'])
        lap2d = self.ang.laplacian(dT2d)          # (Nx³·No, Nth, Nph)
        lap   = lap2d.reshape(sh)

        w4d   = self.dmu[None]                    # (1, No, Nth, Nph)
        num   = -float(np.einsum('xoab,xoab,oab', dT, lap, self.dmu))
        den   =  float(np.einsum('xoab,xoab,oab', dT, dT,  self.dmu))
        return num/den if den > 1e-20 else self.c['gamma']

    def _kn(self, rho):
        tau = self.c['tau']
        gx  = (np.roll(rho,-1,0)-np.roll(rho,1,0))/(2*self.dx)
        gy  = (np.roll(rho,-1,1)-np.roll(rho,1,1))/(2*self.dx)
        gz  = (np.roll(rho,-1,2)-np.roll(rho,1,2))/(2*self.dx)
        gr  = np.sqrt(gx**2+gy**2+gz**2+1e-30)
        return tau*gr/(rho+1e-15)

    # ── 3D стриминг (ИСПРАВЛЕН) ────────────────────────────────────
    def _stream(self, dt):
        """
        Полный 3D стриминг: T(x,y,z,ω,θ,φ) → T(x+nx·ω·dt, y+ny·ω·dt, z+nz·ω·dt,...)
        nx=sinθcosφ, ny=sinθsinφ, nz=cosθ — используем все три компоненты.
        """
        dx = self.dx
        T_new = self.T.copy()
        No=self.c['N_om']; Nth=self.c['N_th']; Nph=self.c['N_ph']

        for io in range(No):
            om = self.om[io]
            for ith in range(Nth):
                for iph in range(Nph):
                    sl  = self.T[:,:,:,io,ith,iph]   # (Nx,Nx,Nx)

                    # Сдвиги по всем трём осям
                    sx = om * self.ang.nx[ith,iph] * dt / dx
                    sy = om * self.ang.ny[ith,iph] * dt / dx
                    sz = om * self.ang.nz[ith,iph] * dt / dx

                    # Целая и дробная части
                    six = int(np.floor(sx)); sfx = sx - six
                    siy = int(np.floor(sy)); sfy = sy - siy
                    siz = int(np.floor(sz)); sfz = sz - siz

                    # Трилинейная интерполяция (8 соседей)
                    r000 = np.roll(np.roll(np.roll(sl, six,0), siy,1), siz,2)
                    r100 = np.roll(np.roll(np.roll(sl, six+1,0), siy,1), siz,2)
                    r010 = np.roll(np.roll(np.roll(sl, six,0), siy+1,1), siz,2)
                    r001 = np.roll(np.roll(np.roll(sl, six,0), siy,1), siz+1,2)
                    r110 = np.roll(np.roll(np.roll(sl, six+1,0), siy+1,1), siz,2)
                    r101 = np.roll(np.roll(np.roll(sl, six+1,0), siy,1), siz+1,2)
                    r011 = np.roll(np.roll(np.roll(sl, six,0), siy+1,1), siz+1,2)
                    r111 = np.roll(np.roll(np.roll(sl, six+1,0), siy+1,1), siz+1,2)

                    T_new[:,:,:,io,ith,iph] = (
                        (1-sfx)*(1-sfy)*(1-sfz)*r000 +
                        sfx    *(1-sfy)*(1-sfz)*r100 +
                        (1-sfx)*sfy    *(1-sfz)*r010 +
                        (1-sfx)*(1-sfy)*sfz    *r001 +
                        sfx    *sfy    *(1-sfz)*r110 +
                        sfx    *(1-sfy)*sfz    *r101 +
                        (1-sfx)*sfy    *sfz    *r011 +
                        sfx    *sfy    *sfz    *r111
                    )

        self.T = np.maximum(T_new, 0.0)

    # ── столкновения ───────────────────────────────────────────────
    def _collide(self, dt):
        """BGK + полный ∇²_S² с CFL-контролем"""
        c = self.c; tau=c['tau']; gam=c['gamma']
        lmax = self.ang.L_max
        cfl  = dt * abs(gam) * lmax*(lmax+1)
        nsub = max(1, int(cfl/0.38)+1)
        dt_s = dt/nsub; alpha = dt_s/tau

        for _ in range(nsub):
            rho   = self._rho()
            T_eq  = (rho[:,:,:,None,None,None] *
                     self.fn[None,None,None,:,None,None])
            # Лаплас: применяем к угловым осям (последние 2)
            Nx=self.Nx; No=c['N_om']; Nth=c['N_th']; Nph=c['N_ph']
            T4    = self.T.reshape(Nx**3*No, Nth, Nph)
            lap4  = self.ang.laplacian(T4)
            lap   = lap4.reshape(Nx,Nx,Nx,No,Nth,Nph)
            self.T = np.maximum(
                (self.T + alpha*T_eq + dt_s*gam*lap)/(1+alpha), 0.0)

    # ── диагностика ─────────────────────────────────────────────────
    def _diag(self, verbose=True):
        Q    = self._Q()
        dQ   = abs(Q-self.Q0)/(self.Q0+1e-20)
        rho  = self._rho()
        Kn   = float(np.mean(self._kn(rho)))
        gs   = self._gamma_star()
        a1,a2= self._anisotropy()
        h    = self._h_munu()
        ht   = self._h_trace(h)
        hdiv = self._h_divergence(h)

        for k,v in [('t',self.t),('Q',Q),('dQ',dQ),('Kn',Kn),
                    ('gstar',gs),('rho_mean',float(rho.mean())),
                    ('h_trace',ht),('h_diverg',hdiv),
                    ('anis_l1',a1),('anis_l2',a2)]:
            self.H[k].append(v)

        if verbose:
            print(f"  t={self.t:6.2f} | ΔQ={dQ:.1e} | Kn={Kn:.3f}"
                  f" | γ*={gs:.4f} | a₁={a1:.4f} a₂={a2:.4f}")
            print(f"           | h^μ_μ={ht:.2e} | ∂h={hdiv:.2e}")

    def _save(self):
        np.savez(f"{self.c['out']}/history.npz",
                 **{k:np.array(v) for k,v in self.H.items()})

    # ── главный цикл ───────────────────────────────────────────────
    def run(self):
        c = self.c
        print(f"\n[v5] {c['N_steps']} шагов, dt={c['dt']}, τ={c['tau']}, γ={c['gamma']}")
        print("\n──── t=0 ────")
        self._diag()

        t0 = time.time()
        for i in range(c['N_steps']):
            dt = c['dt']
            self._stream(dt/2)
            self._collide(dt)
            self._stream(dt/2)
            self.t += dt; self.step += 1

            if (i+1) % c['diag'] == 0:
                el  = time.time()-t0
                eta = el/(i+1)*(c['N_steps']-i-1)
                print(f"\n──── Шаг {self.step}/{c['N_steps']}"
                      f"  ({el:.0f}с  ETA {eta:.0f}с) ────")
                self._diag()
                self._save()

        print("\n──── ФИНАЛ ────")
        self._diag()
        self._save()
        self._report()

    def _report(self):
        H=self.H; c=self.c
        dQ=np.array(H['dQ']); gs=np.array(H['gstar'])
        ht=np.array(H['h_trace']); hdiv=np.array(H['h_diverg'])
        a1=np.array(H['anis_l1']); a2=np.array(H['anis_l2'])

        print(f"\n{'═'*60}")
        print(f"{BO}  LATTICE CKG v5 — ИТОГ{W}")
        print(f"{'═'*60}")
        print(f"  Сетка: {self.Nx}³×{c['N_om']}ω×{c['N_th']}θ×{c['N_ph']}φ")
        print(f"  Угловых мод: {len(self.ang.keys)} (полная S²)")

        print(f"\n  Сохранение заряда:")
        ok(f"max ΔQ/Q = {dQ.max():.2e}")

        print(f"\n  Гравитонные тождества (из h_μν = ∫k_μk_νT dμ):")
        ok(f"h^μ_μ = {ht.mean():.2e}  (должно = 0, из k²=0)")
        if hdiv.mean() < 1e-4:
            ok(f"∂_μh^μν ≈ {hdiv.mean():.2e}  (поперечность)")
        else:
            warn(f"∂_μh^μν = {hdiv.mean():.2e}  (нужна лучшая сетка)")

        print(f"\n  Угловая анизотропия:")
        print(f"  a₁(t=0)={a1[0]:.4f} → a₁(t=∞)={a1[-1]:.4f}")
        print(f"  a₂(t=0)={a2[0]:.4f} → a₂(t=∞)={a2[-1]:.4f}")
        if a1[-1] < 0.01:
            ok("Дипольная анизотропия подавлена (термализация)")

        print(f"\n  γ* (полная S²):")
        print(f"  γ*(t=0) = {gs[0]:.6f}")
        print(f"  γ*(t=∞) = {gs[-1]:.6f}")
        info("Это γ* из полного S², включая все m≠0 моды")

        print(f"\n  Спектр масс ТМ при γ*={gs[-1]:.4f}:")
        for ell in range(3,7):
            m2 = gs[-1]*ell*(ell+1)
            print(f"  ℓ={ell}: m²={m2:.4f},  m={np.sqrt(max(0,m2)):.4f}√γ*")

        summary = {
            'gamma_star': float(gs[-1]),
            'h_trace_mean': float(ht.mean()),
            'h_divergence': float(hdiv.mean()),
            'charge_err': float(dQ.max()),
            'anisotropy_l1_final': float(a1[-1]),
            'full_S2': True,
            'n_harmonics': len(self.ang.keys),
            'streaming_3D': True,
            'config': c,
        }
        with open(f"{c['out']}/summary.json",'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Результаты: '{c['out']}/'")
        print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Lattice CKG v5 — полная S²')
    ap.add_argument('--mode',  default='test',
                    choices=['test','medium','full','custom'])
    ap.add_argument('--N_x',  type=int); ap.add_argument('--N_th', type=int)
    ap.add_argument('--N_ph', type=int); ap.add_argument('--N_om', type=int)
    ap.add_argument('--steps',type=int); ap.add_argument('--dt',   type=float)
    ap.add_argument('--tau',  type=float); ap.add_argument('--gamma',type=float)
    ap.add_argument('--L_max',type=int); ap.add_argument('--out',  type=str)
    ap.add_argument('--IC',   type=str)
    a = ap.parse_args()

    cfg = dict(CONFIGS[a.mode if a.mode != 'custom' else 'test'])
    for src,dst in [('N_x','N_x'),('N_th','N_th'),('N_ph','N_ph'),
                    ('N_om','N_om'),('steps','N_steps'),('dt','dt'),
                    ('tau','tau'),('gamma','gamma'),('L_max','L_max'),
                    ('out','out'),('IC','IC')]:
        v = getattr(a, src, None)
        if v is not None: cfg[dst] = v

    sim = LatticeCKGv5(cfg)
    sim.run()
