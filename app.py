import streamlit as st
import pandas as pd
import math
from math import sqrt, exp, sin, log10

# ----------------------------
#  1) Cosmology calculator
# ----------------------------
def run_cosmology_calculator(z, H0=69.6, WM=0.286, WV=0.714):
    """
    Given redshift z, returns:
      - DL_Mpc: luminosity distance in Mpc
      - DA_Mpc: angular diameter distance in Mpc
      - a      : scale factor = 1/(1+z)
    """
    h = H0 / 100.0
    WR = 4.165E-5 / (h*h)
    WK = 1 - WM - WR - WV

    az = 1.0 / (1.0 + z)
    c = 299792.458  # km/s
    n = 1000

    # comoving radial integral
    DCMR = 0.0
    for i in range(n):
        a = az + (1-az)*(i+0.5)/n
        adot = sqrt(WK + WM/a + WR/(a*a) + WV*(a*a))
        DCMR += 1.0/(a*adot)
    DCMR *= (1-az)/n

    # transverse comoving
    x = sqrt(abs(WK)) * DCMR
    if x > 0.1:
        ratio = (0.5*(exp(x)-exp(-x))/x) if WK>0 else (sin(x)/x)
    else:
        y = x*x
        if WK<0: y = -y
        ratio = 1 + y/6 + y*y/120
    DCMT = ratio * DCMR

    # distances
    DA    = az * DCMT
    DL    = DCMT / az
    DA_Mpc = (c/H0)*DA
    DL_Mpc = (c/H0)*DL

    return DL_Mpc, DA_Mpc, az

# --------------------------------------------------
#  2) Magnetic‐field routine (exactly as you gave it)
# --------------------------------------------------
CGS_KPC  = 3.08567758128e21
CGS_MPC  = 3.08567758128e24
C1       = 6.266e18
C3       = 2.368e-3
M_E      = 9.1093837139e-28
C_LIGHT  = 2.99792458e10
X_FACTOR = 0.0

def compute_fields(alpha, g1, g2, v0, s_v0, l, b, w, z):
    # 1) cosmology
    DL_Mpc, DA_Mpc, Sf = run_cosmology_calculator(z)

    # 2) convert units
    l_cm    = l * Sf * CGS_KPC
    b_cm    = b * Sf * CGS_KPC
    w_cm    = w * Sf * CGS_KPC
    D_l_cm  = DL_Mpc * CGS_MPC
    v0_hz   = v0 * 1e6
    s_v0_cgs= s_v0 * 1e-23

    # 3) your sync math
    p  = 2*alpha + 1
    V  = (4/3)*math.pi * l_cm*b_cm*w_cm * 0.125
    L1 = 4*math.pi * D_l_cm**2 * s_v0_cgs * v0_hz**alpha

    T3 = (g2-1)**(2-p) - (g1-1)**(2-p)
    T4 = (g2-1)**(2*(1-alpha)) - (g1-1)**(2*(1-alpha))
    T5 = (g2-1)**(3-p)   - (g1-1)**(3-p)
    T6 = T3 * T4 / T5

    T1 = 3*L1 / (2*C3*(M_E*C_LIGHT**2)**(2*alpha-1))
    T2 = (1+X_FACTOR)/(1-alpha) * (3-p)/(2-p) * (math.sqrt(2/3)*C1)**(1-alpha)
    A  = T1 * T2 * T6
    L  = L1/(1-alpha) * (math.sqrt(2/3)*C1*(M_E*C_LIGHT**2)**2)**(1-alpha) * T4

    B_min = ((4*math.pi*(1+alpha)*A)/V)**(1/(3+alpha))
    B_eq  = (2/(1+alpha))**(1/(3+alpha)) * B_min

    u_b   = B_min**2/(8*math.pi)
    u_p   = A/V * B_min**(-1+alpha)
    u_tot = u_p + u_b

    return {
        'Alpha':            alpha,
        'Scale Factor (a)': Sf,
        'D_L (Mpc)':        DL_Mpc,
        'D_A (Mpc)':        DA_Mpc,
        'B_min (μG)':       B_min*1e6,
        'B_eq (μG)':        B_eq*1e6,
        'u_p (erg/cm³)':    u_p,
        'u_b (erg/cm³)':    u_b,
        'u_total (erg/cm³)':u_tot,
        'L (erg/s)':        L
    }

# ------------------------
# 3) Streamlit interface
# ------------------------
st.set_page_config(page_title="MagField + Cosmo", layout="wide")
st.title("🌀 Lobe Magnetic Field + Cosmology")

st.markdown("""
**Upload a CSV/TSV** with columns:
- `l,b,w` in kpc  
- `v0` in MHz  
- `s_v0` in Jy  
- `z` = redshift  
""")

up = st.file_uploader("Upload data file", type=["csv","tsv","txt"])
if up:
    sep = "\t" if up.name.endswith((".tsv",".txt")) else ","
    df = pd.read_csv(up, sep=sep, comment="#")

    required = ['Source','alpha','gamma1','gamma2','v0','s_v0','l','b','w','z']
    if any(col not in df.columns for col in required):
        missing = [c for c in required if c not in df.columns]
        st.error(f"Missing columns: {', '.join(missing)}")
    else:
        # Ensure numeric
        for c in required[1:]:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=required[1:])

        # Compute and collect
        out = []
        for _, r in df.iterrows():
            res = compute_fields(
                r['alpha'], r['gamma1'], r['gamma2'],
                r['v0'], r['s_v0'],
                r['l'], r['b'], r['w'],
                r['z']
            )
            res['Source'] = r['Source']
            res['z']      = r['z']
            out.append(res)

        result_df = pd.DataFrame(out)[
            ['Source','z','Scale Factor (a)','D_L (Mpc)','D_A (Mpc)',
             'Alpha','B_min (μG)','B_eq (μG)','u_p (erg/cm³)','u_b (erg/cm³)','u_total (erg/cm³)','L (erg/s)']
        ]

        st.success("✅ Done")
        st.dataframe(result_df)

        st.download_button(
            "Download CSV",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name="magfield_cosmo_results.csv",
            mime="text/csv"
        )
