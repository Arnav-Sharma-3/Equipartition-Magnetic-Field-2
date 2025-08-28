import streamlit as st
import pandas as pd
import math
from math import sqrt, exp, sin, log10
# --------------------------------------------------
# Constants for CGS conversions and synchrotron math
# --------------------------------------------------
CGS_KPC = 3.08567758128e21    # cm per kiloparsec
CGS_MPC = 3.08567758128e24    # cm per Megaparsec
C1 = 6.266e18                 # synchrotron constant
C3 = 2.368e-3                 # synchrotron constant
M_E = 9.1093837139e-28        # electron mass (g)
C_LIGHT = 2.99792458e10       # speed of light (cm/s)
X_FACTOR = 0.0                # proton/electron energy ratio

def run_cosmology_calculator(z, H0, WM, WV):
    """Calculate cosmological distances from redshift"""
    h = H0 / 100
    WR = 4.165E-5 / (h * h)
    WK = 1 - WM - WR - WV
    az = 1.0 / (1.0 + z)
    Tyr = 977.8
    c = 299792.458

    n = 1000  # Integration steps
    # Age at redshift z calculation
    age = 0.0
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        age += 1.0 / adot
    zage = az * age / n

    # Distance calculations
    DTT = 0.0
    DCMR = 0.0
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DTT += 1.0 / adot
        DCMR += 1.0 / (a * adot)
    
    DTT = (1.0 - az) * DTT / n
    DCMR = (1.0 - az) * DCMR / n
    
    # Angular diameter distance
    x = sqrt(abs(WK)) * DCMR
    if x > 0.1:
        ratio = (0.5 * (exp(x) - exp(-x)) / x) if WK > 0 else (sin(x) / x)
    else:
        y = x * x
        ratio = 1. + y / 6. + y * y / 120.
    
    DCMT = ratio * DCMR
    DA = az * DCMT
    DA_Mpc = (c / H0) * DA
    DL = DA / (az * az)  # Luminosity distance
    DL_Mpc = (c / H0) * DL
    kpc_DA = DA_Mpc / 206.264806  # Scale factor (kpc/arcsec)
    
    return {
        'DL_Mpc': DL_Mpc,       # Luminosity distance (Mpc)
        'DA_Mpc': DA_Mpc,       # Angular diameter distance (Mpc)
        'kpc_DA': kpc_DA        # Scale factor (kpc/arcsec)
    }
def compute_fields(alpha, g1, g2, v0, s_v0, l, b, w, z, H0=69.6, WM=0.286, WV=0.714):
    """Calculate magnetic fields using redshift instead of direct distances"""
    # Get cosmology-derived values
    cosmo = run_cosmology_calculator(z, H0, WM, WV)
    D_l = cosmo['DL_Mpc']       # Luminosity distance in Mpc
    D_a = cosmo['DA_Mpc']       # Angular diameter distance in Mpc
    Sf = cosmo['kpc_DA']        # Scale factor (kpc/arcsec)
    
    # Convert kpc → cm, Mpc → cm
    l_cm = l * Sf * CGS_KPC
    b_cm = b * Sf * CGS_KPC
    w_cm = w * Sf * CGS_KPC
    D_l_cm = D_l * CGS_MPC

    # Convert MHz → Hz, Jy → erg/s/cm²/Hz
    v0_hz = v0 * 1e6
    s_v0_cgs = s_v0 * 1e-23

    # Synchrotron calculations
    p = 2 * alpha + 1
    V = (4 / 3) * math.pi * l_cm * b_cm * w_cm * 0.125
    L1 = 4 * math.pi * D_l_cm**2 * s_v0_cgs * v0_hz**alpha

    T3 = (g2 - 1)**(2 - p) - (g1 - 1)**(2 - p)
    T4 = (g2 - 1)**(2 * (1 - alpha)) - (g1 - 1)**(2 * (1 - alpha))
    T5 = (g2 - 1)**(3 - p) - (g1 - 1)**(3 - p)
    T6 = T3 * T4 / T5

    T1 = 3 * L1 / (2 * C3 * (M_E * C_LIGHT**2)**(2 * alpha - 1))
    T2 = (1 + X_FACTOR) / (1 - alpha) * (3 - p) / (2 - p) * (math.sqrt(2/3) * C1)**(1 - alpha)
    A = T1 * T2 * T6
    L = L1 / (1 - alpha) * (math.sqrt(2/3) * C1 * (M_E * C_LIGHT**2)**2)**(1 - alpha) * T4

    B_min = ((4 * math.pi * (1 + alpha) * A) / V)**(1 / (3 + alpha))
    B_eq = (2 / (1 + alpha))**(1 / (3 + alpha)) * B_min

    u_b = B_min**2 / (8 * math.pi)
    u_p = A / V * B_min**(-1 + alpha)
    u_tot = u_p + u_b

    return alpha, B_min * 1e6, B_eq * 1e6, D_l_cm, L, u_p, u_b, u_tot, D_l, D_a, Sf, z
# -----------------------
# Streamlit App Layout
# -----------------------
st.set_page_config(
    page_title="Lobe Magnetic Field Estimator v2 ",
    page_icon="🌌",  
    layout="wide" 
)
st.title("🌀 Lobe Magnetic Field Estimator v2 (Cosmology Calculator Integrated)")

# Cosmology parameters in sidebar
with st.sidebar:
    st.header("Cosmology Parameters")
    H0 = st.number_input("Hubble Constant (H₀)", value=69.6)
    WM = st.number_input("Ω Matter (Ωₘ)", value=0.28600, format="%.5f)
    WV = st.number_input("Ω Vacuum (Ω_Λ)", value=0.71400, format="%.5f)
    #WM = st.slider("Ω Matter (Ωₘ)", 0.001, 1.500, 0.286, format="%.3f")
    #WV = st.slider("Ω Vacuum (Ω_Λ)", 0.001, 1.500, 0.714, format="%.3f")

st.markdown(
    """
    Upload a CSV/TSV with columns:  
    `Source, alpha, gamma1, gamma2, nu0, s_nu0, l, b, w, z`  
    — where **l, b, w** are in **arc second**, **z** is redshift, **nu0** in **MHz**, **s_nu0** in **Jy**.
    """
)

uploaded_file = st.file_uploader("Upload your data file", type=["csv", "tsv", "txt"])
if uploaded_file:
    sep = "\t" if uploaded_file.name.endswith((".tsv", ".txt")) else ","
    try:
        df = pd.read_csv(uploaded_file, sep=sep, comment="#")
    except Exception as e:
        st.error(f"📂 Could not read file: {e}")
    else:
        required = ["Source","alpha","gamma1","gamma2","v0","s_v0","l","b","w","z"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
        else:
            # Data cleaning
            numeric_cols = ["alpha", "gamma1", "gamma2", "v0", "s_v0", "l", "b", "w", "z"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            if df[numeric_cols].isna().any().any():
                invalid_rows = df[df[numeric_cols].isna().any(axis=1)]
                st.warning(f"⚠️ {len(invalid_rows)} rows contain invalid numeric values and will be skipped")
                st.dataframe(invalid_rows)
                df = df.dropna(subset=numeric_cols)
            
            if df.empty:
                st.error("❌ No valid data remaining after cleaning. Please check your input file.")
            else:
                results = df.apply(
                    lambda r: compute_fields(
                        r["alpha"], r["gamma1"], r["gamma2"],
                        r["v0"], r["s_v0"],
                        r["l"], r["b"], r["w"],
                        r["z"], H0, WM, WV
                    ), axis=1, result_type="expand"
                )
                
                df_out = pd.DataFrame({
                    "Source": df["Source"],
                    "Redshift (z)": results[11].round(3),
                    "Spectral Index (α)": results[0],
                    "B_min (μG)": results[1].round(3),
                    "B_eq (μG)": results[2].round(3),
                    "D_L (Mpc)": results[8].round(3),
                    "D_A (Mpc)": results[9].round(3),
                    "Scale (kpc/\")": results[10].round(3),
                    "L (erg/s)": results[4].apply(lambda x: f"{x:.2e}"),
                    "u_p (erg/cm³)": results[5].apply(lambda x: f"{x:.2e}"),
                    "u_B (erg/cm³)": results[6].apply(lambda x: f"{x:.2e}"),
                    "u_total (erg/cm³)": results[7].apply(lambda x: f"{x:.2e}")
                })

                st.success("✅ Calculation complete!")
                st.dataframe(df_out)

                csv_data = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📅 Download Results (CSV)",
                    data=csv_data,
                    file_name="magnetic_fields_results.csv",
                    mime="text/csv"
                )
st.markdown("---")
st.markdown(
    "📌 The cosmology calculator used for this project is based on [James Schombert's python version of the Ned Wright's Cosmology Calculator](https://www.astro.ucla.edu/~wright/CC.python).",
    unsafe_allow_html=True
)
st.markdown(
    "📖 Reference: Wright, E. L. (2006). A Cosmology Calculator for the World Wide Web. *Publications of the Astronomical Society of the Pacific*, 118(850), 1711–1715. [doi:10.1086/510102](https://doi.org/10.1086/510102)",
    unsafe_allow_html=True
)
st.markdown(
    """
    <hr style="margin-top: 3rem; margin-bottom: 1rem;">
    <div style='text-align: center; font-size: 0.9rem; color: gray;'>
        Created by <b>Arnav Sharma</b><br>
        Under the Guidance of <b>Dr. Chiranjib Konar</b>
    </div>
    """,
    unsafe_allow_html=True
)
