import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import solve_ivp

# ==========================================
# CONFIGURACIÓN DE LA INTERFAZ DE STREAMLIT
# ==========================================
st.set_page_config(page_title="Tacho con Variables de Operación", layout="wide")
st.title("Tacho con Variables de Operación")
st.markdown("Ajusta los parámetros operativos en el panel izquierdo (ahora en **kPa**) y presiona simular.")

# ==========================================
# 1. CONSTANTES DEL PROCESO (VALORES BASE)
# ==========================================
Hv_default = 2300.0        # kJ/kg
Cp_default = 3.8           # kJ/kg°C
U_max = 0.5                # kW/m²°C (Coeficiente máximo con fluido limpio)
A = 120.0                  # m²
T_vapor = 115.0            # °C
capacidad_maxima_volumen = 20.0  # m³ (Capacidad física total del tacho)

# ==========================================
# 2. CARGA AUTOMÁTICA DEL EXCEL
# ==========================================
ruta_excel = "base_de_datosTacho.xlsx"

@st.cache_data
def cargar_base_datos(ruta):
    df = pd.read_excel(ruta, header=1)
    df.columns = df.columns.str.strip()
    df.set_index('Fluido', inplace=True)
    return df

try:
    df_fluidos = cargar_base_datos(ruta_excel)
except Exception as e:
    st.error(f"❌ Error al cargar el archivo Excel en la ruta especificada: {e}")
    st.stop()

# ==========================================
# 3. PANEL LATERAL: CONTROLES INTERACTIVOS
# ==========================================
st.sidebar.header("🎛️ Panel de Control del Operador")

# Selección de fluido desde los índices del Excel
lista_fluidos = df_fluidos.index.tolist()
fluido_seleccionado = st.sidebar.selectbox("Selecciona el Fluido a Alimentar:", lista_fluidos)

# Extracción de parámetros del fluido seleccionado
X_in = float(df_fluidos.loc[fluido_seleccionado, 'Brix_in']) / 100.0
Hv = float(df_fluidos.loc[fluido_seleccionado, 'Hv'])
Cp = float(df_fluidos.loc[fluido_seleccionado, 'Cp'])
P_in = float(df_fluidos.loc[fluido_seleccionado, 'Pureza'])

if 'Densidad' in df_fluidos.columns:
    rho_fluido = float(df_fluidos.loc[fluido_seleccionado, 'Densidad'])
else:
    rho_fluido = 1300.0  # kg/m³ valor por defecto

st.sidebar.markdown("---")
st.sidebar.subheader("Variables de Operación Ajustables")

# Variable de operación en Kilopascales (kPa)
P_vacio_kPa = st.sidebar.slider(
    "Presión de Vacío del Tacho (kPa):", 
    min_value=74.5, max_value=88.0, value=81.3, step=0.1
)

# Ecuación termodinámica ajustada a kPa para calcular T_op
T_tacho_calculada = 65.0 - 0.96 * (P_vacio_kPa - 81.3)
st.sidebar.info(f"🌡️ Temperatura de ebullición calculada: **{T_tacho_calculada:.2f} °C**")

# Input para la masa inicial de carga
M_inicial = st.sidebar.number_input(
    f"Masa inicial de {fluido_seleccionado} (kg):", 
    min_value=1000.0, max_value=15000.0, value=5000.0, step=500.0
)

# Flujo de alimentación constante
F_in_constante = st.sidebar.slider("Flujo de alimentación (kg/s):", min_value=0.5, max_value=5.0, value=2.5, step=0.1)

# Botón para ejecutar la simulación
boton_simular = st.sidebar.button("🚀 Correr Simulación", type="primary")

# ==========================================
# 4. MODELO MATEMÁTICO DINÁMICO
# ==========================================
def balances_tacho_profesional(t, estado, F_in_nominal, X_in, Hv, T_op, P_in, rho):
    M_total, X_miel, M_cristal = estado

    # Control por nivel físico (Volumen real)
    V_actual = M_total / rho
    F_in_efectivo = 0.0 if V_actual >= capacidad_maxima_volumen else F_in_nominal

    # Transferencia de calor variable con la viscosidad (Concentración de Brix)
    delta_brix = max(0.0, (X_miel - X_in))
    U_actual = U_max * np.exp(-3.5 * delta_brix) 
    U_actual = max(0.1, U_actual)  # Límite mínimo de seguridad

    # Evaporación con la U dinámica corregida
    Q = U_actual * A * (T_vapor - T_op)
    F_vap = Q / Hv if Hv > 0 else 0.0

    # Sobresaturación
    solubilidad = 64.407 + (0.0725 * T_op) + (0.002057 * T_op**2)
    X_sat = solubilidad / (100 + solubilidad)
    S_sacarosa = (X_miel * (P_in / 100)) / X_sat

    # Cinética de Crecimiento de Cristales
    factor_pureza = P_in / 100
    if M_cristal > 0.01 and S_sacarosa > 1.0:
        k_crecimiento = 0.0005 * factor_pureza
        dMc_dt = k_crecimiento * M_cristal * (S_sacarosa - 1.0)
    else:
        dMc_dt = 0.0

    # Balances Diferenciales ordinarios
    dM_dt = F_in_efectivo - F_vap
    dX_dt = (F_in_efectivo * X_in - dM_dt * X_miel - dMc_dt) / M_total

    return [dM_dt, dX_dt, dMc_dt]

# ==========================================
# 5. RESOLVER MODELO Y GRAFICAR
# ==========================================
if boton_simular:
    X_inicial = X_in
    M_semilla = 50.0  # kg de semilla cristalina base
    estado_inicial_c = [M_inicial, X_inicial, M_semilla]

    t_inicio = 0
    t_fin = 3600
    t_eval = np.linspace(t_inicio, t_fin, 200)

    # Integración numérica
    solucion = solve_ivp(
        balances_tacho_profesional,
        [t_inicio, t_fin],
        estado_inicial_c,
        t_eval=t_eval,
        method='RK45',
        args=(F_in_constante, X_in, Hv, T_tacho_calculada, P_in, rho_fluido)
    )

    # Procesar resultados vectoriales
    m_total_r = solucion.y[0]
    brix_miel_r = solucion.y[1] * 100
    m_cristal_r = solucion.y[2]
    v_total_r = m_total_r / rho_fluido
    tiempo_minutos = solucion.t / 60

    # Recalcular perfil exacto de sobresaturación
    solubilidad_pl = 64.407 + (0.0725 * T_tacho_calculada) + (0.002057 * T_tacho_calculada**2)
    X_sat_pl = solubilidad_pl / (100 + solubilidad_pl)
    sobresaturacion_r = (solucion.y[1] * (P_in / 100)) / X_sat_pl

    # Verificar si se alcanzó el límite físico del tacho
    limite_alcanzado = False
    minuto_limite = 0.0
    for idx, v in enumerate(v_total_r):
        if v >= capacidad_maxima_volumen:
            limite_alcanzado = True
            minuto_limite = tiempo_minutos[idx]
            break

    if limite_alcanzado:
        st.warning(f"⚠️ **¡Alarma de Capacidad Geométrica!** El tacho alcanzó su límite de **{capacidad_maxima_volumen} m³** en el minuto **{minuto_limite:.2f}**. La alimentación se cerró automáticamente.")

    # ==========================================
    # TARJETAS DE DATOS FINALES (TEXTO LIMPIO)
    # ==========================================
    st.subheader("Datos Finales")
    
    # Columnas para los indicadores numéricos principales
    m1, m2, m3, m4 = st.columns(4)
    
    # Proyección del último valor de la simulación
    m1.metric(label="Masa Total Final (Masa Cocida)", value=f"{m_total_r[-1]:,.1f} kg")
    m2.metric(label="Masa de Cristales Obtenida", value=f"{m_cristal_r[-1]:,.1f} kg")
    m3.metric(label="Sobresaturación Final", value=f"{sobresaturacion_r[-1]:.3f}")
    m4.metric(label="Concentración Final (% Brix)", value=f"{brix_miel_r[-1]:.2f} %")
    
    st.markdown("---") 

    # GENERACIÓN DEL MATPLOTLIB (Tus 4 Gráficas Originales)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Masa Total y Cristales
    axs[0, 0].plot(tiempo_minutos, m_total_r, 'b', label='Masa Total (Masa Cocida)')
    axs[0, 0].plot(tiempo_minutos, m_cristal_r, 'r--', label='Masa de Cristales')
    axs[0, 0].set_title('Evolución de Masas')
    axs[0, 0].set_ylabel('Masa (kg)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    # 2. Brix del fluido en el tacho
    axs[0, 1].plot(tiempo_minutos, brix_miel_r, 'g')
    axs[0, 1].set_title('Brix del fluido en el tacho')
    axs[0, 1].set_ylabel('% Brix')
    axs[0, 1].grid(True)

    # 3. Nivel Geométrico (Volumen)
    axs[1, 0].plot(tiempo_minutos, v_total_r, 'purple', linewidth=2)
    axs[1, 0].axhline(capacidad_maxima_volumen, linestyle=':', color='r', label='Capacidad Máxima')
    axs[1, 0].set_title('Volumen ocupado en el Tacho')
    axs[1, 0].set_xlabel('Tiempo (min)')
    axs[1, 0].set_ylabel('Volumen (m³)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # 4. Sobresaturación y zonas de operación
    axs[1, 1].plot(tiempo_minutos, sobresaturacion_r, 'darkorange', linewidth=2)
    axs[1, 1].axhline(1.0, linestyle='--', color='gray', label='Saturación')
    axs[1, 1].axhline(1.1, linestyle='--', color='g', label='Inicio Metaestable')
    axs[1, 1].axhline(1.25, linestyle='--', color='r', label='Límite Metaestable (Falso grano)')
    axs[1, 1].set_title('Sobresaturación (σ) vs Tiempo')
    axs[1, 1].set_xlabel('Tiempo (min)')
    axs[1, 1].set_ylabel('Sobresaturación')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("👈 Selecciona los parámetros en el panel izquierdo y haz clic en **'Correr Simulación'**.")