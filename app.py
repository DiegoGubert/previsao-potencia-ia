import streamlit as st
import pandas as pd
import joblib
 
st.set_page_config(
    page_title="Previsão de Potência em IA",
    page_icon="⚡",
    layout="wide"
)
 
st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}
 
.titulo {
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 0;
}
 
.subtitulo {
    font-size: 1.1rem;
    opacity: 0.75;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)
 
 
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_potencia_ia_deploy.joblib")
 
 
try:
    pacote = carregar_modelo()
except Exception as erro:
    st.error(f"Erro ao carregar o modelo: {erro}")
    st.stop()
 
 
modelo = pacote["modelo"]
scaler = pacote["scaler"]
colunas_numericas = pacote["colunas_numericas"]
colunas_modelo = pacote["colunas_modelo"]
limites = pacote["limites_entradas"]
metricas = pacote["metricas"]
 
 
def numero_br(valor, casas=2):
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")
 
 
st.markdown(
    '<div class="titulo">⚡ Previsão de Potência em Cargas de IA</div>',
    unsafe_allow_html=True
)
 
st.markdown(
    '<div class="subtitulo">'
    'Estime a potência consumida pelas GPUs de uma carga de '
    'Inteligência Artificial utilizando Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)
 
st.info(
    "Informe as características da carga abaixo e clique em "
    "**Prever potência**."
)
 
st.divider()
 
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("🖥️ GPU e carga")
 
    gpu_model = st.selectbox(
        "Modelo da GPU",
        pacote["gpu_models"]
    )
 
    workload = st.selectbox(
        "Tipo de workload",
        pacote["workloads"]
    )
 
    gpu_utilization = st.slider(
        "Utilização média da GPU (%)",
        min_value=float(limites["gpu_utilization_mean"]["min"]),
        max_value=float(limites["gpu_utilization_mean"]["max"]),
        value=float(limites["gpu_utilization_mean"]["mediana"])
    )
 
    gpu_temp = st.number_input(
        "Temperatura média da GPU (°C)",
        min_value=float(limites["gpu_temp_mean_C"]["min"]),
        max_value=float(limites["gpu_temp_mean_C"]["max"]),
        value=float(limites["gpu_temp_mean_C"]["mediana"])
    )
 
    gpu_mem_utilization = st.slider(
        "Utilização média da memória da GPU (%)",
        min_value=float(limites["gpu_mem_utilization_mean"]["min"]),
        max_value=float(limites["gpu_mem_utilization_mean"]["max"]),
        value=float(limites["gpu_mem_utilization_mean"]["mediana"])
    )
 
with col2:
    st.subheader("📊 Memória e CPU")
 
    gpu_mem_used = st.number_input(
        "Memória média utilizada da GPU (MB)",
        min_value=float(limites["gpu_mem_used_mean_MB"]["min"]),
        max_value=float(limites["gpu_mem_used_mean_MB"]["max"]),
        value=float(limites["gpu_mem_used_mean_MB"]["mediana"])
    )
 
    cpu_utilization = st.slider(
        "Utilização da CPU (%)",
        min_value=float(limites["cpu_utilization_percent"]["min"]),
        max_value=float(limites["cpu_utilization_percent"]["max"]),
        value=float(limites["cpu_utilization_percent"]["mediana"])
    )
 
    cpu_freq = st.number_input(
        "Frequência da CPU (MHz)",
        min_value=float(limites["cpu_freq_MHz"]["min"]),
        max_value=float(limites["cpu_freq_MHz"]["max"]),
        value=float(limites["cpu_freq_MHz"]["mediana"])
    )
 
    batch_size = st.number_input(
        "Batch size",
        min_value=int(limites["batch_size"]["min"]),
        max_value=int(limites["batch_size"]["max"]),
        value=int(round(limites["batch_size"]["mediana"])),
        step=1
    )
 
st.divider()
 
with st.expander("💰 Simulação de energia e custo"):
    st.caption(
        "Esta parte não é uma previsão adicional do modelo. "
        "Ela calcula energia e custo a partir da potência prevista."
    )
 
    c1, c2 = st.columns(2)
 
    with c1:
        horas = st.number_input(
            "Horas de operação",
            min_value=1.0,
            value=10.0,
            step=1.0
        )
 
    with c2:
        tarifa = st.number_input(
            "Tarifa de energia (R$/kWh)",
            min_value=0.0,
            value=0.75,
            step=0.05
        )
 
st.divider()
 
if st.button(
    "⚡ Prever potência",
    type="primary",
    use_container_width=True
):
    valores = {
        "cpu_utilization_percent": cpu_utilization,
        "cpu_freq_MHz": cpu_freq,
        "gpu_utilization_mean": gpu_utilization,
        "gpu_mem_utilization_mean": gpu_mem_utilization,
        "gpu_mem_used_mean_MB": gpu_mem_used,
        "gpu_temp_mean_C": gpu_temp,
        "batch_size": batch_size
    }
 
    entrada_numerica = pd.DataFrame(
        [valores],
        columns=colunas_numericas
    )
 
    entrada_numerica[colunas_numericas] = scaler.transform(
        entrada_numerica[colunas_numericas]
    )
 
    entrada_final = pd.DataFrame(
        0.0,
        index=[0],
        columns=colunas_modelo
    )
 
    for coluna in colunas_numericas:
        entrada_final[coluna] = entrada_numerica[coluna]
 
    coluna_gpu = f"gpu_model_{gpu_model}"
 
    if coluna_gpu in entrada_final.columns:
        entrada_final[coluna_gpu] = 1
 
    coluna_workload = f"workload_{workload}"
 
    if coluna_workload in entrada_final.columns:
        entrada_final[coluna_workload] = 1
 
    potencia_w = float(modelo.predict(entrada_final)[0])
    potencia_kw = potencia_w / 1000
 
    energia_kwh = potencia_kw * horas
    custo = energia_kwh * tarifa
 
    st.success("Previsão realizada com sucesso!")
 
    st.subheader("🎯 Resultado do modelo")
 
    r1, r2 = st.columns(2)
 
    r1.metric(
        "Potência prevista",
        f"{numero_br(potencia_w)} W"
    )
 
    r2.metric(
        "Potência em kW",
        f"{numero_br(potencia_kw)} kW"
    )
 
    st.subheader("💡 Simulação operacional")
 
    s1, s2 = st.columns(2)
 
    s1.metric(
        f"Energia estimada em {numero_br(horas, 0)} horas",
        f"{numero_br(energia_kwh)} kWh"
    )
 
    s2.metric(
        "Custo estimado",
        f"R$ {numero_br(custo)}"
    )
 
    st.warning(
        "A energia e o custo são estimativas calculadas supondo que "
        "a potência prevista permaneça constante durante todo o período. "
        "O modelo prevê a potência das GPUs, não o consumo total do servidor "
        "ou do data center."
    )
 
st.divider()
 
st.subheader("📈 Desempenho do modelo de deploy")
 
m1, m2, m3 = st.columns(3)
 
m1.metric(
    "MAE",
    f"{numero_br(metricas['MAE'])} W"
)
 
m2.metric(
    "RMSE",
    f"{numero_br(metricas['RMSE'])} W"
)
 
m3.metric(
    "R²",
    f"{metricas['R2']:.4f}"
)
 
st.caption(
    "Versão otimizada do Random Forest para execução web. "
    "Projeto Final de Machine Learning — Diego e Isaque | SENAC."
)
