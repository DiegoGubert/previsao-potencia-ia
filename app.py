import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Previsão de Potência em IA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_potencia_ia_deploy.joblib")


pacote = carregar_modelo()

modelo = pacote["modelo"]
scaler = pacote["scaler"]
colunas_numericas = pacote["colunas_numericas"]
colunas_modelo = pacote["colunas_modelo"]
limites = pacote["limites_entradas"]
metricas = pacote["metricas"]


if "tela" not in st.session_state:
    st.session_state.tela = "entrada"

if "resultado" not in st.session_state:
    st.session_state.resultado = None


def numero_br(valor, casas=2):
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


st.html("""
<style>

.stApp {
    background: #ffffff;
    color: #0f172a;
}

.block-container {
    max-width: 1080px;
    padding-top: 4.5rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: rgba(255,255,255,0.96);
}


/* CABEÇALHO */

.cabecalho {
    display: flex;
    align-items: center;
    min-height: 52px;
}

.marca {
    display: flex;
    align-items: center;
    gap: 10px;
    color: #0f172a !important;
    font-size: 1.08rem;
    font-weight: 700;
}

.marca span {
    color: #0f172a !important;
}

.marca-icone {
    font-size: 1.65rem;
    color: #1677ff !important;
}

.linha-cabecalho {
    height: 1px;
    background: #dfe5ec;
    margin-top: 8px;
    margin-bottom: 26px;
}


/* NAVEGAÇÃO */

div[data-testid="stSegmentedControl"] button {
    color: #475569 !important;
    background: #ffffff !important;
    border-color: #e2e8f0 !important;
}

div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    color: #1677ff !important;
    background: #eef6ff !important;
    border-color: #c7ddff !important;
}


/* TÍTULOS */

.hero-title {
    color: #0f172a;
    font-size: 2.25rem;
    font-weight: 750;
    letter-spacing: -0.035em;
    text-align: center;
    margin-top: 24px;
}

.hero-subtitle {
    max-width: 680px;
    margin: 8px auto 32px auto;
    text-align: center;
    color: #64748b;
    font-size: 0.98rem;
    line-height: 1.55;
}

.section-title {
    color: #0f172a;
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 3px;
}

.section-description {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 16px;
}


/* FORMULÁRIO */

div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.03);
}

.stButton > button,
.stFormSubmitButton > button {
    background: #1677ff;
    color: #ffffff;
    border: none;
    border-radius: 9px;
    min-height: 46px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #0868df;
    color: #ffffff;
    border: none;
}


/* RESULTADO PRINCIPAL */

.resultado-card {
    background: #f7faff;
    border: 1px solid #a8ceff;
    border-radius: 16px;
    padding: 34px 24px;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 18px;
}

.resultado-label {
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 650;
    letter-spacing: 0.035em;
}

.resultado-numero {
    color: #0f172a;
    font-size: 3.35rem;
    line-height: 1.05;
    font-weight: 780;
    letter-spacing: -0.045em;
    margin-top: 8px;
}

.resultado-kw {
    color: #64748b;
    font-size: 1.45rem;
    margin-top: 7px;
}

.resultado-contexto {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 8px;
}


/* ÍCONES DOS PARÂMETROS */

.ajuda-icones {
    text-align: center;
    color: #64748b;
    font-size: 0.86rem;
    margin: 14px 0 10px 0;
}

div[data-testid="stPopover"] > button {
    min-height: 56px !important;
    height: 56px !important;
    background: #ffffff !important;
    border: 1px solid #d7e0eb !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    color: #1e3a5f !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 0 10px !important;
}

div[data-testid="stPopover"] > button:hover {
    background: #f5f9ff !important;
    border-color: #1677ff !important;
    color: #1677ff !important;
}

div[data-testid="stPopover"] svg {
    width: 14px !important;
    height: 14px !important;
}


/* MÉTRICAS */

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 13px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
}

div[data-testid="stMetricValue"] {
    color: #0f172a;
}


/* AVISO */

.info-box {
    background: #f7faff;
    border-left: 4px solid #1677ff;
    border-radius: 8px;
    padding: 15px 18px;
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.55;
    margin-top: 18px;
}


/* RODAPÉ */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 42px;
}

</style>
""")


# CABEÇALHO

cab1, cab2 = st.columns([1.8, 1.2], vertical_alignment="center")

with cab1:
    st.html("""
    <div class="cabecalho">
        <div class="marca">
            <span class="marca-icone">⚡</span>
            <span>Previsão de Potência em Cargas de IA</span>
        </div>
    </div>
    """)

with cab2:
    menu = st.segmented_control(
        "Navegação",
        ["Previsão", "Sobre", "Metodologia"],
        default="Previsão",
        label_visibility="collapsed"
    )

st.html('<div class="linha-cabecalho"></div>')


# SOBRE

if menu == "Sobre":

    st.html('<div class="hero-title">Sobre o projeto</div>')

    st.html("""
    <div class="hero-subtitle">
        Aplicação desenvolvida para estimar a potência consumida
        por cargas de Inteligência Artificial.
    </div>
    """)

    with st.container(border=True):

        st.subheader("Objetivo")

        st.write(
            "Utilizar Machine Learning para estimar a potência total "
            "consumida pelas GPUs a partir das características "
            "da carga computacional."
        )

        st.subheader("Aplicação")

        st.write(
            "A estimativa pode auxiliar no planejamento energético, "
            "dimensionamento de infraestrutura e análise de custos "
            "relacionados à execução de cargas de IA."
        )

        st.subheader("Equipe")

        st.write("Diego e Isaque — SENAC")

    st.stop()


# METODOLOGIA

if menu == "Metodologia":

    st.html('<div class="hero-title">Metodologia</div>')

    st.html("""
    <div class="hero-subtitle">
        Resumo das etapas utilizadas para desenvolver e validar
        o modelo de Machine Learning.
    </div>
    """)

    with st.container(border=True):

        st.subheader("Dataset")

        st.write(
            "Foram utilizadas medições de cargas de Inteligência Artificial "
            "executadas em GPUs NVIDIA H100 e B200."
        )

        st.write(
            "A base preparada possui aproximadamente "
            "**1,44 milhão de registros**."
        )

        st.subheader("Modelos avaliados")

        st.write(
            "Baseline, Regressão Linear, Random Forest e Gradient Boosting."
        )

        st.subheader("Modelo escolhido")

        st.write(
            "O Random Forest apresentou o melhor desempenho geral. "
            "Uma versão otimizada foi utilizada no mini deploy."
        )

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

    st.stop()


# TELA 1 — ENTRADA

if st.session_state.tela == "entrada":

    st.html(
        '<div class="hero-title">'
        'Previsão de Potência em Cargas de IA'
        '</div>'
    )

    st.html("""
    <div class="hero-subtitle">
        Informe as características da carga computacional.
        O modelo utilizará esses parâmetros para estimar
        a potência total das GPUs.
    </div>
    """)

    with st.form("form_previsao"):

        st.subheader("Configuração da carga")

        st.caption(
            "Preencha os parâmetros abaixo e clique em Calcular previsão."
        )

        esquerda, direita = st.columns(2, gap="large")

        with esquerda:

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
                float(limites["gpu_utilization_mean"]["min"]),
                float(limites["gpu_utilization_mean"]["max"]),
                float(limites["gpu_utilization_mean"]["mediana"])
            )

            gpu_temp = st.slider(
                "Temperatura média da GPU (°C)",
                float(limites["gpu_temp_mean_C"]["min"]),
                float(limites["gpu_temp_mean_C"]["max"]),
                float(limites["gpu_temp_mean_C"]["mediana"])
            )

        with direita:

            gpu_mem_utilization = st.slider(
                "Utilização da memória da GPU (%)",
                float(limites["gpu_mem_utilization_mean"]["min"]),
                float(limites["gpu_mem_utilization_mean"]["max"]),
                float(limites["gpu_mem_utilization_mean"]["mediana"])
            )

            gpu_mem_used = st.number_input(
                "Memória utilizada da GPU (MB)",
                min_value=float(
                    limites["gpu_mem_used_mean_MB"]["min"]
                ),
                max_value=float(
                    limites["gpu_mem_used_mean_MB"]["max"]
                ),
                value=float(
                    limites["gpu_mem_used_mean_MB"]["mediana"]
                )
            )

            cpu_utilization = st.slider(
                "Utilização da CPU (%)",
                float(limites["cpu_utilization_percent"]["min"]),
                float(limites["cpu_utilization_percent"]["max"]),
                float(limites["cpu_utilization_percent"]["mediana"])
            )

            cpu_freq = st.number_input(
                "Frequência da CPU (MHz)",
                min_value=float(
                    limites["cpu_freq_MHz"]["min"]
                ),
                max_value=float(
                    limites["cpu_freq_MHz"]["max"]
                ),
                value=float(
                    limites["cpu_freq_MHz"]["mediana"]
                )
            )

            batch_size = st.number_input(
                "Batch size",
                min_value=int(limites["batch_size"]["min"]),
                max_value=int(limites["batch_size"]["max"]),
                value=int(
                    round(limites["batch_size"]["mediana"])
                ),
                step=1
            )

        st.divider()

        st.subheader("Simulação de operação")

        sim1, sim2 = st.columns(2)

        horas = sim1.number_input(
            "Horas de operação",
            min_value=1.0,
            value=1.0,
            step=1.0
        )

        tarifa = sim2.number_input(
            "Tarifa de energia (R$/kWh)",
            min_value=0.0,
            value=0.75,
            step=0.05
        )

        calcular = st.form_submit_button(
            "Calcular previsão →",
            use_container_width=True
        )

    if calcular:

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

        potencia_w = float(
            modelo.predict(entrada_final)[0]
        )

        potencia_kw = potencia_w / 1000
        energia = potencia_kw * horas
        custo = energia * tarifa

        st.session_state.resultado = {
            "potencia_w": potencia_w,
            "potencia_kw": potencia_kw,
            "energia": energia,
            "custo": custo,
            "horas": horas,
            "tarifa": tarifa,
            "gpu": gpu_model,
            "workload": workload,
            "gpu_utilization": gpu_utilization,
            "gpu_mem_utilization": gpu_mem_utilization,
            "gpu_mem_used": gpu_mem_used,
            "cpu_utilization": cpu_utilization,
            "cpu_freq": cpu_freq,
            "gpu_temp": gpu_temp,
            "batch_size": batch_size
        }

        st.session_state.tela = "resultado"
        st.rerun()


# TELA 2 — RESULTADO

else:

    resultado = st.session_state.resultado

    if resultado is None:
        st.session_state.tela = "entrada"
        st.rerun()

    if st.button("← Voltar para parâmetros"):
        st.session_state.tela = "entrada"
        st.rerun()

    st.html(
        '<div class="hero-title">'
        'Resultado da estimativa'
        '</div>'
    )

    st.html("""
    <div class="hero-subtitle">
        Resultado calculado a partir das características
        informadas para a carga de IA.
    </div>
    """)

    st.html(
        f"""
        <div class="resultado-card">

            <div class="resultado-label">
                POTÊNCIA TOTAL ESTIMADA DAS GPUs
            </div>

            <div class="resultado-numero">
                {numero_br(resultado["potencia_w"])} W
            </div>

            <div class="resultado-kw">
                {numero_br(resultado["potencia_kw"])} kW
            </div>

            <div class="resultado-contexto">
                {resultado["gpu"]} • {resultado["workload"]}
            </div>

        </div>
        """
    )

    st.html("""
    <div class="ajuda-icones">
        ⓘ Clique nos ícones para ver os parâmetros utilizados na previsão.
    </div>
    """)

    i1, i2, i3, i4 = st.columns(4, gap="medium")

    with i1:

        with st.popover(
            "🖥️  GPU",
            use_container_width=True
        ):

            st.subheader("Uso da GPU")

            st.metric(
                "Utilização",
                f'{resultado["gpu_utilization"]:.1f}%'
            )

            st.write(
                "Indica quanto da capacidade de processamento "
                "da GPU está sendo utilizada."
            )

            st.caption(
                f'Modelo: {resultado["gpu"]}'
            )

    with i2:

        with st.popover(
            "▰  Memória",
            use_container_width=True
        ):

            st.subheader("Memória da GPU")

            st.metric(
                "Memória utilizada",
                f'{numero_br(resultado["gpu_mem_used"], 0)} MB'
            )

            st.write(
                "Representa a quantidade de memória da GPU "
                "ocupada pela carga."
            )

            st.caption(
                f'Utilização: '
                f'{resultado["gpu_mem_utilization"]:.1f}%'
            )

    with i3:

        with st.popover(
            "⚙️  CPU",
            use_container_width=True
        ):

            st.subheader("Uso da CPU")

            st.metric(
                "Utilização",
                f'{resultado["cpu_utilization"]:.1f}%'
            )

            st.write(
                "Mostra quanto do processador está sendo "
                "utilizado durante a execução."
            )

            st.caption(
                f'Frequência: '
                f'{numero_br(resultado["cpu_freq"], 0)} MHz'
            )

    with i4:

        with st.popover(
            "🌡️  Temperatura",
            use_container_width=True
        ):

            st.subheader("Temperatura da GPU")

            st.metric(
                "Temperatura",
                f'{resultado["gpu_temp"]:.1f} °C'
            )

            st.write(
                "Temperatura média considerada pelo modelo "
                "para realizar a previsão."
            )

    st.html(
        '<div class="section-title">'
        'Estimativa de operação'
        '</div>'
    )

    st.html(
        '<div class="section-description">'
        'Simulação calculada a partir da potência prevista.'
        '</div>'
    )

    energia_col, custo_col = st.columns(2)

    energia_col.metric(
        "Energia estimada",
        f'{numero_br(resultado["energia"])} kWh'
    )

    custo_col.metric(
        "Custo estimado",
        f'R$ {numero_br(resultado["custo"])}'
    )

    st.html(
        f"""
        <div class="info-box">
            A energia e o custo são estimativas calculadas a partir
            da potência prevista, considerando que essa potência
            permaneça constante durante
            {numero_br(resultado["horas"], 0)} hora(s).

            O modelo prevê a potência das GPUs, e não o consumo
            elétrico total do servidor ou do data center.
        </div>
        """
    )

    st.html(
        '<div class="section-title">'
        'Sobre a precisão do modelo'
        '</div>'
    )

    st.write(
        f'O modelo utilizado no deploy apresentou **R² de '
        f'{metricas["R2"]:.4f}**. Isso significa que ele explicou '
        f'aproximadamente **{metricas["R2"] * 100:.1f}%** '
        f'da variação da potência no conjunto de teste.'
    )

    with st.expander("Ver métricas técnicas"):

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "MAE",
            f'{numero_br(metricas["MAE"])} W'
        )

        m2.metric(
            "RMSE",
            f'{numero_br(metricas["RMSE"])} W'
        )

        m3.metric(
            "R²",
            f'{metricas["R2"]:.4f}'
        )


st.html("""
<div class="footer">
    Projeto Final de Machine Learning • Diego e Isaque • SENAC
</div>
""")
