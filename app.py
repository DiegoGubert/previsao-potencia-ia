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


try:
    pacote = carregar_modelo()
except Exception as erro:
    st.error(f"Não foi possível carregar o modelo: {erro}")
    st.stop()


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


st.markdown("""
<style>

.stApp {
    background: #ffffff;
    color: #0f172a;
}

.block-container {
    max-width: 1080px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: #ffffff;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

.app-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 16px;
    border-bottom: 1px solid #e7edf5;
    margin-bottom: 22px;
}

.app-logo {
    color: #1677ff;
    font-size: 1.65rem;
}

.app-name {
    color: #0f172a;
    font-size: 1rem;
    font-weight: 700;
}

.hero-title {
    font-size: 2.35rem;
    font-weight: 750;
    letter-spacing: -0.035em;
    text-align: center;
    color: #0f172a;
    margin-top: 28px;
}

.hero-subtitle {
    max-width: 680px;
    margin: 8px auto 32px auto;
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    line-height: 1.6;
}

.section-title {
    color: #0f172a;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 3px;
}

.section-text {
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 18px;
}

div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.025);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e2e8f0;
    border-radius: 16px;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px;
}

div[data-testid="stMetricLabel"] {
    color: #64748b;
}

div[data-testid="stMetricValue"] {
    color: #0f172a;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #1677ff;
    color: #ffffff;
    border: 0;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 650;
    transition: 0.15s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #0868df;
    color: #ffffff;
    border: 0;
}

div[data-testid="stPopover"] > button {
    background: #ffffff;
    border: 1px solid #d8e1ec;
    border-radius: 14px;
    min-height: 76px;
    font-size: 1.65rem;
    color: #1677ff;
}

div[data-testid="stPopover"] > button:hover {
    background: #f7faff;
    border-color: #1677ff;
    color: #1677ff;
}

.result-card {
    background: #f7faff;
    border: 1px solid #b9d6ff;
    border-radius: 18px;
    padding: 34px 20px;
    text-align: center;
    margin: 20px 0 16px 0;
}

.result-label {
    color: #475569;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.result-value {
    color: #0f172a;
    font-size: 3.5rem;
    font-weight: 780;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-top: 8px;
}

.result-kw {
    color: #64748b;
    font-size: 1.55rem;
    margin-top: 6px;
}

.result-context {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 9px;
}

.icon-help {
    text-align: center;
    color: #64748b;
    font-size: 0.87rem;
    margin: 18px 0 9px 0;
}

.friendly-box {
    background: #f8fbff;
    border-left: 4px solid #1677ff;
    border-radius: 8px;
    padding: 16px 18px;
    color: #475569;
    margin-top: 18px;
    line-height: 1.55;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)


header_col, nav_col = st.columns([1.8, 1.2])

with header_col:
    st.markdown("""
    <div class="app-header">
        <span class="app-logo">⚡</span>
        <span class="app-name">Previsão de Potência em IA</span>
    </div>
    """, unsafe_allow_html=True)

with nav_col:
    try:
        menu = st.segmented_control(
            "Navegação",
            ["Previsão", "Sobre", "Metodologia"],
            default="Previsão",
            label_visibility="collapsed"
        )
    except AttributeError:
        menu = st.radio(
            "Navegação",
            ["Previsão", "Sobre", "Metodologia"],
            horizontal=True,
            label_visibility="collapsed"
        )


if menu == "Sobre":

    st.markdown(
        '<div class="hero-title">Sobre o projeto</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero-subtitle">
        Uma ferramenta desenvolvida para estimar a potência consumida
        por cargas de Inteligência Artificial.
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):

        st.markdown("### Objetivo")

        st.write(
            "O projeto utiliza Machine Learning para estimar a potência "
            "total consumida pelas GPUs a partir das características "
            "da carga computacional."
        )

        st.markdown("### Aplicação")

        st.write(
            "A estimativa pode auxiliar no planejamento energético, "
            "dimensionamento de infraestrutura e análise de custos "
            "relacionados à execução de cargas de IA."
        )

        st.markdown("### Equipe")

        st.write("Diego e Isaque — SENAC")

    st.stop()


if menu == "Metodologia":

    st.markdown(
        '<div class="hero-title">Metodologia</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero-subtitle">
        Resumo das etapas utilizadas para construir e avaliar o modelo.
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):

        st.markdown("### Dataset")

        st.write(
            "Foram utilizadas medições de cargas de Inteligência Artificial "
            "executadas em GPUs NVIDIA H100 e B200."
        )

        st.write(
            "A base preparada para o projeto possui aproximadamente "
            "**1,44 milhão de registros**."
        )

        st.markdown("### Modelos avaliados")

        st.write(
            "Baseline, Regressão Linear, Random Forest e Gradient Boosting."
        )

        st.markdown("### Modelo selecionado")

        st.write(
            "O Random Forest apresentou o melhor desempenho geral. "
            "Uma versão otimizada foi utilizada na aplicação web."
        )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE do deploy",
        f"{numero_br(metricas['MAE'])} W"
    )

    m2.metric(
        "RMSE do deploy",
        f"{numero_br(metricas['RMSE'])} W"
    )

    m3.metric(
        "R² do deploy",
        f"{metricas['R2']:.4f}"
    )

    st.stop()


if st.session_state.tela == "entrada":

    st.markdown(
        '<div class="hero-title">Previsão de Potência em Cargas de IA</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero-subtitle">
        Informe as características da carga computacional.
        O modelo utilizará esses parâmetros para estimar
        a potência total das GPUs.
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_previsao"):

        st.markdown(
            '<div class="section-title">Configuração da carga</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-text">'
            'Preencha os parâmetros abaixo e clique em Calcular previsão.'
            '</div>',
            unsafe_allow_html=True
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
                min_value=float(
                    limites["gpu_utilization_mean"]["min"]
                ),
                max_value=float(
                    limites["gpu_utilization_mean"]["max"]
                ),
                value=float(
                    limites["gpu_utilization_mean"]["mediana"]
                )
            )

            gpu_temp = st.slider(
                "Temperatura média da GPU (°C)",
                min_value=float(
                    limites["gpu_temp_mean_C"]["min"]
                ),
                max_value=float(
                    limites["gpu_temp_mean_C"]["max"]
                ),
                value=float(
                    limites["gpu_temp_mean_C"]["mediana"]
                )
            )

        with direita:

            gpu_mem_utilization = st.slider(
                "Utilização da memória da GPU (%)",
                min_value=float(
                    limites["gpu_mem_utilization_mean"]["min"]
                ),
                max_value=float(
                    limites["gpu_mem_utilization_mean"]["max"]
                ),
                value=float(
                    limites["gpu_mem_utilization_mean"]["mediana"]
                )
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
                min_value=float(
                    limites["cpu_utilization_percent"]["min"]
                ),
                max_value=float(
                    limites["cpu_utilization_percent"]["max"]
                ),
                value=float(
                    limites["cpu_utilization_percent"]["mediana"]
                )
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
                min_value=int(
                    limites["batch_size"]["min"]
                ),
                max_value=int(
                    limites["batch_size"]["max"]
                ),
                value=int(
                    round(limites["batch_size"]["mediana"])
                ),
                step=1
            )

        st.markdown("---")
        st.markdown("#### Simulação de operação")

        simulacao1, simulacao2 = st.columns(2)

        horas = simulacao1.number_input(
            "Horas de operação",
            min_value=1.0,
            value=1.0,
            step=1.0
        )

        tarifa = simulacao2.number_input(
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


else:

    resultado = st.session_state.resultado

    if resultado is None:
        st.session_state.tela = "entrada"
        st.rerun()

    if st.button("← Voltar para parâmetros"):
        st.session_state.tela = "entrada"
        st.rerun()

    st.markdown(
        '<div class="hero-title">Resultado da estimativa</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="hero-subtitle">
        Resultado calculado a partir das características
        informadas para a carga de IA.
    </div>
    """, unsafe_allow_html=True)

    resultado_html = (
        '<div class="result-card">'
        '<div class="result-label">POTÊNCIA TOTAL ESTIMADA DAS GPUs</div>'
        f'<div class="result-value">{numero_br(resultado["potencia_w"])} W</div>'
        f'<div class="result-kw">{numero_br(resultado["potencia_kw"])} kW</div>'
        f'<div class="result-context">{resultado["gpu"]} • {resultado["workload"]}</div>'
        '</div>'
    )

    st.markdown(
        resultado_html,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="icon-help">'
        'ⓘ Clique nos ícones para ver os parâmetros utilizados na previsão.'
        '</div>',
        unsafe_allow_html=True
    )

    icone1, icone2, icone3, icone4 = st.columns(4)

    with icone1:

        with st.popover(
            "🖥️",
            use_container_width=True
        ):

            st.markdown("#### Uso da GPU")

            st.metric(
                "Utilização",
                f'{resultado["gpu_utilization"]:.1f}%'
            )

            st.write(
                "Mostra quanto da capacidade de processamento "
                "da GPU está sendo utilizada."
            )

            st.caption(
                f'Modelo: {resultado["gpu"]}'
            )

    with icone2:

        with st.popover(
            "▦",
            use_container_width=True
        ):

            st.markdown("#### Memória da GPU")

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

    with icone3:

        with st.popover(
            "⚙️",
            use_container_width=True
        ):

            st.markdown("#### CPU")

            st.metric(
                "Utilização",
                f'{resultado["cpu_utilization"]:.1f}%'
            )

            st.write(
                "Indica quanto do processador está sendo "
                "utilizado durante a execução."
            )

            st.caption(
                f'Frequência: '
                f'{numero_br(resultado["cpu_freq"], 0)} MHz'
            )

    with icone4:

        with st.popover(
            "🌡️",
            use_container_width=True
        ):

            st.markdown("#### Temperatura da GPU")

            st.metric(
                "Temperatura",
                f'{resultado["gpu_temp"]:.1f} °C'
            )

            st.write(
                "Temperatura média da GPU considerada "
                "durante a previsão."
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Estimativa de operação</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Simulação calculada a partir da potência prevista.'
        '</div>',
        unsafe_allow_html=True
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

    st.markdown(
        (
            '<div class="friendly-box">'
            'A energia e o custo são estimativas calculadas a partir '
            'da potência prevista, considerando que essa potência '
            f'permaneça constante durante {numero_br(resultado["horas"], 0)} '
            'hora(s). '
            'O modelo prevê a potência das GPUs e não o consumo elétrico '
            'total do servidor ou do data center.'
            '</div>'
        ),
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Sobre a precisão do modelo</div>',
        unsafe_allow_html=True
    )

    st.write(
        f'O modelo utilizado no deploy apresentou **R² de '
        f'{metricas["R2"]:.4f}**. Isso significa que ele explicou '
        f'aproximadamente **{metricas["R2"] * 100:.1f}%** da variação '
        f'da potência no conjunto de teste.'
    )

    with st.expander("Ver métricas técnicas"):

        metrica1, metrica2, metrica3 = st.columns(3)

        metrica1.metric(
            "MAE",
            f'{numero_br(metricas["MAE"])} W'
        )

        metrica2.metric(
            "RMSE",
            f'{numero_br(metricas["RMSE"])} W'
        )

        metrica3.metric(
            "R²",
            f'{metricas["R2"]:.4f}'
        )

        st.caption(
            "MAE: erro absoluto médio. "
            "RMSE: penaliza mais os erros maiores. "
            "R²: proporção da variação da potência explicada pelo modelo."
        )


st.markdown("""
<div class="footer">
    Projeto Final de Machine Learning • Diego e Isaque • SENAC
</div>
""", unsafe_allow_html=True)
