import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Previsão de Potência em Cargas de IA",
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

st.markdown("""
<style>

.stApp {
    background: #f7f9fc;
    color: #172033;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    color: #172033;
}

.topo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 18px;
    border-bottom: 1px solid #e3e8ef;
    margin-bottom: 12px;
}

.logo {
    font-size: 1.7rem;
}

.nome-app {
    font-size: 1.05rem;
    font-weight: 700;
    color: #172033;
}

.titulo {
    text-align: center;
    margin-top: 38px;
    font-size: 2.3rem;
    font-weight: 750;
    color: #172033;
}

.subtitulo {
    text-align: center;
    color: #667085;
    font-size: 1rem;
    max-width: 720px;
    margin: 8px auto 35px auto;
}

.card {
    background: #ffffff;
    border: 1px solid #e1e7ef;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 3px 12px rgba(17, 24, 39, 0.035);
}

.resultado-principal {
    background: #ffffff;
    border: 1px solid #b9d5ff;
    border-radius: 18px;
    padding: 34px;
    text-align: center;
    margin-top: 20px;
}

.resultado-label {
    color: #667085;
    font-size: 0.95rem;
}

.resultado-numero {
    color: #172033;
    font-size: 3.5rem;
    line-height: 1.05;
    font-weight: 750;
    margin-top: 8px;
}

.resultado-kw {
    color: #667085;
    font-size: 1.6rem;
    margin-top: 5px;
}

.ajuda {
    text-align: center;
    color: #7a8699;
    font-size: 0.88rem;
    margin-top: 18px;
    margin-bottom: 10px;
}

.secao {
    font-size: 1.3rem;
    font-weight: 700;
    color: #172033;
    margin-top: 30px;
    margin-bottom: 4px;
}

.descricao {
    color: #667085;
    font-size: 0.94rem;
    margin-bottom: 18px;
}

.info-box {
    border-left: 4px solid #2583f7;
    background: #eef6ff;
    padding: 16px 18px;
    border-radius: 8px;
    color: #40516c;
    margin-top: 18px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e1e7ef;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #667085;
}

div[data-testid="stMetricValue"] {
    color: #172033;
}

div[data-testid="stForm"] {
    background: white;
    border: 1px solid #e1e7ef;
    border-radius: 18px;
    padding: 28px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #2583f7;
    color: white;
    border: none;
    border-radius: 10px;
    height: 48px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #126fde;
    color: white;
    border: none;
}

div[data-testid="stPopover"] button {
    min-height: 68px;
    font-size: 1.7rem;
    background: white;
    border: 1px solid #dce3ec;
    border-radius: 13px;
}

div[data-testid="stPopover"] button:hover {
    border-color: #2583f7;
    background: #f4f8ff;
}

div[role="radiogroup"] {
    justify-content: center;
    gap: 18px;
}

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 0.82rem;
    margin-top: 45px;
}

</style>
""", unsafe_allow_html=True)


def numero_br(valor, casas=2):
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


st.markdown("""
<div class="topo">
    <div class="logo">⚡</div>
    <div class="nome-app">Previsão de Potência em Cargas de IA</div>
</div>
""", unsafe_allow_html=True)

menu = st.radio(
    "Navegação",
    ["Previsão", "Sobre", "Metodologia"],
    horizontal=True,
    label_visibility="collapsed"
)


if menu == "Sobre":

    st.markdown(
        '<div class="titulo">Sobre o projeto</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitulo">
        Uma aplicação de Machine Learning desenvolvida para estimar
        a potência consumida por cargas de Inteligência Artificial.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    ### Objetivo

    A proposta é permitir que características de uma carga computacional
    sejam informadas para que o modelo estime a potência total consumida
    pelas GPUs.

    ### Aplicação

    A previsão pode auxiliar no planejamento energético,
    dimensionamento de infraestrutura e análise de custos associados
    à execução de cargas de Inteligência Artificial.

    ### Projeto

    Projeto Final de Machine Learning para Problemas Reais.

    **Integrantes:** Diego e Isaque.

    </div>
    """, unsafe_allow_html=True)

    st.stop()


if menu == "Metodologia":

    st.markdown(
        '<div class="titulo">Metodologia</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitulo">
        Como os dados foram preparados e o modelo foi desenvolvido.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    ### Dados

    O projeto utilizou um dataset público contendo medições de cargas
    de Inteligência Artificial executadas em GPUs NVIDIA H100 e B200.

    A base final utilizada no projeto possui aproximadamente
    **1,44 milhão de registros**.

    ### Modelagem

    Foram comparados:

    - baseline;
    - Regressão Linear;
    - Random Forest;
    - Gradient Boosting.

    O Random Forest apresentou o melhor desempenho e foi escolhido
    como modelo principal.

    ### Deploy

    Para a aplicação web foi utilizada uma versão otimizada do
    Random Forest, reduzindo o tamanho do modelo e mantendo desempenho
    próximo ao modelo completo.

    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE do deploy",
        f"{numero_br(metricas['MAE'])} W"
    )

    c2.metric(
        "RMSE do deploy",
        f"{numero_br(metricas['RMSE'])} W"
    )

    c3.metric(
        "R² do deploy",
        f"{metricas['R2']:.4f}"
    )

    st.stop()


if st.session_state.tela == "entrada":

    st.markdown(
        '<div class="titulo">Previsão de Potência em Cargas de IA</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="subtitulo">
    Informe as características da carga computacional para estimar
    a potência total consumida pelas GPUs.
    </div>
    """, unsafe_allow_html=True)

    with st.form("formulario"):

        st.markdown("### Configuração da carga")

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

        st.markdown("#### Simulação de operação")

        operacao1, operacao2 = st.columns(2)

        horas = operacao1.number_input(
            "Horas de operação",
            min_value=1.0,
            value=1.0,
            step=1.0
        )

        tarifa = operacao2.number_input(
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

    if st.button("← Voltar para parâmetros"):
        st.session_state.tela = "entrada"
        st.rerun()

    st.markdown(
        '<div class="titulo">Resultado da estimativa</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="subtitulo">
    Estimativa calculada a partir dos parâmetros informados.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="resultado-principal">

            <div class="resultado-label">
                POTÊNCIA TOTAL ESTIMADA DAS GPUs
            </div>

            <div class="resultado-numero">
                {numero_br(resultado["potencia_w"])} W
            </div>

            <div class="resultado-kw">
                {numero_br(resultado["potencia_kw"])} kW
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ajuda">ⓘ Clique nos ícones para entender os parâmetros utilizados na previsão.</div>',
        unsafe_allow_html=True
    )

    i1, i2, i3, i4 = st.columns(4)

    with i1:
        with st.popover("🖥️", use_container_width=True):

            st.markdown("### Uso da GPU")

            st.metric(
                "Utilização",
                f'{resultado["gpu_utilization"]:.1f}%'
            )

            st.write(
                "Indica quanto da capacidade de processamento da GPU "
                "está sendo utilizada pela carga."
            )

            st.caption(
                f'Modelo utilizado: {resultado["gpu"]}'
            )

    with i2:
        with st.popover("▦", use_container_width=True):

            st.markdown("### Memória da GPU")

            st.metric(
                "Memória utilizada",
                f'{numero_br(resultado["gpu_mem_used"], 0)} MB'
            )

            st.write(
                "Representa a quantidade de memória da GPU ocupada "
                "durante a execução da carga."
            )

            st.caption(
                f'Utilização da memória: '
                f'{resultado["gpu_mem_utilization"]:.1f}%'
            )

    with i3:
        with st.popover("⚙️", use_container_width=True):

            st.markdown("### Uso da CPU")

            st.metric(
                "Utilização",
                f'{resultado["cpu_utilization"]:.1f}%'
            )

            st.write(
                "Mostra quanto do processador está sendo utilizado "
                "durante a execução."
            )

            st.caption(
                f'Frequência: {numero_br(resultado["cpu_freq"], 0)} MHz'
            )

    with i4:
        with st.popover("🌡️", use_container_width=True):

            st.markdown("### Temperatura da GPU")

            st.metric(
                "Temperatura média",
                f'{resultado["gpu_temp"]:.1f} °C'
            )

            st.write(
                "Temperatura média observada na GPU durante "
                "a execução da carga."
            )

            st.caption(
                f'Workload: {resultado["workload"]}'
            )

    st.markdown(
        '<div class="secao">Estimativa de operação</div>',
        unsafe_allow_html=True
    )

    energia_col, custo_col = st.columns(2)

    energia_col.metric(
        "Energia estimada",
        f'{numero_br(resultado["energia"])} kWh',
        f'em {numero_br(resultado["horas"], 0)} hora(s)',
        delta_color="off"
    )

    custo_col.metric(
        "Custo estimado",
        f'R$ {numero_br(resultado["custo"])}',
        f'R$ {numero_br(resultado["tarifa"])} / kWh',
        delta_color="off"
    )

    st.markdown("""
    <div class="info-box">
    A energia e o custo são estimativas calculadas a partir da potência
    prevista. O cálculo considera que essa potência permanece constante
    durante o período informado.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="secao">Sobre a precisão do modelo</div>',
        unsafe_allow_html=True
    )

    st.write(
        f'O modelo de deploy apresentou **R² de '
        f'{metricas["R2"]:.4f}**, indicando que conseguiu explicar '
        f'aproximadamente **{metricas["R2"] * 100:.1f}%** da variação '
        f'da potência no conjunto de teste.'
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

        st.caption(
            "MAE representa o erro absoluto médio. "
            "RMSE dá maior peso a erros maiores. "
            "R² indica quanto da variação dos dados o modelo consegue explicar."
        )


st.markdown("""
<div class="footer">
Projeto Final de Machine Learning • Diego e Isaque • SENAC
</div>
""", unsafe_allow_html=True)
