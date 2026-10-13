# Previsão de Potência em Cargas de Inteligência Artificial

## Integrantes

- Diego
- Isaque

## Sobre o projeto

Este projeto foi desenvolvido como trabalho final de Machine Learning do curso Técnico em Inteligência Artificial do SENAC.

O objetivo é prever a potência elétrica consumida por cargas de Inteligência Artificial utilizando informações como utilização da GPU, memória, CPU, temperatura da GPU, modelo da GPU e tipo de workload.

## Problema

É possível prever a potência consumida por uma carga de Inteligência Artificial com base nas características de utilização do hardware?

## Dataset

Foi utilizado o dataset público:

**High-resolution AI Data Center Training Workloads Dataset**

O conjunto possui medições de cargas de IA executadas em GPUs NVIDIA H100 e B200.

Após a preparação dos dados, foram utilizados aproximadamente 1,44 milhão de registros.

## Variável-alvo

A variável prevista pelo modelo é:

`total_gpu_power_W`

Ela representa a potência total das GPUs em watts.

## Etapas realizadas

- Análise Exploratória dos Dados
- Tratamento dos dados
- Divisão entre treino e teste
- Padronização das variáveis
- Criação de baseline
- Regressão Linear
- Random Forest
- Gradient Boosting
- Comparação dos modelos
- Ajuste de hiperparâmetros
- Explicabilidade
- Análise de viés
- Simulação de impacto de negócio
- Análise crítica

## Modelos utilizados

Foram avaliados:

- Baseline
- Regressão Linear
- Random Forest
- Gradient Boosting

O Random Forest apresentou o melhor desempenho e foi selecionado para ajuste de hiperparâmetros.

## Melhor modelo

O melhor resultado foi obtido pelo **Random Forest Ajustado**.

- MAE: aproximadamente 150,20 W
- RMSE: aproximadamente 317,76 W
- R²: aproximadamente 0,9732

## Explicabilidade

A utilização média da GPU foi a variável de maior importância para o modelo, seguida pela temperatura média da GPU e pela utilização da memória da GPU.

## Análise de viés

O modelo apresentou diferenças de erro entre os grupos analisados.

### Modelo de GPU

- B200: aproximadamente 171 W
- H100: aproximadamente 129 W

### Tipo de carga

- Image Generation: aproximadamente 46 W
- LLM: aproximadamente 231 W

O modelo apresentou maior dificuldade para realizar previsões em cargas LLM.

## Impacto de negócio

Em comparação ao baseline, o modelo final reduziu o erro médio em aproximadamente 91,4%.

Em uma simulação considerando 720 horas de operação por mês e tarifa hipotética de R$ 0,75 por kWh, foi obtida uma diferença estimada de aproximadamente R$ 861,78.

Esse valor representa uma simulação baseada em premissas e não uma economia financeira real comprovada.

## Como executar

1. Abra o notebook `projeto_final_machine_learning.ipynb` no Google Colab.
2. Faça upload do dataset utilizado no projeto.
3. Execute as células na ordem apresentada.
4. Aguarde o processamento dos dados e o treinamento dos modelos.
5. Analise as métricas e gráficos gerados.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab

## Conclusão

Os resultados mostraram que é possível utilizar Machine Learning para estimar a potência consumida por cargas de Inteligência Artificial.

Entre os modelos avaliados, o Random Forest Ajustado apresentou o melhor desempenho, com R² de aproximadamente 0,973 e erros significativamente menores que o baseline.
