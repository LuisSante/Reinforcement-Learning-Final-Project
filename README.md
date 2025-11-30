# Relatório: Algoritmo de Aprendizado por Reforço Multi-Agente Speaker-Listener
**Autores:** Luis Sante & Joel Perca & Andres de la Puente 
**Data:** 29/11/2025

## 1. Introdução e Objetivo
O objetivo deste projeto é desenvolver e implementar um novo algoritmo de Aprendizado por Reforço Multi-Agente (MARL) ou otimizar um algoritmo existente, como o **MATD3 (Multi-Agent Twin Delayed DDPG)**, para o ambiente **Speaker-Listener**.  
O critério de sucesso é que o agente *Listener* consiga navegar até o alvo de forma mais eficiente, ou seja, alcançar uma **pontuação média populacional superior a -60**.

## 2. Configurações e Metodologia de Treinamento
O treinamento foi conduzido em múltiplos ciclos, cada um com uma combinação distinta de hiperparâmetros. Cada iteração representa milhares de passos no ambiente, totalizando **milhões de interações**. O objetivo central foi avaliar a **convergência** e a **estabilidade** do agente *Listener* ao longo de diferentes regimes de treinamento.

### 2.1 Variações nas Configurações ao Longo do Treinamento
As execuções apresentadas neste relatório utilizaram ajustes iterativos nos principais componentes do algoritmo. Em cada rodada foram modificados elementos estruturais (como a arquitetura da rede), dinâmicos (exploração e taxas de aprendizado) e evolutivos (tamanho da população e frequência de atualização). O foco não foi comparar configurações específicas, mas **examinar como diferentes regimes afetam o comportamento do sistema Speaker-Listener**.

| Categoria | Ajuste Realizado | Razão |
|----------|------------------|--------|
| **Arquitetura da Rede** | Capacidade e profundidade da rede | Testar maior expressividade do modelo |
| **População Evolutiva** | Número de agentes na população | Aumentar diversidade para HPO |
| **Batch Size** | Tamanho do batch | Controlar estabilidade do gradiente |
| **Exploração** | Intensidade do ruído | Ajustar exploração vs. convergência |
| **Learning Rates** | Taxas do ator e crítico | Regular velocidade de aprendizado |
| **TAU (Redes-Alvo)** | Ritmo de atualização | Suavizar ou acelerar adaptação |
| **Policy/Update Freq** | Frequência de atualizações da política | Melhorar estabilidade antes da atualização |
| **Evo Steps** | Frequência de ciclos evolutivos | Controlar responsividade da busca evolutiva |

De forma geral, cada execução explorou um ponto diferente do espaço de hiperparâmetros. Ainda que as combinações exatas variem, observaram-se padrões consistentes:  
- **Estabilidade crescente após ~50 iterações**, mesmo com mudanças estruturais.  
- **Aumento da volatilidade** quando a exploração era mais intensa ou as atualizações do TD3 eram menos espaçadas.  
- **Recuperação rápida após quedas profundas**, indicando resiliência do método.

Assim, os treinamentos de 0–2M, 0–3M e 0–5M iterações devem ser entendidos como **variações controladas** dentro de um mesmo experimento, cada um ilustrando como ajustes moderados impactam a trajetória de aprendizado do sistema.


## 3. Análise de Resultados
Os resultados são apresentados em gráficos que mostram a evolução das pontuações médias ao longo do treinamento, em diferentes escalas.

### 3.1 Treinamento Inicial (0 a 2 milhões de iterações)
O desempenho melhora rapidamente nas primeiras iterações: de cerca de **-100** para a faixa entre **-50 e -75**.

<p align="center">
  <img src="Projeto%20Final/models/MATD3_backup/training_scores_evolution.png" width="1000">
</p>

Observa-se uma **alta volatilidade**, com quedas para **-150** (iteração ~100) e até **-210** (iteração ~135).  
Apesar disso, o agente atinge a meta de **-60** já a partir da iteração ~20.  
Entre ~150 e ~200 iterações, opera de forma mais estável, com valores frequentemente próximos de **-35**.

### 3.2 Treinamento Ampliado (0 a 3 milhões de iterações)
O segundo gráfico aprofunda a observação da fase inicial.

<p align="center">
  <img src="Projeto%20Final/models/MATD3_backup3/training_scores_evolution.png" width="1000">
</p>

- **Estabilização:** Após ~50 iterações, a pontuação média fica entre **-50 e -75**.  
- **Volatilidade extrema:** Há quedas severas abaixo de **-250** (iterações ~110, ~190, ~250), indicando falhas críticas na política ou na comunicação Speaker–Listener.  
- **Meta mantida:** Apesar das quedas, o agente recupera rapidamente o desempenho e frequentemente supera o limiar de **-60**.

### 3.3 Treinamento Estendido (0 a 5 milhões de iterações)
O treinamento prolongado revela o comportamento de longo prazo.

<p align="center">
  <img src="Projeto%20Final/models/MATD3_backup2/training_scores_evolution.png" width="1000">
</p>

- **Consistência:** Após ~50 iterações, a pontuação média permanece entre **-30 e -70**.  
- **Sucesso consistente:** O sistema supera de forma contínua o limiar de **-60**, alcançando picos entre **-30 e -40**.  
- **Volatilidade persistente:** Quedas para **-150 a -200** ocorrem ocasionalmente, mas sem comprometer a tendência geral de bom desempenho.

## 4. Conclusão
O algoritmo de Aprendizado por Reforço Multi-Agente exibiu desempenho robusto no ambiente Speaker-Listener.  
A meta de manter a pontuação média acima de **-60** foi cumprida de forma consistente ao longo de **5 milhões de iterações**.  
O valor de convergência permanece entre **-40 e -60**, o que representa um avanço significativo comparado ao baseline inicial.

## 5. Trabalhos futuros
1. **Reduzir a volatilidade crítica:** Investigar quedas abruptas (até -275), possivelmente causadas por excesso de exploração, instabilidade multi-agente ou problemas no *replay buffer*.  
2. **Suavizar a curva:** Utilizar **médias móveis** para produzir gráficos mais estáveis e informativos.  

