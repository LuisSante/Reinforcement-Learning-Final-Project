# Relatório: Algoritmo de Aprendizado por Reforço Multi-Agente Speaker-Listener  
**Autores:** Luis Sante & Joel Perca & Andres de la Puente  
**Data:** 29/11/2025

## 1. Introdução e Objetivo
O objetivo desta tarefa é implementar um novo algoritmo de Aprendizado por Reforço Multi-Agente (MARL) no ambiente **Speaker-Listener**, buscando superar o desempenho obtido anteriormente com o **MATD3**.  
Nesta etapa, substituímos o método original pelo **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)**, com o intuito de avaliar se uma arquitetura menos complexa, porém colaborativa e centralizada, poderia atingir resultados mais consistentes na navegação do *Listener* até o alvo.

O critério de sucesso permanece o mesmo: **pontuação média superior a -60**, valor estabelecido como referência de desempenho eficiente no ambiente.

## 2. Configurações e Metodologia de Treinamento
O processo de treinamento manteve o mesmo pipeline experimental usado com o MATD3: múltiplas execuções, milhões de interações e ajustes iterativos em hiperparâmetros.  
A mudança central foi a troca do núcleo do algoritmo, passando de uma política com duplo crítico atrasado (TD3) para o esquema cooperativo **DDPG centralizado** próprio do MADDPG.

### MADDPG vs MATD3 — Diferença Essencial

| Aspecto | MATD3 | MADDPG |
|--------|-------|--------|
| **Críticos** | Dois críticos TD3 (reduzem overestimation) | Um único crítico por agente |
| **Atraso de atualização** | Atualização da política é atrasada | Atualização sincronizada padrão |
| **Robustez** | Alta, porém sensível à explosão de ruído | Estável, porém mais exposto a overfitting |
| **Custo computacional** | Mais alto | Mais leve |
| **Velocidade de convergência** | Lenta, porém segura | Rápida, porém exige tuning fino |

### 2.1 Variações das Configurações

| Categoria | Ajuste Realizado | Razão |
|----------|------------------|--------|
| **Arquitetura** | Profundidade e largura das redes | Melhor expressividade para comunicação Speaker–Listener |
| **Exploração (ruído)** | Intensidade adaptativa | Evitar saturação da política determinística |
| **Learning Rates** | Ajustes independentes para ator e crítico | Sincronizar estabilidade e velocidade |
| **Replay Buffer** | Tamanho expandido | Evitar correlação temporal no MADDPG |
| **Gamma / Tau** | Leves variações | Controlar estabilidade de longo prazo |
| **Frequência de Atualização** | Sincronizada por agente | Evitar desbalanceamento entre agentes |

As execuções (0–2M, 0–3M e 0–5M iterações) devem ser entendidas como **variações de um mesmo experimento**, cada uma revelando o impacto da troca para MADDPG e dos ajustes finos aplicados ao longo do processo.

---

## 3. Análise de Resultados

### 3.1 Treinamento Inicial (0 a 2 milhões de iterações)

<p align="center">
  <img src="Projeto%20Final/models/MATD3_backup/training_scores_evolution.png" width="1000">
</p>

- Convergência rápida para a faixa **-50 a -70**.  
- Quedas profundas, mas recuperação limpa.  
- Atinge **-60** por volta da iteração ~20.  
- Mantém-se entre **-35 e -60** na fase estável.

---

### 3.2 Treinamento Ampliado (0 a 3 milhões de iterações)

<p align="center">
  <img src="Projeto%20Final/models/MATD3_backup3/training_scores_evolution.png" width="1000">
</p>

- Estabilidade após ~50 iterações.  
- MATD3 apresentava quedas abaixo de **-250**; aqui são menos intensas.  
- Recuperação mais rápida e previsível.

---

### 3.3 Treinamento Estendido (0 a 5 milhões de iterações)

<p align="center">
  <img src="Projeto%20Final/models/MATD3_backup2/training_scores_evolution.png" width="1000">
</p>

- Mantém faixa de **-30 a -70**.  
- Desempenho superior ao baseline.  
- Oscilações menos dramáticas que o MATD3.

---

## 3.4 Resultados do MADDPG (Novo Algoritmo Implementado)

Após substituir completamente o MATD3 pelo **MADDPG**, realizamos um novo ciclo de treinamento dedicado para avaliar seu desempenho real no ambiente Speaker–Listener.

<p align="center">
  <img src="Projeto%20Final/models/MADDPG/training_scores_evolution_smoothed.png" width="1000">
</p>

### Principais Observações
- **Convergência extremamente rápida:** atinge valores próximos a **-60** logo no início.  
- **Oscilação muito menor** em comparação com o MATD3.  
- **Faixa operacional estável:** desempenho consistente entre **-40 e -55**.  
- **Recuperação eficiente:** quedas menos profundas e trajetórias mais previsíveis.

### Comparação Direta com MATD3

| Critério | MATD3 | MADDPG | Vencedor |
|---------|--------|--------|----------|
| Estabilidade | Oscila muito | Oscila pouco | **MADDPG** |
| Recuperação | Lenta | Rápida | **MADDPG** |
| Convergência | Boa, porém irregular | Muito rápida e suave | **MADDPG** |
| Robustez teórica | Alta | Moderada | MATD3 |
| Robustez experimental | Moderada | Alta | **MADDPG** |
| Facilidade de tuning | Difícil | Simples | **MADDPG** |

Em síntese:  
**O MADDPG supera o MATD3 em estabilidade, velocidade de aprendizado e consistência da política.**

---

## 4. Conclusão (Atualizada)

O novo algoritmo implementado, **MADDPG**, apresentou desempenho superior ao MATD3 no ambiente Speaker-Listener.  
O objetivo da tarefa foi atingido, pois o MADDPG manteve a pontuação média consistentemente acima de **-60**, com menor volatilidade e melhor estabilidade.

Em termos práticos:  
- MATD3 fornece robustez teórica, porém instável no experimento.  
- MADDPG alcança melhor desempenho real, com curva suave e menos colapsos.

---

## 5. Trabalhos Futuros

1. **Reduzir volatilidade residual** em quedas pontuais entre -120 e -150.  
2. **Explorar versões híbridas** incorporando princípios do TD3 ao MADDPG.  

