# Projeto final IND8217E: guia completo explicado

**Dataset:** *Preventive to Predictive Maintenance* (filtros de ar, Hochschule Esslingen)
**Equipe:** Bruno Albino Bennertz (2557056) e Rodrigo Fortunato Carvalho (2557001)

Este documento explica **tudo** o que foi feito, em português, para estudo e para preparar a apresentação. Os gráficos são feitos com **seaborn**.

- **Organização:** segue o *Project Workflow* do professor (slides "6.1 Intro to Course Project").
- **Arquivos-fonte:** o código está em `Filter_RUL_Project.ipynb` e o relatório entregue em `docs/Project_Report.pdf`.
- **Números:** todos saem de `results/numbers.json`, gerado pelo próprio notebook.

---

## Sumário
1. [Problem Definition](#1-problem-definition)
2. [Data Definition](#2-data-definition)
3. [Variables Categorization](#3-variables-categorization)
4. [Data Processing](#4-data-processing)
5. [Choosing Analysis Tools](#5-choosing-analysis-tools)
6. [Setting the Execution Plan](#6-setting-the-execution-plan)
7. [Executing the Plan](#7-executing-the-plan)
8. [Analyzing the Results](#8-analyzing-the-results)
9. [Documentation: comparação, discussão e gráficos](#9-documentation-comparação-discussão-e-gráficos)
10. [Apêndices: apresentação, perguntas da banca, reprodução](#10-apêndices)

---

## 1. Problem Definition

### 1.1 Description
- Uma bancada de testes injeta poeira num filtro de ar. Conforme a poeira se acumula, a **pressão diferencial Δp** através do filtro aumenta. **O filtro "falha" quando Δp passa de 600 Pa.**
- **Situação atual: manutenção preventiva.** Troca-se o filtro após um intervalo fixo, escolhido para que só os ~10 % de filtros de vida mais curta falhem antes da troca.
  - Consequência 1: joga-se fora vida útil da maioria dos filtros. No teste, a política preventiva aproveita só **72,4 %** da vida.
  - Consequência 2: o histórico de treino quase nunca mostra uma falha. Os dados são **censurados à direita**: o filtro foi retirado antes de falhar, então não sabemos quando ele falharia.

### 1.2 Representation
| Pergunta do slide | Resposta |
|---|---|
| Diagnóstico ou prognóstico? | **Prognóstico**: prever o futuro (quanto tempo falta), não identificar uma falha atual |
| Classificação ou regressão? | **Regressão** do RUL (*Remaining Useful Life*, em segundos), com dados de sobrevivência (censurados) |
| Qual é a saída? | RUL de um filtro em operação, a cada instante, a partir do histórico de Δp e das condições de operação |

### 1.3 Objectives
1. Prever o RUL nos dados de **teste** com o menor erro possível. É a tarefa oficial do dataset, e o teste nunca é usado para treinar.
2. Lidar com a **censura** dos dados de treino, o "desafio principal" segundo a documentação.
3. Verificar se o monitoramento de condição melhora a previsão em relação a um modelo de confiabilidade clássico (Weibull).
4. Comparar duas estratégias de ML: **regressão direta** do RUL (Random Forest e XGBoost, treinados com pseudo-rótulos) e **simulação autorregressiva** (M4, sem nenhum rótulo de RUL). Traduzir o erro em decisão de manutenção.

**Hipóteses** (testadas na seção 8):
- **H1:** monitorar Δp melhora a previsão do RUL em relação a um modelo que só conhece as condições de operação.
- **H2:** feed e tipo de poeira dominam a vida do filtro; vida ∝ 1/feed.
- **H3:** o modelo autorregressivo recursivo, treinado **sem nenhum rótulo de RUL**, supera os regressores diretos treinados com pseudo-rótulos (Random Forest e XGBoost).

---

## 2. Data Definition

### 2.1 Source of data
- **Origem:** Hochschule Esslingen (Alemanha), grupo de *Reliability Engineering and PHM*. Autores: Hagmeyer, Mauthe, Dutt e Zeiler (2021).
- **Publicação:** Kaggle, usuário "Prognostics @ HSE".
- **Artigo de referência:** Hagmeyer et al. (2021), *International Journal of PHM*, 12(2), DOI 10.36001/ijphm.2021.v12i2.3087.
- **Arquivos usados:** `data/Train_Data_CSV.csv` (39 420 linhas) e `data/Test_Data_CSV.csv` (39 414 linhas). Cada um tem 50 testes de vida (runs) amostrados a **10 Hz**.
- **Arquivos não usados:** os `.mat` (tabelas MATLAB, incluindo a granulometria e a versão "uncensored"), por decisão da equipe.

### 2.2 Physical meaning: como o experimento funciona
- **Filtro:** manta de fibras não tecidas (CC 600 G), área de 6131 mm², 20 mm de espessura; Δp de ~25 Pa quando limpo.
- **Poeira:** Arizona Test Dust (ISO 12103-1) em três granulometrias, com densidades aparentes:
  - **A2 fina:** 0,900 g/cm³;
  - **A3 média:** 1,025 g/cm³;
  - **A4 grossa:** 1,200 g/cm³.
- **Alimentação:** um fuso empurra a poeira para um bico Venturi. A taxa (`Dust_feed`, mm³/s) é **constante dentro de cada run**.
- **Controle:** todos os testes rodam em **controle de vazão** (a vazão é mantida num set-point).
- **Sensores:** pressão diferencial (faixa de 0–2500 Pa) e medidor de vazão (com valores já corrigidos).
- **Como Δp evolui:**
  1. **Filtração em profundidade:** as partículas ficam presas dentro das fibras e Δp sobe devagar.
  2. **Filtração de torta (cake):** forma-se uma camada de pó na superfície e Δp acelera, quase exponencialmente.
- Por isso a curva é **convexa**, e a forma exponencial funciona bem para extrapolar (seção 7.1).

### 2.3 Description: tipos e faixas de valores
| Variável | Tipo | Unidade | Faixa no treino | Faixa no teste | Significado físico |
|---|---|---|---|---|---|
| `Data_No` | inteiro (ID) | – | 1–50 | 1–50 | Número do teste de vida (run) |
| `Time` | contínua | s | 0,1–179,4 | 0,1–258,1 | Idade do filtro dentro do run (10 Hz) |
| `Differential_pressure` | contínua | Pa | 0–607,9 | 0–575,0 | Perda de carga no filtro = **indicador de saúde**; falha em 600 Pa |
| `Flow_rate` | contínua | não informada (≈ L/min) | 0–84,2 | 0–84,2 | Vazão de ar; **dois set-points**, ~58 e ~81 |
| `Dust_feed` | contínua, 8 níveis | mm³/s | 59,1–317,0 | 59,1–317,0 | Volume de poeira injetado por segundo |
| `Dust` | categórica, 3 níveis | – | A2/A3/A4 | A2/A3/A4 | Granulometria da poeira |
| `RUL` | contínua | s | não existe | 1,9–335,1 | Segundos até a falha (**ground truth** do teste) |

**Desenho experimental:** fatorial parcial com 3 poeiras × 8 níveis de feed × 2 vazões.
- Treino: 11 runs de A2, 26 de A3 e 13 de A4; 16 runs com vazão alta.
- Teste: 14 runs de A2, 24 de A3 e 12 de A4; **26 runs com vazão alta**, e feed médio de 179 mm³/s contra 161 no treino. É uma leve mudança de distribuição.

### 2.4 O que a EDA mostrou (visualização)
1. **Distribuições:**
   - Δp é muito assimétrico, porque a maior parte das amostras está no trecho plano inicial.
   - A vazão é bimodal (dois set-points).
   - O feed tem 8 valores discretos.
   - O RUL-alvo no último ponto do teste vai de 1,9 a 136,6 s (mediana 26,5 s).
2. **Curvas de degradação (`fig03`):**
   - Contra o *tempo*, as curvas ficam espalhadas.
   - Contra o **volume de poeira acumulado V = feed × t**, elas colapsam numa faixa estreita: quem manda é a carga, não o relógio.
   - **A troca preventiva acontece sempre em V ≈ 10 604 mm³**, ou seja, o intervalo é ∝ 1/feed.
3. **Efeito das condições** (`fig04`, mesmo volume V ≈ 10 600 mm³):
   - **Vazão:** para A3, a mediana de Δp é 296 Pa na vazão alta contra 180 Pa na baixa (lei de Darcy).
   - **Granulometria:** na vazão baixa, Δp cai de fina para grossa: 294 (A2), 180 (A3) e 138 Pa (A4). Partícula fina forma torta mais densa.
4. **Correlação de Spearman com Δp:** V = 0,82, tempo = 0,71, vazão = 0,31, densidade da poeira = −0,32.
5. **Sobrevivência (Kaplan-Meier, `fig05`):** com 45/50 censurados, S(t) só cai até ~0,9. O treino descreve só **os 10 % mais curtos**.
6. **Os 5 runs que falharam no treino são todos de poeira fina A2.** Não existe nenhuma falha observada de A3/A4 no treino.

---

## 3. Variables Categorization

O slide pede dois eixos: **dependência** (resposta vs estado) e **controlabilidade** (settings vs medições de sensor).

| Variável | Dependência | Controlabilidade | Justificativa |
|---|---|---|---|
| `Differential_pressure` | **Dependente (resposta)** | **Não controlável** (medição de sensor) | É a consequência do entupimento; ninguém a define |
| `RUL` | **Dependente (alvo)** | – | É derivada da trajetória futura de Δp |
| `Dust_feed` | Independente (estado) | **Controlável** (setting) | Definida na interface antes do teste |
| `Dust` | Independente (estado) | **Controlável** (setting) | A poeira escolhida para o teste |
| `Flow_rate` | Independente (estado) | **Controlável** (setting medido) | O set-point é controlado, mas o valor lido tem ruído e dropouts |
| `Time` | Independente (estado) | Não controlável | É o relógio; avança sozinho |
| `Data_No` | – | – | Só identifica o run (usado para agrupar e para evitar vazamento) |

**Por que isso importa:**
- Δp é o único **sinal de condição**: é o que muda com o desgaste.
- Feed, poeira e vazão definem a *velocidade* do desgaste.
- O tempo sozinho engana: com feed 5× maior, o filtro entope 5× mais rápido.

---

## 4. Data Processing

### 4.1 Missing data
**Primeira constatação: não existe nenhum `NaN`, nenhuma linha duplicada e nenhum buraco na grade de 10 Hz.** Mesmo assim faltam informações, de forma *implícita*:

| # | Problema | Tamanho | O que é | Tratamento |
|---|---|---|---|---|
| 1 | **Censura** (tempo de falha desconhecido) | 45 dos 50 runs de treino | O filtro foi trocado antes de falhar. **É o faltante mais importante: o alvo (RUL) não existe para 90 % dos runs de treino.** | Pseudo-rótulos por extrapolação física, validados (seção 7.1) |
| 2 | Δp = 0 | 5 672 linhas no treino (14 %) e 2 722 no teste (7 %) | Concentrado nos primeiros ~20 s (mediana t ≈ 10 s). Filtro limpo dá ~25 Pa, abaixo da resolução do sensor (zona morta). Não é "zero de verdade". | Mantido (é informativo: "filtro limpo") e suavizado |
| 3 | Vazão = 0 | 45 linhas no treino (8 runs) e 13 no teste | Dropouts do sensor: a vazão é controlada e não pode ir a 0 com Δp subindo | Marcado como faltante → interpolação linear dentro do run |
| 4 | Timestamp corrompido | 1 linha (run 28 do treino) | A primeira linha tem `Time = 157,2`, seguida de 0,2 s | Linha removida |
| 5 | Primeira amostra ausente | 3 runs de treino e 9 de teste começam em 0,2 s | Perdeu-se uma amostra no início | Nada (0,1 s é desprezível) |
| 6 | Teste truncado | 50 runs de teste | O futuro de cada run não é observado | É exatamente o que precisamos prever |

### 4.2 Remove outliers
**Cuidado importante:** valores altos de Δp (perto de 600 Pa) **não são outliers, são o sinal de falha**. A regra do IQR aplicada no conjunto todo marcaria 3 292 amostras de treino como "outliers", justamente as mais importantes. Por isso procuramos anomalias **locais, dentro de cada run**:

| Verificação | Treino | Teste | Decisão |
|---|---|---|---|
| Δp fora da faixa do sensor (0–2500 Pa) | 0 | 0 | – |
| Saltos de Δp > 50 Pa entre amostras consecutivas | 0 | 0 | – |
| **Picos isolados de Δp** (filtro de Hampel: desvio > 5 MAD **e** > 20 Pa da mediana da janela de 1 s) | **17** | **28** | **Substituídos pela mediana local** (não apagados, para manter a grade de 10 Hz) |
| Vazão > 8 unidades longe do set-point (fora os zeros) | 104 | 84 | Mantidos: transientes de partida (mediana t = 0,2 s). A vazão entra no modelo só pela média e pelo indicador de set-point |

**Detalhes do filtro de Hampel:**
- Usamos uma janela **centrada** (±0,5 s).
- Testamos antes uma janela "só para trás", que marcava 218 amostras: 211 delas *acima* da mediana e quase todas com Δp > 300 Pa. Ou seja, ela confundia a subida rápida e legítima perto da falha com picos. A janela centrada pega só os picos reais.
- Em operação real isso só adiciona 0,5 s de atraso.
- O valor bruto fica guardado em `dp_raw`, e **o instante de falha (primeiro Δp ≥ 600 Pa) é sempre medido no sinal bruto**.

### 4.3 Generate new features
**Limpeza** (aplicada igualmente a cada run de treino e de teste, sem aprender nada do teste):
1. remover o timestamp corrompido;
2. interpolar a vazão;
3. filtro de Hampel;
4. Δp suavizado com **média móvel causal de 1 s** (só usa o passado, então é calculável em tempo real);
5. reamostragem para 1 Hz na modelagem.

**Features criadas** (todas causais):

| Feature | Significado | Por que ajuda |
|---|---|---|
| `dp_s`, `dp_frac` | Δp suavizado e Δp/600 | Quão perto da falha o filtro está |
| `slope5`, `slope10` | Inclinação de Δp nos últimos 5 e 10 s (Pa/s) | Quão rápido o filtro entope agora |
| `curv` | Mudança de inclinação | Aceleração do entupimento (entrada na fase de torta) |
| `gr10` | Taxa de crescimento de log Δp em 10 s | Taxa exponencial local |
| `V`, `mass` | Volume (mm³) e massa (g) de poeira acumulada | A "carga" do filtro; explica melhor que o tempo |
| `Time` | Idade | Referência temporal |
| `log_feed`, `rho`, `A3`, `A4` | Feed (log) e tipo de poeira (densidade + one-hot; A2 é a referência) | Definem a velocidade do desgaste |
| `flow_s`, `flow_high` | Vazão média até agora e indicador do set-point alto | Vazão alta → mais perda de carga |

---

## 5. Choosing Analysis Tools

### 5.1 What algorithms? Why?
| Etapa / modelo | Algoritmo | Por que |
|---|---|---|
| Rótulos para os runs censurados | Extrapolação exponencial da curva de Δp ("physics-informed") | Sem rótulo não há regressão; a forma exponencial vem da física da filtração e foi validada |
| **M1 (referência)** | Weibull AFT (*accelerated failure time*), lifelines | É o modelo de confiabilidade do curso, que usa só as condições de operação. Serve de base para medir o ganho do sensor (H1) |
| **M2 (ML)** | Random Forest | Média de muitas árvores profundas (bagging): reduz a variância, captura não linearidades, pouco tuning |
| **M3 (ML)** | **XGBoost** (eXtreme Gradient Boosting) | Árvores rasas adicionadas em sequência, cada uma corrigindo o erro (gradiente) deixado pelas anteriores (boosting): reduz o viés; tem regularização L2 e subamostragem de linhas e colunas; costuma ser o melhor em dados tabulares |
| **M4 (ML)** | **Autorregressivo recursivo** (XGBoost prevendo o crescimento de Δp + simulação) | Aprende só a **dinâmica** do Δp (como ele cresce nos próximos 5 s a partir dos seus lags). Usa todas as transições dos 50 runs, inclusive os censurados, e **não precisa de rótulo de RUL**. O RUL sai simulando o futuro até 600 Pa (H3) |

### 5.2 Which libraries?
- `pandas` e `numpy` (dados);
- `scipy` (ajuste de curvas);
- `lifelines` (Weibull AFT, Kaplan-Meier);
- `scikit-learn` (Random Forest, GroupKFold, métricas);
- **`xgboost`** (XGBoost);
- `matplotlib` + **`seaborn`** (gráficos).

As versões estão em `requirements.txt`.

### 5.3 Evaluation criteria
- **Principal:** o RUL previsto no **último ponto observado** de cada um dos 50 runs de teste, **comparado com o ground truth** (coluna `RUL` do teste). Métricas:
  - **RMSE e MAE**, o erro típico em segundos;
  - **Viés**, a média de (previsto − real): negativo = conservador (troca cedo); positivo = otimista (risco de falha);
  - **% dentro de ±10 s**;
  - **Score PHM08**, assimétrico: penaliza mais o atraso (`e^(d/10) − 1`) que o adiantamento (`e^(−d/13) − 1`), porque atraso = parada não planejada.
- **Teste estatístico:** *bootstrap pareado* sobre os 50 runs (5 000 reamostragens). Uma diferença só é aceita se o IC de 95 % não contém 0.
- **Secundária:**
  - erro em todas as amostras de 1 Hz por horizonte;
  - alarme de falha iminente (RUL ≤ 15 s: precisão e recall);
  - simulação de política de troca.
- **Regra de ouro contra vazamento:**
  - tudo que é "aprendido" usa **só o treino**;
  - as divisões de validação são sempre **por run**;
  - o teste é usado **uma única vez**, no final.

---

## 6. Setting the Execution Plan

| Etapa | Conteúdo | Responsável | Prazo interno |
|---|---|---|---|
| 1 | Ler a documentação, entender a bancada e a tarefa | ambos | *[definir]* |
| 2 | EDA: variáveis, faltantes, outliers, visualizações | *[definir]* | *[definir]* |
| 3 | Limpeza e features | *[definir]* | *[definir]* |
| 4 | Pseudo-rótulos e validação | *[definir]* | *[definir]* |
| 5 | Modelos (M1–M4) com validação cruzada por run | *[definir]* | *[definir]* |
| 6 | Avaliação no teste, bootstrap, hipóteses | *[definir]* | *[definir]* |
| 7 | Relatório (≤ 10 páginas) e apresentação (10 min) | ambos | entrega em **10/dez, 15h55**; apresentação em **10/dez** |

> Preencham a divisão de tarefas e os prazos reais (o slide pede "quem faz o quê" e "prazos internos").

**Backup plan** (definido antes de começar):
- se os pseudo-rótulos não fossem confiáveis → abordagem só de sobrevivência (Weibull/Kaplan-Meier com censura);
- se a simulação autorregressiva fosse instável → ficar só com a regressão direta (M2/M3).

---

## 7. Executing the Plan

### 7.1 Pseudo-rótulos: o ponto central
**Por que é necessário:**
- Para treinar um regressor de RUL precisamos do RUL de cada amostra de treino, que é `T_falha − t`.
- Só conhecemos `T_falha` em 5 runs.
- As alternativas ingênuas são ruins:
  - **treinar só com os 5 runs que falharam:** só vidas curtas, só poeira fina → subestima tudo;
  - **fingir que a troca foi a falha:** subestima sistematicamente.

**O que fizemos: extrapolação física.**
1. Para cada run censurado, ajustamos `Δp = a·e^(bt) + c` ao Δp observado.
2. Calculamos quando a curva cruzaria 600 Pa: `T̂_falha = ln[(600 − c)/a] / b`, com a restrição `T̂_falha ≥ T_troca` (sabemos que o filtro sobreviveu até a troca).

**Validação honesta:**
- Pegamos os 5 runs que falharam, cortamos em 30/50/70/90 % da vida, ajustamos só no pedaço cortado e comparamos com a falha real.
- Testamos 4 formatos de curva:

| Vida observada | Exponencial: erro RMS | Exponencial: viés | Quadrática | Potência | Reta final |
|---|---|---|---|---|---|
| 30 % | 124 % | +64 % | 313 % | 461 % | 285 % |
| 50 % | **15 %** | −5 % | 41 % | 49 % | 147 % |
| 70 % | **9 %** | −6 % | 18 % | 11 % | 36 % |
| 90 % | 4 % | −4 % | 4 % | **2 %** | 3 % |

**Leitura:**
- A exponencial tem o menor erro global e é a melhor até 70 % observado.
- A partir de 50 %, ela **subestima** um pouco a vida, então os rótulos são **conservadores** (bom para manutenção).
- Os runs censurados foram observados por uma **mediana de 79 % da vida extrapolada (mínimo 64 %)**, dentro da faixa confiável.
- Volume de falha extrapolado (mediana): A2 12 455, A3 13 318 e A4 14 144 mm³. É fisicamente coerente: a poeira fina entope antes.

**Resultado:** 3 907 amostras de treino (1 Hz), 395 delas de runs com falha real.

> **Duas maneiras de resolver "como tenho RUL para treinar?"**
> - **Regressão direta (M2, M3):** usa os pseudo-rótulos acima e aprende `RUL = f(features)`.
> - **Autorregressiva (M4, seção 7.4):** **não precisa de RUL**. Aprende só a dinâmica `Δp(t) → Δp(t+5 s)`, e cada transição observada é um dado válido, inclusive nos runs censurados. O RUL sai por simulação.

### 7.2 M1: Weibull AFT
- **Modelo:** `log λ = β0 + β1·log(feed) + β2·A3 + β3·A4 + β4·vazão_alta`, com forma ρ. A previsão para um filtro de idade t é a **vida residual média**, `E[T − t | T > t]`.
- **Ajuste "de livro" só com dados censurados** (5 falhas + 45 censuras):
  - β(log feed) = −1,11 (IC de −1,48 a −0,74);
  - A3 e A4 **não identificáveis** (IC de A3 de −0,9 a +3,3), porque não há falha dessas poeiras.
- **M1 usado na comparação:** ajustado nas 50 vidas (reais + pseudo-rótulos), as mesmas informações que os modelos de ML recebem. A única diferença passa a ser **usar ou não o sensor**.
  - β(log feed) = **−1,00** (IC de −1,08 a −0,93);
  - A3/A4 vivem ~20 % mais que A2 (p < 0,001);
  - a vazão não é significativa (p = 0,22);
  - forma ρ ≈ 9,2.

### 7.3 M2 (Random Forest) e M3 (XGBoost)
**Seleção** por validação cruzada GroupKFold de 5 folds, **agrupada por run** (amostras do mesmo filtro nunca ficam no treino e na validação ao mesmo tempo):

| Modelo | Configuração | RMSE de CV (s) |
|---|---|---|
| M2 | **RF, 300 árvores, min_samples_leaf = 20, max_features = 1,0** | **13,2 ± 5,1** |
| M2 | RF leaf = 5, features = 1,0 | 13,3 ± 5,8 |
| M2 | RF leaf = 5, features = 0,5 | 13,8 ± 3,1 |
| M2 | RF leaf = 20, features = 0,5 | 14,8 ± 2,0 |
| M3 | **XGBoost, profundidade 3, lr 0,05, 400 árvores** (subsample 0,8, colsample 0,8, min_child_weight 5) | **10,3 ± 2,8** |
| M3 | XGBoost, profundidade 3, lr 0,1 | 10,8 ± 3,0 |
| M3 | XGBoost, profundidade 5, lr 0,05 | 11,4 ± 3,9 |
| M3 | XGBoost, profundidade 6, lr 0,1 | 12,4 ± 3,1 |

**Importância das features no M3** (permutação):
- `log_feed` (+44,8 s de RMSE se embaralhada);
- `dp_s` (+8,9 s);
- `slope10` (+7,4 s).

O modelo combina "quão rápido esse filtro *deveria* entupir" (feed) com "quão perto e quão rápido ele *está* entupindo" (Δp atual e inclinação).

### 7.4 M4: modelo autorregressivo recursivo (sem rótulos de RUL)
**Ideia:** em vez de aprender o RUL, aprender **como o Δp cresce** e simular o futuro.

**1. Modelo de um passo.**
- Um XGBoost prevê o crescimento de `log(1+Δp)` nos próximos 5 s: `y = log(1+Δp(t+5)) − log(1+Δp(t))`.
- Entradas, os **lags autorregressivos**:
  - `log(1+Δp)` agora;
  - o incremento dos últimos 5 s;
  - o incremento de 5 a 10 s atrás;
  - mais a carga de poeira V e as condições (feed, poeira, vazão).
- Treinado com **3 657 transições** de **todos os 50 runs** de treino. Os censurados também valem: cada par "agora → daqui a 5 s" observado é um exemplo correto.
- A escolha de hiperparâmetros usa o erro de um passo na validação cruzada por run (RMSE de 0,25 em unidades de log, contra 0,46 de "crescimento zero"). **Nenhum rótulo de RUL é usado em lugar nenhum.**

**2. Simulação recursiva (rollout).**
- A partir do estado atual do filtro, aplica o modelo, usa a própria previsão como novo lag e repete, em passos de 5 s, até `log(1+Δp)` chegar a `log(601)`.
- RUL = tempo simulado até o cruzamento (interpolado dentro do último passo).

**3. Restrições físicas.**
- O incremento é forçado a ser ≥ 0, porque o filtro não desentope.
- Teto de 500 s se nunca chegar a 600 Pa. Nenhuma simulação do teste chegou ao teto.

**Por que prever o crescimento e não o nível?**
- Árvores não extrapolam. Quase não há treino com Δp > 400 Pa, então um modelo de *nível* nunca chegaria a 600.
- A *taxa* de crescimento fica dentro da faixa vista no treino, então a simulação consegue chegar lá.

**Ilustração** (`fig09_ar_rollout`): o modelo foi retreinado **sem** o run 43 (que falhou) e simulado a partir de t = 30, 50 e 70 s. As curvas simuladas acompanham o Δp real e cruzam 600 Pa perto da falha verdadeira.

---

## 8. Analyzing the Results

### 8.1 Comparação com o ground truth do teste
**Toda métrica é calculada contra a coluna `RUL` do arquivo de teste** (ground truth: o RUL real, medido no run que foi até a falha).

- **Tabela run a run:** `results/test_ground_truth_comparison.csv` traz, para cada um dos 50 runs de teste:
  - as condições;
  - o instante de corte;
  - o **RUL verdadeiro**;
  - a previsão e o erro de cada modelo.
- **Figura:** `fig15_ground_truth_by_run.png` mostra o ground truth (traço preto) e as quatro previsões em cada run.
- **Modelo mais próximo do ground truth em cada run:** M4 autorregressivo em 20 runs, M2 RF em 14, M1 Weibull em 10 e M3 XGBoost em 6.
  - Atenção: "ganhar mais runs" não é o mesmo que ter o menor erro médio. O Weibull acerta bem os runs "típicos", mas erra muito nos atípicos, e por isso tem o maior RMSE.

### 8.2 Métricas no último ponto observado (n = 50)

| Modelo | RMSE (s) | MAE (s) | Viés (s) | Dentro de ±10 s | Score PHM | RMSE em todas as amostras (s) |
|---|---|---|---|---|---|---|
| M1 Weibull AFT (referência) | 24,3 | 17,1 | −11,5 | 46 % | 791 | 46,8 |
| M2 Random Forest | 19,5 | 13,8 | **−5,9** | 52 % | 683 | 40,3 |
| M3 XGBoost | 18,8 | 13,3 | −10,7 | 54 % | 321 | 38,3 |
| **M4 Autorregressivo** | **15,2** | **10,9** | −9,3 | **62 %** | **161** | **26,6** |

**Bootstrap pareado** (diferença de RMSE, IC de 95 %; positivo = o segundo modelo é melhor):

| Comparação | Diferença | IC de 95 % | Significativo? |
|---|---|---|---|
| M1 − M2 | +4,8 | [−1,8; +10,8] | não |
| **M1 − M3** | **+5,5** | **[+1,0; +9,9]** | **sim** |
| **M1 − M4** | **+9,1** | **[+3,7; +14,8]** | **sim** |
| M2 − M3 | +0,7 | [−2,3; +4,9] | não |
| **M2 − M4** | **+4,3** | **[+0,6; +8,8]** | **sim** |
| **M3 − M4** | **+3,6** | **[+1,4; +5,7]** | **sim** |

### 8.3 Hipóteses: aceitas ou rejeitadas

| | Hipótese | Resultado | Evidência |
|---|---|---|---|
| **H1** | Monitorar Δp melhora a previsão em relação ao modelo só com as condições | **Aceita** | M4 (+9,1 s, IC [+3,7; +14,8]) e M3 (+5,5 s, IC [+1,0; +9,9]) são significativamente melhores que M1; o M2 é melhor em média, mas não significativo |
| **H2** | Feed e poeira dominam a vida; vida ∝ 1/feed | **Aceita** | β(log feed) = −1,00 [−1,08; −0,93]; A3/A4 significativos (p < 0,001); vazão não (p = 0,22) |
| **H3** | O autorregressivo (sem rótulos de RUL) supera os regressores diretos | **Aceita** | M2 − M4 = +4,3 s [+0,6; +8,8]; M3 − M4 = +3,6 s [+1,4; +5,7]: o M4 é significativamente melhor que os dois |

> A antiga H4 (política preditiva vs preventiva) foi removida das hipóteses. A simulação continua como análise de apoio (8.5).

### 8.4 Alarme de falha iminente (RUL ≤ 15 s, todas as amostras)

| Modelo | Precisão | Recall | F1 |
|---|---|---|---|
| M1 | 0,24 | 0,70 | 0,36 |
| M2 RF | **0,40** | 0,45 | 0,43 |
| M3 XGBoost | 0,31 | 0,66 | 0,42 |
| M4 Autorregressivo | 0,36 | **0,82** | **0,50** |

### 8.5 Valor para a manutenção (análise de apoio)
**Simulação nos 50 filtros de teste** (a falha real vem do ground truth: `T_falha = t_último + RUL_último`):
- **Preventiva:** trocar quando V = 10 600 mm³.
- **Preditiva:** no último ponto, agendar a troca para `t + RUL previsto`.

| Política | Falhas não planejadas (de 50) | Vida aproveitada |
|---|---|---|
| Preventiva atual | 5 | 72,4 % |
| Preditiva – M1 | 11 | 89,5 % |
| Preditiva – M2 RF | 19 | 92,5 % |
| Preditiva – M3 XGBoost | 10 | 90,1 % |
| Preditiva – M4 Autorregressivo | 10 | 91,5 % |

---

## 9. Documentation: comparação, discussão e gráficos

### 9.1 Discussão
- **O sensor ajuda (H1).** Os três modelos de ML erram menos que o Weibull, e em todas as amostras a diferença é clara (27–40 s contra 47 s). O M4 e o XGBoost são significativamente melhores que o Weibull.
- **Autorregressivo > regressão direta (H3 aceita).** O M4, que **nunca viu um rótulo de RUL**, é o mais preciso (RMSE de 15,2 s contra 18,8 e 19,5) e significativamente melhor que os dois regressores diretos.
  - **Por que ele ganha:** usa todas as transições dos runs censurados e **não depende dos pseudo-rótulos**, que subestimam a vida longa dos filtros A3/A4. Por isso seu erro cresce bem mais devagar com o horizonte (37 s contra 57–60 s na faixa de RUL de 130–400 s) e ele acerta os runs longos (ex.: run 30 de teste).
  - **Ponto fraco:** nos primeiros ~15 s de um run, com Δp ainda na zona morta do sensor, não há crescimento para extrapolar, e algumas simulações saem muito otimistas (até +200 s). Depois de 15 s, nenhum erro passa de 30 s. Na prática, só usar a previsão quando Δp sair da zona morta.
- **Random Forest × XGBoost.** O XGBoost é melhor em **todos** os critérios:
  - RMSE no último ponto: 18,8 contra 19,5 s;
  - validação cruzada: 10,3 contra 13,2 s;
  - todas as amostras: 38,3 contra 40,3 s;
  - score PHM: 321 contra 683.

  Mesmo assim, a diferença no teste não é significativa (IC [−2,3; +4,9]). O RF é o menos enviesado (−5,9 s): o boosting se ajusta mais fielmente aos pseudo-rótulos conservadores e herda mais viés negativo (−10,7 s), enquanto a média de árvores profundas do RF fica mais perto de zero, à custa de mais erros otimistas.
- **Cada modelo tem um perfil diferente:**
  - **RF:** o menos enviesado (−5,9 s), mas por isso comete mais erros otimistas (pior para agendar troca).
  - **XGBoost:** menor RMSE e MAE, melhor score PHM, mais conservador que o RF.
  - **Autorregressivo:** menor erro, melhor score PHM (161), melhor F1 de alarme (0,50) e o mais próximo do ground truth em 20 dos 50 runs.
- **Viés negativo em todos.** Vem dos rótulos conservadores: a exponencial subestima, e as falhas de treino são todas de poeira fina, a de vida mais curta. É o lado seguro.
- **O erro cresce com o horizonte.** Longe da falha a curva ainda é plana e ninguém "enxerga" o fim. É coerente com o erro de 124 % da extrapolação quando só 30 % da vida foi observada.
- **Manutenção.** Usar o RUL previsto aproveita muito mais vida (90–93 %), mas, sem margem de segurança, todos os modelos causam mais falhas que a preventiva (10–19 contra 5). Os modelos estão prontos como **apoio à decisão** (ranking de filtros, alarme); para troca automática é preciso calibrar uma margem conforme o risco aceito pela empresa.

### 9.2 Comparação com os notebooks de referência
1. **"Predictive maintenance using ML and LSTM RNN"** (mesmo dataset):
   - **O que faz:** prevê **Δp 1 s à frente** a partir de 20 amostras, com LR, SVR, RF, KNN e LSTM. Reporta R² ≈ 0,996 e MAE ≈ 4,5 Pa.
   - **Reproduzimos:** a regressão linear dá MAE 4,41 Pa e R² 0,9968, e a regra **ingênua "Δp daqui a 1 s = Δp agora"** dá **MAE 4,41 Pa e R² 0,9960**, praticamente igual. O R² alto não mede capacidade de prever falha.
   - Esse notebook também não prevê o RUL, não trata a censura e valida com os últimos 10 % das linhas (vazamento entre runs).
   - **O que aproveitamos:** a ideia de usar lags (janelas) e de prever o próximo valor. Nós a levamos até o fim: prevemos o crescimento e **iteramos até 600 Pa** (M4), o que transforma a previsão de um passo em RUL. Também a ideia de alarme com matriz de confusão.
2. **"ML-powered maintenance: smarter & proactive":**
   - É sobre **outro dataset** (AI4I 2020: classificação de falha).
   - Serve só como referência de método: features, correlação, SMOTE, métricas de classificação.

### 9.3 Gráficos (em `figures/`)
| Arquivo | O que mostra | No relatório |
|---|---|---|
| `fig01_missing_data` | Zona morta de Δp, dropouts de vazão, censura do treino | Fig. 1 |
| `fig02_distributions` | Distribuições de Δp, vazão, feed e RUL-alvo | Fig. 2 |
| `fig03_degradation_curves` | Δp × tempo e Δp × volume (treino e teste) | Fig. 3 |
| `fig04_settings_effect` | Δp na troca por poeira/vazão; correlação | Fig. 4 |
| `fig05_kaplan_meier` | Sobrevivência do treino (só cai até ~0,9) | – |
| `fig06_features` | Suavização e features de inclinação | – |
| `fig07_pseudo_labels` | Validação da extrapolação e exemplos | Fig. 5 |
| `fig08_weibull_aft` | Curvas de sobrevivência e coeficientes do AFT | – |
| `fig09_ar_rollout` | Simulação autorregressiva num run fora do treino | Fig. 6 |
| `fig15_ground_truth_by_run` | **Ground truth × previsões, run a run** | Fig. 7 |
| `fig10_pred_vs_true` | Previsto × real no último ponto | Fig. 8 |
| `fig11_error_horizon` | Erro por horizonte; RMSE/MAE/viés | Fig. 9 |
| `fig12_rul_trajectories` | RUL previsto ao longo de 4 runs | Fig. 10 |
| `fig13_importance` | Importância das features (M3 XGBoost) | – |
| `fig14_policy` | Falhas e vida aproveitada por política | – |

### 9.4 Limitações e trabalhos futuros
**Limitações:**
- 45 dos 50 rótulos de treino são modelados.
- Nenhuma falha de A3/A4 no treino.
- Só 50 runs de teste (ICs largos).
- A política é simulada com uma única decisão no ponto de corte.

**Futuro:**
1. Calibrar os pseudo-rótulos, ou usar uma perda de sobrevivência (deep survival).
2. **Modelo híbrido:** usar a extrapolação física como feature das árvores.
3. Coletar falhas de A3/A4 e usar intervalos de previsão (regressão quantílica) para definir a margem de segurança.

---

## 10. Apêndices

### 10.1 Roteiro para a apresentação de 10 minutos
> **Formato:** "conselho de diretores pedindo verba". **Os dois falam** (nota individual). Sugestão: ~5 min cada.

| Slide | Tempo | Quem | Conteúdo | Figura |
|---|---|---|---|---|
| 1. O problema | 1:00 | A | Troca a intervalo fixo: 28 % da vida jogada fora e, mesmo assim, falhas. Objetivo: prever o RUL | Fig. 1 do PDF do dataset |
| 2. Os dados | 1:15 | A | 50 + 50 testes, 10 Hz; categorização das variáveis; **só 5 falhas no treino, todas de poeira fina** | `fig01` (c) |
| 3. EDA e limpeza | 1:15 | A | Faltantes implícitos, outliers, a carga de poeira explica o entupimento | `fig03` |
| 4. Como resolvemos a censura | 1:30 | A | Pseudo-rótulos pela extrapolação exponencial, validados | `fig07` |
| 5. Modelos | 1:00 | B | Weibull (referência) → regressão direta (RF, XGBoost) com pseudo-rótulos → **autorregressivo sem rótulos** | `fig09_ar_rollout` |
| 6. Resultados vs ground truth | 1:30 | B | Autorregressivo: RMSE de 15,2 s contra 24,3 s do Weibull e 18,8 s do XGBoost; H1, H2 e H3 aceitas | `fig15` + tabela |
| 7. Valor para a empresa | 1:30 | B | 90–93 % da vida aproveitada contra 72 %; usar como apoio à decisão e alarme (recall de 0,82); próximo passo: margem de segurança | `fig14` |
| 8. Conclusão | 1:00 | B | Hipóteses; pedido de verba para o piloto de monitoramento + alarme | – |

### 10.2 Perguntas prováveis da banca
- **"Por que não usaram só os 5 runs com falha?"**
  São todos de poeira fina e de vida curta. O modelo aprenderia que todo filtro morre cedo. A censura é informação: diz que o filtro **sobreviveu** até a troca.
- **"Os pseudo-rótulos não contaminam a avaliação?"**
  Não. A avaliação é contra o **ground truth do teste**, que é real e medido. Os pseudo-rótulos só afetam o treino; o viés negativo no teste mostra o efeito deles (conservador).
- **"Como vocês têm RUL para treinar se os runs não vão até a falha?"**
  De duas formas. Nos modelos diretos (RF, XGBoost), estimamos o RUL dos 45 runs censurados extrapolando a curva de Δp até 600 Pa (validado nos 5 que falharam). No modelo autorregressivo, **não precisamos de RUL**: ele aprende só como o Δp cresce de um instante para o próximo, e o RUL sai simulando até 600 Pa.
- **"Por que o autorregressivo foi o melhor?"**
  Porque não herda o viés dos pseudo-rótulos (que subestimam vidas longas) e aproveita todas as transições dos runs censurados. A desvantagem é o início da vida (zona morta do sensor), quando não há crescimento para extrapolar.
- **"Prever o próximo valor não é trivial?"**
  Prever *só* o próximo valor é (a regra "repetir o valor atual" tem R² de 0,996). O que dá valor é **iterar** a previsão até a falha. E prevemos o *crescimento*, não o nível, para conseguir extrapolar.
- **"Por que tirar o modelo físico da comparação?"**
  Porque ele não aprende com os dados (não é ML). Ele continua no projeto como gerador dos rótulos de treino.
- **"Como trataram outliers?"**
  Valores altos de Δp são o sinal, não outliers. Removemos só picos isolados (filtro de Hampel, 17 no treino e 28 no teste), trocando pela mediana local.
- **"E o notebook do Kaggle com R² = 0,996?"**
  Ele prevê Δp 1 s à frente, não o RUL. A regra "repetir o valor atual" dá o mesmo R² (0,996).
- **"Por que XGBoost e Random Forest?"**
  São os dois grandes tipos de ensemble de árvores. O Random Forest faz *bagging* (média de árvores independentes, reduz a variância); o XGBoost faz *boosting* (árvores em sequência corrigindo o erro, reduz o viés). Aqui o XGBoost venceu em todos os critérios, mas sem diferença significativa no teste.
- **"Qual modelo vocês recomendam?"**
  O **autorregressivo (M4)**: menor erro, melhor score PHM e significativamente melhor que todos os outros, usado a partir do momento em que Δp sai da zona morta do sensor. Para troca automática, deve ser combinado com uma margem de segurança calibrada.

### 10.3 Como reproduzir
```bash
pip install -r requirements.txt
cd Projeto
jupyter nbconvert --to notebook --execute --inplace Filter_RUL_Project.ipynb   # alguns minutos em CPU
```

- Seeds fixas (42).
- O notebook regrava `figures/` e `results/`.
- O relatório foi gerado por script a partir de `results/numbers.json` e `figures/`.

### 10.4 Arquivos da pasta
| Arquivo | O que é |
|---|---|
| `Project Description-2026V2.pdf`, `6.1 Intro to Course Project.pdf` | Instruções do professor |
| `Preventive to Predictive Maintenance dataset.pdf` | Documentação do dataset |
| `data/Train_Data_CSV.csv`, `data/Test_Data_CSV.csv` | **Dados usados** |
| `*.mat` | Versões MATLAB (não usadas) |
| `predictive-maintance-using-ml-and-lstm-rnn.ipynb`, `ml-powered-maintenance-smarter-proactive.ipynb` | Notebooks de referência |
| `Filter_RUL_Project.ipynb` | **Nosso notebook** (gera tudo) |
| `figures/`, `results/` | Figuras, `numbers.json` e CSVs (métricas, previsões, **comparação com o ground truth**, faltantes, política) |
| `docs/Project_Report.docx` / `.pdf` | Relatório final |
| `requirements.txt` | Versões das bibliotecas |
