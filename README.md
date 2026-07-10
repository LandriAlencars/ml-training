# 🏠 ml-training — Previsão de Preços de Imóveis com Machine Learning

Projeto prático de Machine Learning desenvolvido no **Grupo de Estudos: Aprendizado de Máquina (GE-AM)**, do núcleo **Orion — Pesquisa e Inovação**, como aplicação dos fundamentos de ML apresentados no Encontro 2 (Fundamentos de ML).

O projeto resolve um problema de **regressão**: prever o valor mediano de imóveis (`median_house_value`) a partir de características socioeconômicas e geográficas de distritos da Califórnia (California Housing Dataset), percorrendo todo o pipeline de um projeto de ML — da coleta dos dados à avaliação dos modelos.

---

## 📌 Objetivo Geral

Aplicar os fundamentos de Machine Learning (coleta, pré-processamento, engenharia de atributos, treinamento, validação cruzada e avaliação de modelos) para resolver um problema real de regressão, utilizando um dataset público amplamente usado como referência didática em ML.

---

## 🗂️ Dataset

- **Nome:** California Housing Dataset
- **Fonte:** [ageron/handson-ml2](https://github.com/ageron/handson-ml2) (dataset clássico do livro *Hands-On Machine Learning*, de Aurélien Géron)
- **Registros:** 20.640 distritos
- **Variável-alvo:** `median_house_value` (valor mediano dos imóveis, em USD)
- **Principais features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity` (categórica)
- **Observação relevante:** a coluna `total_bedrooms` possui valores ausentes (20.433 de 20.640 registros), tratados via imputação pela mediana. Os valores de `median_house_value` também são *capados* em 500.001 USD, um viés conhecido do dataset.

---

## 🧠 Metodologia (Etapas do Projeto)

1. **Coleta de dados** — download automático do dataset via script (`fetch_housing_data`).
2. **Exploração (EDA)** — `.info()`, `.describe()`, histogramas de todas as features e mapas de dispersão geográfica (`longitude` x `latitude`) coloridos por valor do imóvel.
3. **Divisão treino/teste estratificada** — split 80/20 estratificado pela categoria de renda (`income_cat`, derivada de `median_income`) usando `StratifiedShuffleSplit`, garantindo que treino e teste representem proporcionalmente as faixas de renda da população.
4. **Análise de correlação** — identificação de `median_income` como variável mais correlacionada com o preço do imóvel (correlação ≈ 0,69).
5. **Engenharia de atributos (Feature Engineering)** — criação de `rooms_per_household`, `bedrooms_per_room` e `population_per_household`, atributos combinados mais informativos que os originais.
6. **Pré-processamento** — pipeline com `SimpleImputer` (mediana), transformador customizado de atributos combinados, `StandardScaler` para variáveis numéricas e `OneHotEncoder` para a variável categórica `ocean_proximity`, unificados via `ColumnTransformer`.
7. **Treinamento** — três algoritmos de regressão treinados sobre os dados preparados.
8. **Avaliação** — cálculo do RMSE (Root Mean Squared Error) no conjunto de treino e via validação cruzada (10-fold).

---

## 🤖 Modelos Utilizados

| Modelo | Biblioteca |
|---|---|
| Regressão Linear | `sklearn.linear_model.LinearRegression` |
| Árvore de Decisão | `sklearn.tree.DecisionTreeRegressor` |
| Random Forest | `sklearn.ensemble.RandomForestRegressor` (`n_estimators=100`) |

---

## 📊 Resultados

RMSE (erro médio, em USD — quanto menor, melhor). Para referência, o desvio padrão do preço dos imóveis no dataset é de ≈ 115.396 USD.

| Modelo | RMSE (treino) | RMSE médio (validação cruzada, 10-fold) | Desvio padrão (CV) |
|---|---:|---:|---:|
| Regressão Linear | 68.627,87 | 69.104,08 | 2.880,33 |
| Árvore de Decisão | 0,00 | 71.324,53 | 2.649,50 |
| **Random Forest** | 18.650,70 | **50.435,58** | 2.203,34 |

**Destaque:** o Random Forest apresentou o melhor desempenho em validação cruzada, reduzindo o erro em cerca de 27% em relação à Regressão Linear e à Árvore de Decisão.

---

## 💡 Conclusões e Principais Aprendizados

- **Overfitting na prática:** a Árvore de Decisão atingiu RMSE = 0 no treino (decorou os dados), mas seu erro subiu para ~71 mil na validação cruzada — um exemplo direto de que **baixo erro de treino não garante boa generalização**.
- **Random Forest generaliza melhor:** por ser um método *ensemble* (combina várias árvores), reduziu a variância do modelo e teve o menor RMSE em validação cruzada entre os três algoritmos testados.
- **A validação cruzada foi essencial** para revelar o overfitting da árvore de decisão, que passaria despercebido se avaliássemos apenas o erro de treino.
- **Feature engineering agregou valor:** o atributo derivado `rooms_per_household` mostrou correlação mais forte com o preço do que `total_rooms` isoladamente.
- **Qualidade dos dados importa:** valores ausentes em `total_bedrooms` e o teto artificial em `median_house_value` (500.001 USD) são limitações reais do dataset que impactam o desempenho de qualquer modelo.

---

## ⚠️ Desafios e Limitações

- Dados ausentes em `total_bedrooms`, tratados por imputação (mediana).
- Teto (*cap*) no valor-alvo `median_house_value`, o que pode distorcer a avaliação de imóveis de alto padrão.
- Nenhum ajuste de hiperparâmetros (`GridSearchCV`/`RandomizedSearchCV`) foi realizado até o momento — os modelos foram treinados com parâmetros majoritariamente padrão.
- Modelos avaliados apenas por RMSE; outras métricas (MAE, R²) podem trazer visões complementares.

---

## 📁 Estrutura do Repositório

```
ml-training/
├── Housing.ipynb          # Notebook principal com todo o pipeline de ML
├── Untitled.ipynb          # Testes exploratórios com TensorFlow
├── datasets/housing/       # Dados baixados (housing.csv)
├── grafico.png             # Histogramas das features do dataset
├── requirements.txt        # Dependências do projeto
└── README.md
```

---

## ⚙️ Como Executar

```bash
# Clonar o repositório
git clone https://github.com/LandriAlencars/ml-training.git
cd ml-training

# Instalar dependências
pip install pandas matplotlib numpy scikit-learn
pip install tensorflow   # opcional, usado apenas em Untitled.ipynb

# Rodar o notebook
jupyter notebook Housing.ipynb
```

**Linguagem:** Python · **Bibliotecas:** pandas, numpy, matplotlib, scikit-learn · **Ambiente:** Jupyter Notebook

---

## 🚀 Próximos Passos

- Ajustar hiperparâmetros dos modelos (`GridSearchCV` / `RandomizedSearchCV`), especialmente do Random Forest.
- Avaliar o modelo final no conjunto de teste (`strat_test_set`), reservado desde o início e ainda não utilizado na avaliação.
- Analisar a importância das features (`feature_importances_`) do Random Forest.
- Testar outros algoritmos (SVM para regressão, XGBoost).
- Aplicar os mesmos fundamentos a um problema de **classificação binária** (ex.: Titanic, Heart Disease), conforme o desafio proposto no Encontro 2 do GE-AM.

---

## 🙏 Créditos

- Dataset e estrutura de pipeline inspirados no livro **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow**, de Aurélien Géron.
- Projeto desenvolvido no contexto do **Grupo de Estudos: Aprendizado de Máquina (GE-AM)** — Orion Pesquisa e Inovação, sob orientação de Ms. Keila Barbosa Costa.
