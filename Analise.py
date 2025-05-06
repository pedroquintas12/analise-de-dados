import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import numpy as np


# USAR streamlit run analise.py NO TERMINAL PARA RODAR O CODIGO

# Carregar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_excel("dataset_alunos.xlsx")
    df['Frequência (%)'] = df['Frequência (%)'].apply(lambda x: 1 if x < 40 else 0)
    return df

dados = carregar_dados()

st.title("📊 Análise de Evasão Escolar com Filtros Interativos e SHAP")

# Filtros na barra lateral
st.sidebar.header("Filtros")

turmas = st.sidebar.multiselect("Turmas", options=dados['Turma'].unique(), default=dados['Turma'].unique())
generos = st.sidebar.multiselect("Gênero", options=dados['Gênero'].unique(), default=dados['Gênero'].unique())
idades = st.sidebar.slider("Idade", int(dados['Idade'].min()), int(dados['Idade'].max()), (int(dados['Idade'].min()), int(dados['Idade'].max())))
faixa_renda = st.sidebar.multiselect("Faixa de Renda", options=dados['Faixa de Renda'].unique(), default=dados['Faixa de Renda'].unique())


# Aplicar filtros
dados_filtrados = dados[
    (dados['Turma'].isin(turmas)) &
    (dados['Gênero'].isin(generos)) &
    (dados['Idade'].between(idades[0], idades[1])) &
    (dados['Faixa de Renda'].isin(faixa_renda))
]

st.subheader("Distribuição da Evasão nos Dados Filtrados")
evadidos = dados_filtrados['Frequência (%)'].value_counts(normalize=True)
fig, ax = plt.subplots()
evadidos.plot(kind='bar', color=['green', 'red'], edgecolor='black', ax=ax)
ax.set_xticklabels(['Não Evadiu', 'Evadido'], rotation=0)
ax.set_ylabel('Proporção')
ax.set_title('Distribuição da Evasão')
st.pyplot(fig)

# Preparar dados para o modelo
dados_modelo = dados_filtrados.copy()
colunas_remover = ['ID','Nome' ,'Ano Letivo', 'Série', 'Turma']
dados_modelo = dados_modelo.drop(columns=colunas_remover)
dados_dummies = pd.get_dummies(dados_modelo, drop_first=True)
X = dados_dummies.drop(columns=['Frequência (%)'])
y = dados_dummies['Frequência (%)']

# Treino/teste
if len(X) > 5:
    X = X.fillna(0).astype('float64')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y[y == 1]) > 0 else None)

    modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    modelo.fit(X_train, y_train)

    explainer = shap.Explainer(modelo, X_train)
    shap_values = explainer(X_test)

    st.subheader("Importância das Variáveis (SHAP Summary)")
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(bbox_inches='tight')
    print(X_test.columns)


    # Selecionar aluno individual para análise
    st.subheader("Análise Individual com SHAP")
    idx = st.selectbox("Escolha um aluno do conjunto de teste:", range(len(X_test)))
    st.write("Aluno selecionado:")
    st.write(X_test.iloc[idx])
    # Opção para o usuário escolher a classe (0=Não Evadiu, 1=Evadiu)
    classe_escolhida = st.selectbox("Classe para análise (0=Não Evadiu, 1=Evadiu)", [0, 1])

    # Inspecionar a estrutura de shap_values.base_values
    st.write(type(shap_values))
    st.write(type(shap_values.base_values))
    st.write(shap_values.base_values)

    # Ajustar base_values para garantir que é um valor escalar
    if isinstance(shap_values.base_values, (list, np.ndarray)):
        # Caso seja uma lista ou array, podemos pegar o primeiro valor
        base_values = float(shap_values.base_values[0])  # Usando o primeiro valor
    else:
        # Se for um único valor escalar, usamos diretamente
        base_values = float(shap_values.base_values)


    # Plotar gráfico waterfall
    shap.plots.waterfall(shap_values[idx][classe_escolhida], max_display=10, show=False)
    st.pyplot(bbox_inches='tight')


else:
    st.warning("Poucos dados após o filtro. Amplie os critérios para visualizar os gráficos SHAP.")
