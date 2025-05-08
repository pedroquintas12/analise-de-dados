
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import numpy as np

@st.cache_data
def carregar_dados():
    df = pd.read_excel("dataset_alunos.xlsx")
    df['Frequência (%)'] = df['Frequência (%)'].apply(lambda x: 1 if x < 40 else 0)
    return df

dados = carregar_dados()

st.title("📊 Análise de Evasão Escolar com Filtros Interativos e SHAP")

st.sidebar.header("Filtros")
turmas_opts = sorted(dados['Turma'].unique())
generos_opts = sorted(dados['Gênero'].unique())
faixa_renda_opts = sorted(dados['Faixa de Renda'].unique())

turmas = st.sidebar.multiselect("Turmas", options=turmas_opts, default=turmas_opts)
generos = st.sidebar.multiselect("Gênero", options=generos_opts, default=generos_opts)
idades = st.sidebar.slider("Idade", int(dados['Idade'].min()), int(dados['Idade'].max()), (int(dados['Idade'].min()), int(dados['Idade'].max())))
faixa_renda = st.sidebar.multiselect("Faixa de Renda", options=faixa_renda_opts, default=faixa_renda_opts)

dados_filtrados = dados[
    (dados['Turma'].isin(turmas)) &
    (dados['Gênero'].isin(generos)) &
    (dados['Idade'].between(idades[0], idades[1])) &
    (dados['Faixa de Renda'].isin(faixa_renda))
]

if dados_filtrados.empty:
    st.warning("Nenhum dado corresponde aos filtros selecionados. Por favor, ajuste os filtros.")
else:
    st.subheader("Distribuição da Evasão nos Dados Filtrados")
    evasao_counts = dados_filtrados['Frequência (%)'].value_counts(normalize=True).sort_index()

    fig_dist, ax_dist = plt.subplots()
    plot_data = pd.Series([0.0, 0.0], index=['Não Evadiu (0)', 'Evadiu (1)'])
    if 0 in evasao_counts.index:
        plot_data['Não Evadiu (0)'] = evasao_counts[0]
    if 1 in evasao_counts.index:
        plot_data['Evadiu (1)'] = evasao_counts[1]

    plot_data.plot(kind='bar', color=['green', 'red'], edgecolor='black', ax=ax_dist)
    ax_dist.set_xticklabels(plot_data.index, rotation=0)
    ax_dist.set_ylabel('Proporção')
    ax_dist.set_title('Distribuição da Evasão')
    st.pyplot(fig_dist)
    plt.close(fig_dist)

    st.markdown("""
                **🔍 Interpretação:**  
                Este gráfico mostra a proporção de alunos que **evadiram (vermelho)** e **não evadiram (verde)** com base nos filtros aplicados.  
                A evasão é considerada quando a frequência está abaixo de 40%.
                """)

    dados_modelo = dados_filtrados.copy()
    colunas_remover = ['ID','Nome' ,'Ano Letivo', 'Série', 'Turma']
    dados_modelo = dados_modelo.drop(columns=colunas_remover, errors='ignore')
    dados_dummies = pd.get_dummies(dados_modelo, drop_first=True, dummy_na=False)

    if 'Frequência (%)' not in dados_dummies.columns:
        st.error("A coluna 'Frequência (%)' (variável alvo) não foi encontrada após a transformação dos dados.")
    else:
        X = dados_dummies.drop(columns=['Frequência (%)'])
        y = dados_dummies['Frequência (%)']

        if X.empty:
            st.warning("Não há features para treinar o modelo.")
        elif len(X) > 5 and len(y.unique()) > 1:
            X = X.fillna(0).astype('float64')
            stratify_option = y if y.value_counts().min() > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_option
            )

            if len(X_train) > 0 and len(X_test) > 0 and len(y_train.unique()) > 1:
                modelo = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                modelo.fit(X_train, y_train)

                explainer = shap.Explainer(modelo, X_train)
                shap_values = explainer(X_test)

                st.subheader("Importância das Variáveis (Gráfico de Pizza)")
                feature_names = X_test.columns
                shap_vals_abs_mean = np.abs(shap_values.values).mean(axis=0)
                if shap_vals_abs_mean.ndim == 2:
                    shap_vals_abs_mean = shap_vals_abs_mean.mean(axis=1)

                shap_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': shap_vals_abs_mean
                }).sort_values(by='importance', ascending=False)

                top_n = 10
                fig_pizza, ax_pizza = plt.subplots()
                ax_pizza.pie(
                    shap_importance_df['importance'].head(top_n),
                    labels=shap_importance_df['feature'].head(top_n),
                    autopct='%1.1f%%',
                    startangle=140
                )
                ax_pizza.axis('equal')
                plt.title("Importância das Variáveis (Top 10 - SHAP)")
                st.pyplot(fig_pizza)
                plt.close(fig_pizza)

                st.markdown("""
                            **🔍 Interpretação:**  
                            Este gráfico de pizza mostra as **10 variáveis mais importantes** para o modelo prever a evasão.  
                            Quanto maior a fatia, maior o impacto daquela variável no resultado.
                            """)

                st.subheader("Análise Individual com SHAP")
                if not X_test.empty:
                    idx = st.selectbox("Escolha um aluno do conjunto de teste:", range(len(X_test)), key="aluno_select_shap")
                    st.write("Aluno selecionado (features):")
                    st.write(X_test.iloc[idx])

                    explicacao_individual = shap_values[idx]

                    st.markdown("##### Explicação para a Classe 'Não Evadiu' (0)")
                    valores = explicacao_individual.values[:, 0]
                    features = X_test.columns
                    top_indices = np.argsort(np.abs(valores))[-10:][::-1]

                    fig_bar_nao_evadiu, ax_bar_nao_evadiu = plt.subplots()
                    ax_bar_nao_evadiu.barh(
                        [features[i] for i in top_indices],
                        valores[top_indices],
                        color='green'
                    )
                    ax_bar_nao_evadiu.set_xlabel("Valor SHAP")
                    ax_bar_nao_evadiu.set_title("Top 10 Contribuições - Não Evadiu (0)")
                    ax_bar_nao_evadiu.invert_yaxis()
                    st.pyplot(fig_bar_nao_evadiu)
                    plt.close(fig_bar_nao_evadiu)

                    st.markdown("""
**🔍 Interpretação:**  
Este gráfico mostra **os fatores que levaram o modelo a prever que este aluno não evadiu**.  
Barras verdes positivas ajudam a manter o aluno na escola, enquanto barras negativas sugerem risco.
""")

                    if explicacao_individual.values.shape[1] > 1:
                        st.markdown("##### Explicação para a Classe 'Evadiu' (1)")
                        valores = explicacao_individual.values[:, 1]
                        top_indices = np.argsort(np.abs(valores))[-10:][::-1]

                        fig_bar_evadiu, ax_bar_evadiu = plt.subplots()
                        ax_bar_evadiu.barh(
                            [features[i] for i in top_indices],
                            valores[top_indices],
                            color='red'
                        )
                        ax_bar_evadiu.set_xlabel("Valor SHAP")
                        ax_bar_evadiu.set_title("Top 10 Contribuições - Evadiu (1)")
                        ax_bar_evadiu.invert_yaxis()
                        st.pyplot(fig_bar_evadiu)
                        plt.close(fig_bar_evadiu)

                        st.markdown("""
**🔍 Interpretação:**  
Este gráfico mostra **os fatores que levaram o modelo a prever que este aluno evadiu**.  
Barras vermelhas positivas aumentam a chance de evasão, e negativas atuam como proteção.
""")

                    else:
                        st.info("Modelo não produziu explicações para a classe 'Evadiu (1)'.")
                else:
                    st.warning("Conjunto de teste vazio.")
            else:
                st.warning("Não há dados suficientes nos conjuntos de treino ou teste.")
        elif len(X) <= 2:
            st.warning("Poucos dados após o filtro (< 6 amostras).")
        else:
            st.warning("Não há variabilidade suficiente na variável de evasão.")
