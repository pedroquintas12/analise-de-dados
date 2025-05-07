import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import numpy as np

# USAR streamlit run analise.py NO TERMINAL PARA RODAR O CÓDIGO

# Carregar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_excel("dataset_alunos.xlsx")
    # Transforma 'Frequência (%)': 1 se < 40 (potencial evasão), 0 caso contrário
    df['Frequência (%)'] = df['Frequência (%)'].apply(lambda x: 1 if x < 40 else 0)
    return df

dados = carregar_dados()

st.title("📊 Análise de Evasão Escolar com Filtros Interativos e SHAP")

# Filtros na barra lateral
st.sidebar.header("Filtros")
turmas_opts = sorted(dados['Turma'].unique())
generos_opts = sorted(dados['Gênero'].unique())
faixa_renda_opts = sorted(dados['Faixa de Renda'].unique())


turmas = st.sidebar.multiselect("Turmas", options=turmas_opts, default=turmas_opts)
generos = st.sidebar.multiselect("Gênero", options=generos_opts, default=generos_opts)
idades = st.sidebar.slider("Idade", int(dados['Idade'].min()), int(dados['Idade'].max()), (int(dados['Idade'].min()), int(dados['Idade'].max())))
faixa_renda = st.sidebar.multiselect("Faixa de Renda", options=faixa_renda_opts, default=faixa_renda_opts)

# Aplicar filtros
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

    # Mapear os índices (0, 1) para rótulos e cores corretos
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
    plt.close(fig_dist) # Fecha a figura para liberar memória


    # Preparar dados para o modelo
    dados_modelo = dados_filtrados.copy()
    colunas_remover = ['ID','Nome' ,'Ano Letivo', 'Série', 'Turma']
    dados_modelo = dados_modelo.drop(columns=colunas_remover, errors='ignore') # errors='ignore' para não falhar se alguma coluna já foi removida

    # Converter colunas categóricas em dummies
    dados_dummies = pd.get_dummies(dados_modelo, drop_first=True, dummy_na=False)

    if 'Frequência (%)' not in dados_dummies.columns:
        st.error("A coluna 'Frequência (%)' (variável alvo) não foi encontrada após a transformação dos dados. Verifique os passos anteriores.")
    else:
        X = dados_dummies.drop(columns=['Frequência (%)'])
        y = dados_dummies['Frequência (%)']

        if X.empty:
            st.warning("Não há features (colunas X) para treinar o modelo após o pré-processamento. Verifique os dados e filtros.")
        elif len(X) > 5 and len(y.unique()) > 1: # Adicionado verificação para número mínimo de amostras
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

                # --- Correção para o SHAP Summary Plot ---
                st.subheader("Importância das Variáveis (SHAP Summary)")
                # shap.summary_plot já desenha na figura/eixo atual
                shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
                fig_summary = plt.gcf() # Pega a figura que o SHAP plotou
                st.pyplot(fig_summary)
                plt.close(fig_summary) # Fecha a figura para liberar memória
                # --- Fim da correção ---

                st.subheader("Análise Individual com SHAP")
                if not X_test.empty:
                    idx = st.selectbox("Escolha um aluno do conjunto de teste:", range(len(X_test)), key="aluno_select_shap")
                    st.write("Aluno selecionado (features):")
                    st.write(X_test.iloc[idx])

                    explicacao_individual = shap_values[idx]

                    # Gráfico waterfall para a Classe 0 ('Não Evadiu')
                    st.markdown("##### Explicação para a Classe 'Não Evadiu' (0)")
                    fig_waterfall_nao_evadiu, ax_waterfall_nao_evadiu = plt.subplots()
                    # !!! CORREÇÃO APLICADA AQUI !!!
                    plt.sca(ax_waterfall_nao_evadiu) # Define o eixo como atual
                    shap.plots.waterfall(explicacao_individual[:, 0], max_display=10, show=False) # Remove ax=
                    ax_waterfall_nao_evadiu.set_title("Explicação para Não Evadiu (0)") # Adiciona título ao eixo
                    st.pyplot(fig_waterfall_nao_evadiu)
                    plt.close(fig_waterfall_nao_evadiu) # Fecha a figura para liberar memória

                    # Gráfico waterfall para a Classe 1 ('Evadiu')
                    # Verifica se a classe 1 existe na explicação (shap_values pode ter saídas para menos classes se y_train não tiver todas)
                    if explicacao_individual.values.shape[1] > 1:
                        st.markdown("##### Explicação para a Classe 'Evadiu' (1)")
                        fig_waterfall_evadiu, ax_waterfall_evadiu = plt.subplots()
                        # !!! CORREÇÃO APLICADA AQUI !!!
                        plt.sca(ax_waterfall_evadiu) # Define o eixo como atual
                        shap.plots.waterfall(explicacao_individual[:, 1], max_display=10, show=False) # Remove ax=
                        ax_waterfall_evadiu.set_title("Explicação para Evadiu (1)") # Adiciona título ao eixo
                        st.pyplot(fig_waterfall_evadiu)
                        plt.close(fig_waterfall_evadiu) # Fecha a figura para liberar memória
                    else:
                        st.info("Modelo não produziu explicações para a classe 'Evadiu (1)', possivelmente devido à ausência ou raridade desta classe nos dados de treino.")

                else:
                    st.warning("Conjunto de teste vazio. Não é possível exibir análise individual.")
            elif not (len(y_train.unique()) > 1):
                 st.warning("Não há variabilidade suficiente na variável de evasão nos dados de TREINO filtrados (apenas uma classe presente). Ajuste os filtros para incluir ambas as classes ('Evadiu' e 'Não Evadiu') para treinar o modelo e visualizar os gráficos SHAP.")
            else:
                st.warning("Não há dados suficientes nos conjuntos de treino ou teste após a divisão. Ajuste os filtros.")

        elif len(X) <= 5:
             st.warning("Poucos dados após o filtro (< 6 amostras). Amplie os critérios para visualizar os gráficos SHAP.")
        else: # len(y.unique()) <= 1
            st.warning("Não há variabilidade suficiente na variável de evasão nos dados filtrados (apenas uma classe presente). Ajuste os filtros para incluir ambas as classes ('Evadiu' e 'Não Evadiu') para treinar o modelo e visualizar os gráficos SHAP.")