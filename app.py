# Instalar as dependências necessárias
# pip install fpdf plotly matplotlib selenium streamlit

# Importações
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from fpdf import FPDF
import streamlit as st
import base64
import io
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import tempfile
import os

# Configuração do WebDriver do Selenium
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Configuração para executar no navegador disponível no sistema
driver = webdriver.Chrome(options=chrome_options)

# Criar pasta 'result_img' se não existir
if not os.path.exists("result_img"):
    os.makedirs("result_img")

# Função para salvar o gráfico Plotly como imagem
def save_plotly_as_image(fig, file_name):
    file_path = os.path.join("result_img", file_name)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_html:
        fig.write_html(temp_html.name)
        driver.get("file:///{}".format(temp_html.name))
        time.sleep(2)  # Esperar o gráfico carregar completamente
        screenshot = driver.find_element(By.TAG_NAME, "body").screenshot_as_png
        with open(file_path, "wb") as f:
            f.write(screenshot)
    return screenshot

# Título da aplicação
st.set_page_config(page_title="Análise de Performance de Hardware", layout="wide")

st.title('Análise de Performance de Hardware')

# Botão para upload de arquivo CSV do PC local
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

if uploaded_file is not None:
    
    # Carregar o arquivo CSV
    df = pd.read_csv(uploaded_file)

    # Renomeando colunas
    df.columns = ["Nome_vendedor", "Nome_modelo", "Tempo_ciclo_maquina_ns", "Memoria_principal_min_kb", "Memoria_principal_max_kb", "Memoria_cache_kb", "Canais_min", "Canais_max", "Desemp_relat_publicado", "Desemp_relat_estimado"]

    # Criação do KMeans para adicionar a coluna 'Cluster' ao DataFrame
    cpu_features = df[["Tempo_ciclo_maquina_ns", "Memoria_principal_max_kb", "Memoria_cache_kb", "Canais_max", "Memoria_principal_min_kb", "Canais_min"]]
    kmeans = KMeans(n_clusters=3, n_init=10)
    df["Cluster"] = kmeans.fit_predict(cpu_features)

    # Confirmação da criação da coluna 'Cluster'
    st.write(df.head())

    # Análise preliminar dos dados
    desc = df.describe()
    st.write(desc)

    # Seleção de informações para visualização
    options = ["Análise Descritiva", "Relação PRP e ERP", "Avaliação do Modelo", "Clusterização 3D", "Clusterização 2D", "Importância das Características"]
    selected_options = st.multiselect("Selecione as informações para visualizar e incluir no relatório", options)

    if "Análise Descritiva" in selected_options:
        # Salvar análise descritiva como imagem
        plt.figure(figsize=(7, 3))
        sns.heatmap(desc.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=.5, linecolor='black')
        plt.title("Análise Descritiva do Dataset")
        plt.savefig("result_img/analise_descritiva.png")
        plt.close()
        st.image("result_img/analise_descritiva.png")

    if "Relação PRP e ERP" in selected_options:
        # Visualizar a relação entre PRP e ERP de forma interativa
        fig = px.scatter(
            df,
            x="Desemp_relat_publicado",
            y="Desemp_relat_estimado",
            color="Nome_vendedor",
            title="Relação entre Desempenho Relativo Publicado e Estimado",
            log_x=True,
            log_y=True,
            labels={
                "Desemp_relat_publicado": "Desempenho Relativo Publicado",
                "Desemp_relat_estimado": "Desempenho Relativo Estimado",
                "Nome_vendedor": "Nome do Vendedor"
            }
        )
        # Salvar o gráfico como imagem
        save_plotly_as_image(fig, "relacao_desempenho.png")
        st.image("result_img/relacao_desempenho.png")

    if "Avaliação do Modelo" in selected_options:
        # Definir X e y para avaliação do modelo
        X = df.drop(columns=["Nome_vendedor", "Nome_modelo", "Desemp_relat_estimado", "Desemp_relat_publicado"])
        y = df["Desemp_relat_estimado"]

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

        # Treinamento do modelo
        model = GradientBoostingRegressor(n_estimators=2000, max_depth=19, min_samples_leaf=4, learning_rate=0.002, random_state=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Cálculo das métricas
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - y_pred) / np.abs(y_test)))
        accuracy = 100 * (1 - mape)

        # Criar um dataframe para visualização dos erros
        metrics_df = pd.DataFrame({
            "Métrica": ["Erro Absoluto Médio (MAE)", "Erro Quadrático Médio (MSE)", "Raiz do Erro Quadrático Médio (RMSE)", "Erro Médio Absoluto Percentual (MAPE)", "Acurácia"],
            "Valor": [mae, mse, rmse, mape * 100, accuracy]
        })

        # Visualizar os resultados em um gráfico de barras interativo com Plotly
        fig = px.bar(
            metrics_df,
            x="Métrica",
            y="Valor",
            title="Avaliação do Modelo usando PRP como Predição Inicial",
            labels={"Valor": "Valor das Métricas"},
            text="Valor",
            width=800,
            height=400
        )

        # Adicionar formatação ao texto das barras
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

        # Melhorar layout do gráfico
        fig.update_layout(
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=40, b=80)
        )

        # Salvar o gráfico como imagem
        save_plotly_as_image(fig, "avaliacao_modelo.png")
        st.image("result_img/avaliacao_modelo.png")

    if "Clusterização 3D" in selected_options:
        # Visualização dos clusters de forma interativa em 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=df["Canais_max"],
            y=df["Memoria_principal_max_kb"],
            z=df["Memoria_principal_min_kb"],
            mode='markers',
            marker=dict(
                size=5,
                color=df["Cluster"],  # cores dos pontos de acordo com o cluster
                colorscale='Viridis',  # escolha da paleta de cores
                opacity=0.8
            ),
            text=df["Nome_vendedor"],  # informações ao passar o mouse sobre os pontos
            hoverinfo='text'
        )])

        fig.update_layout(
            title="Clusterização de CPUs",
            scene=dict(
                xaxis_title='Canais_max',
                yaxis_title='Memoria_principal_max_kb',
                zaxis_title='Memoria_principal_min_kb'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Salvar o gráfico como imagem
        save_plotly_as_image(fig, "clusterizacao_cpus_3d.png")
        st.image("result_img/clusterizacao_cpus_3d.png")

    if "Clusterização 2D" in selected_options:
        # Visualização dos clusters de forma interativa em 2D
        fig = px.scatter(
            df,
            x="Memoria_principal_max_kb",
            y="Memoria_cache_kb",
            color="Cluster",
            title="Clusterização de CPUs",
            labels={
                "Memoria_principal_max_kb": "Memória Principal Máxima (KB)",
                "Memoria_cache_kb": "Memória Cache (KB)",
                "Cluster": "Cluster"
            },
            hover_data=["Nome_vendedor", "Tempo_ciclo_maquina_ns", "Canais_min", "Canais_max"]
        )

        # Ajuste do layout
        fig.update_layout(
            title_font_size=20,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            legend_title_font_size=14,
            margin=dict(l=0, r=0, b=0, t=40),
            width=800,
            height=600
        )

        # Salvar o gráfico como imagem
        save_plotly_as_image(fig, "clusterizacao_cpus_2d.png")
        st.image("result_img/clusterizacao_cpus_2d.png")

    if "Importância das Características" in selected_options:
    # Preparação dos dados e treinamento do modelo
        X = df.drop(columns=["Nome_vendedor", "Nome_modelo", "Desemp_relat_estimado", "Desemp_relat_publicado"])
        y = df["Desemp_relat_estimado"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

        # Treinamento do modelo GradientBoostingRegressor
        gbr = GradientBoostingRegressor(n_estimators=2000, max_depth=19, min_samples_leaf=4, learning_rate=0.002, random_state=1000)
        gbr.fit(X_train, y_train)

        # Calcular a importância das características
        feature_importance = gbr.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        # Plotar a importância das características usando matplotlib
        plt.figure(figsize=(12, 6))
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(X.columns)[sorted_idx])
        plt.title('Importância das Características (CPU)')

        # Salvar o gráfico diretamente na pasta result_img
        importance_file_path = "result_img/importancia_caracteristicas.png"
        plt.savefig(importance_file_path)
        plt.close()

        # Verificar se o arquivo foi gerado corretamente antes de exibir
        if os.path.exists(importance_file_path):
            st.image(importance_file_path)
        else:
            st.error("Erro ao gerar a imagem de importância das características.")

    # Exportar resultados para PDF
    if st.button("Gerar Relatório em PDF"):
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, 'Relatório de Análise de Performance de Hardware', 0, 1, 'C')

            def chapter_title(self, title):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 10, title, 0, 1, 'L')
                self.ln(10)

            def chapter_body(self, body):
                self.set_font('Arial', '', 12)
                self.multi_cell(0, 10, body)
                self.ln()

            def insert_image(self, image_path, x, y, w, h):
                self.image(image_path, x=x, y=y, w=w, h=h)
                self.ln(h + 10)

        # Criação do PDF
        pdf = PDF()
        pdf.add_page()
        pdf.chapter_title('Análise Exploratória dos Dados')
        pdf.chapter_body(str(df.describe()))

        # Inserir imagens no PDF conforme seleção
        if "Análise Descritiva" in selected_options and os.path.exists("analise_descritiva.png"):
            pdf.chapter_title('Heatmap da Análise Descritiva')
            pdf.insert_image("analise_descritiva.png", x=10, y=None, w=190, h=60)

        if "Relação PRP e ERP" in selected_options and os.path.exists("relacao_desempenho.png"):
            pdf.chapter_title('Relação entre Desempenho Relativo Publicado e Estimado')
            pdf.insert_image("relacao_desempenho.png", x=10, y=None, w=190, h=100)

        if "Avaliação do Modelo" in selected_options and os.path.exists("avaliacao_modelo.png"):
            pdf.chapter_title('Avaliação do Modelo usando PRP como Predição Inicial')
            pdf.insert_image("avaliacao_modelo.png", x=10, y=None, w=190, h=60)

        if "Clusterização 3D" in selected_options and os.path.exists("clusterizacao_cpus_3d.png"):
            pdf.chapter_title('Clusterização de CPUs (3D)')
            pdf.insert_image("clusterizacao_cpus_3d.png", x=10, y=None, w=190, h=100)

        if "Clusterização 2D" in selected_options and os.path.exists("clusterizacao_cpus_2d.png"):
            pdf.chapter_title('Clusterização de CPUs (2D)')
            pdf.insert_image("clusterizacao_cpus_2d.png", x=10, y=None, w=190, h=60)

        if "Importância das Características" in selected_options and os.path.exists("importancia_caracteristicas.png"):
            pdf.chapter_title('Importância das Características')
            pdf.insert_image("importancia_caracteristicas.png", x=10, y=None, w=190, h=60)

        # Salvar o PDF
        pdf_file = "Relatorio_Analise_Performance_Hardware.pdf"
        pdf.output(pdf_file)

        # Download do PDF
        with open(pdf_file, "rb") as file:
            btn = st.download_button(
                label="Download PDF",
                data=file,
                file_name=pdf_file,
                mime="application/pdf"
            )

# Fechar o driver do Selenium
driver.quit()
    
