﻿# Análise de Performance de Hardware com Streamlit

Este projeto consiste em uma aplicação web desenvolvida em Python com Streamlit, destinada a analisar a performance de hardware a partir de dados fornecidos. A aplicação permite visualizar métricas de desempenho, realizar clusterização e análise descritiva dos dados, e exportar resultados em PDF.

## 📋 Índice

- [Recursos](#recursos)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Funcionalidades](#funcionalidades)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Tecnologias](#tecnologias)
- [Contribuição](#contribuição)

---

## 🌟 Recursos

- Análise descritiva dos dados
- Relação de desempenho entre variáveis (PRP e ERP)
- Avaliação de modelo preditivo com diversas métricas
- Visualização interativa de clusterização em 2D e 3D
- Importância das características dos dados para o modelo
- Exportação de relatórios em PDF

---

## 🛠️ Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/streamlit-app.git
   cd streamlit-app
   ```

2. Crie um ambiente virtual e ative-o:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Linux/macOS
   venv\Scripts\activate     # No Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Como Usar

1. Execute o aplicativo Streamlit:
   ```bash
   streamlit run app.py
   ```

2. Acesse o aplicativo em seu navegador no endereço:
   ```
   http://localhost:8501
   ```

3. Faça upload de um arquivo CSV com os dados de desempenho de hardware e selecione as análises desejadas para visualizar e exportar.

---

## ⚙️ Funcionalidades

A aplicação possui as seguintes funcionalidades:

- **Análise Descritiva**: Exibe um heatmap das estatísticas descritivas dos dados.
- **Relação PRP e ERP**: Mostra uma análise de dispersão entre as variáveis `Desemp_relat_publicado` e `Desemp_relat_estimado`.
- **Avaliação do Modelo**: Avalia um modelo preditivo com métricas como MAE, MSE, RMSE, MAPE e Acurácia.
- **Clusterização 3D e 2D**: Visualizações interativas dos clusters formados a partir de variáveis selecionadas.
- **Importância das Características**: Exibe a importância das variáveis utilizadas pelo modelo preditivo.
- **Exportação em PDF**: Gera um relatório PDF com as análises e visualizações selecionadas.

---

## 📂 Estrutura do Projeto

```plaintext
streamlit-app/
├── assets/                   # Arquivos de imagens e recursos visuais
├── data/                     # Dados de entrada (não incluídos no repositório)
├── pages/                    # Páginas adicionais do Streamlit (opcional)
├── result_img/               # Imagens de resultados geradas pela aplicação
├── app.py                    # Código principal da aplicação
├── requirements.txt          # Dependências do projeto
└── .gitignore                # Arquivos e pastas ignorados pelo Git
```

---

## 🛠️ Tecnologias

As principais bibliotecas e frameworks utilizados no desenvolvimento deste projeto são:

- **[Streamlit](https://streamlit.io/)** - Interface web interativa para dados em Python
- **[Pandas](https://pandas.pydata.org/)** - Manipulação e análise de dados
- **[Matplotlib](https://matplotlib.org/)** - Visualização de dados
- **[Plotly](https://plotly.com/python/)** - Visualização interativa de dados
- **[Seaborn](https://seaborn.pydata.org/)** - Visualização estatística de dados
- **[Scikit-learn](https://scikit-learn.org/stable/)** - Ferramentas de machine learning
- **[FPDF](http://www.fpdf.org/)** - Geração de PDFs
- **[Selenium](https://www.selenium.dev/)** - Automação de captura de gráficos em PNG

---

## 🤝 Contribuição

Contribuições são sempre bem-vindas! Para contribuir, siga os passos abaixo:

1. Faça um fork do projeto
2. Crie uma nova branch:
   ```bash
   git checkout -b feature/nova-funcionalidade
   ```
3. Commit suas alterações:
   ```bash
   git commit -m "Adiciona nova funcionalidade"
   ```
4. Envie para a branch principal:
   ```bash
   git push origin feature/nova-funcionalidade
   ```
5. Abra um Pull Request

---

## 👤 Autor

Criado por **[Ismael Jefte](https://github.com/ijbs-dev)**. 

---

