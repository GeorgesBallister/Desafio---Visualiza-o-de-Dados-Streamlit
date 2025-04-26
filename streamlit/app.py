import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Análise de Vendas", layout="wide")

# --- TÍTULO ---
st.title('🛒 Data Storytelling: Análise de Vendas')

st.markdown("""
Este projeto tem como objetivo analisar dados de vendas para entender padrões de consumo,
identificar oportunidades de crescimento e sugerir melhorias baseadas nos dados propostos pelo professor **Geraldo** do ****MBA*** de Data Science e IA.*
""")

st.markdown("""
Alem de nos proporcionar vizualisações dos dados, com esta analise, eu consegui visualizar possiveis estrategias que podem ser tomadas para entregar de forma mais acertiva resultados que se converta em retorno financeiro positivo!
            """)

st.markdown("""
> Codigo desenvolvido por **Georges Ballister**.
            """)

# --- CARREGAR DADOS ---
@st.cache_data
def load_data():
    data = pd.read_csv('../sales_data.csv')
    return data

df = load_data()

# --- EXIBIR DADOS INICIAIS ---
st.subheader('Visualização Inicial dos Dados')
st.dataframe(df.head())

# --- PRÉ-PROCESSAMENTO ---
df['Date_Sold'] = pd.to_datetime(df['Date_Sold'], errors='coerce')
df.dropna(subset=['Date_Sold'], inplace=True)

# Criar coluna de Mês
df['Month'] = df['Date_Sold'].dt.month

# --- ANÁLISES ---

import calendar

# Vendas por Mês
st.header('📈 Vendas por Mês')

# Agrupar vendas por mês
sales_by_month = df.groupby('Month')['Total_Sales'].sum().reset_index()

# Converter número do mês para nome do mês (em português)
sales_by_month['Month_Name'] = sales_by_month['Month'].apply(lambda x: calendar.month_name[x])

# Traduzir para português manualmente (porque calendar.month_name retorna em inglês)
# Alternativamente, você pode fazer um mapeamento personalizado:
meses_pt = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril'
}
sales_by_month['Month_Name'] = sales_by_month['Month'].map(meses_pt)

# Garantir que os meses fiquem na ordem correta
sales_by_month['Month_Name'] = pd.Categorical(
    sales_by_month['Month_Name'],
    categories=list(meses_pt.values()),
    ordered=True
)

# Criar o gráfico
fig_month = px.bar(
    sales_by_month,
    x='Month_Name', y='Total_Sales',
    labels={'Month_Name': 'Mês', 'Total_Sales': 'Faturamento'},
    text_auto='.2s',
    title='Faturamento Total por Mês'
)

# Exibir no Streamlit
st.plotly_chart(fig_month, use_container_width=True)

# Observações
st.markdown("""
**O que podemos observar:**  
- Existe uma tendência clara de aumento nas vendas em determinados meses.
- Podemos focar campanhas promocionais nos meses de maior movimento para aumentar ainda mais o faturamento.
""")

# Faturamento por Categoria
st.header('🏷️ Faturamento por Categoria')
sales_by_category = df.groupby('Category')['Total_Sales'].sum().reset_index()

fig_category = px.bar(
    sales_by_category,
    x='Category', y='Total_Sales',
    labels={'Category': 'Categoria', 'Total_Sales': 'Faturamento'},
    text_auto='.2s',
    title='Faturamento Total por Categoria'
)
st.plotly_chart(fig_category, use_container_width=True)

st.markdown("""
**O que podemos observar:**  
- Algumas categorias concentram uma parte significativa do faturamento.
- Investir em categorias com maior potencial pode trazer crescimento mais rápido.
""")

# Produto Mais Vendido
st.header('🎯 Produtos mais Vendidos')
products = df.groupby('Product_Name')['Quantity_Sold'].sum().sort_values(ascending=False).reset_index()

fig_product = px.bar(
    products,
    x='Product_Name', y='Quantity_Sold',
    labels={'Product_Name': 'Produto', 'Quantity_Sold': 'Quantidade Vendida'},
    text_auto=True,
    title='Produtos Mais Vendidos'
)
st.plotly_chart(fig_product, use_container_width=True)

st.markdown("""
**O que podemos observar:**  
- Alguns produtos têm uma aceitação muito maior no mercado.
- Produtos com alta demanda devem ter maior disponibilidade em estoque para evitar perda de vendas.
""")

# Relação Preço x Quantidade Vendida
st.header('💸 Relação entre Preço e Quantidade Vendida')
fig_relation = px.scatter(
    df,
    x='Price', y='Quantity_Sold',
    labels={'Price': 'Preço Unitário', 'Quantity_Sold': 'Quantidade Vendida'},
    title='Preço vs Quantidade Vendida'
)
st.plotly_chart(fig_relation, use_container_width=True)

st.markdown("""
**O que podemos observar:**  
- Produtos mais baratos tendem a vender em maior volume.
- Estratégias de precificação podem ser ajustadas para maximizar o volume de vendas sem perder margens.
""")

st.header('📈 Previsão de Vendas por Categorias')


from sklearn.linear_model import LinearRegression
import numpy as np

# Agrupar vendas por Categoria e Mês
sales_category_month = df.groupby(['Month', 'Category'])['Total_Sales'].sum().reset_index()

# Adicionar coluna de tempo para usar na regressão
sales_category_month['Time'] = sales_category_month['Month']

# Criar dicionário para guardar previsões
predictions = {}

# Obter lista única de categorias
categories = sales_category_month['Category'].unique()

# Definir horizonte de previsão (ex: próximos 3 meses)
future_months = [6, 7, 8, 9]  # considerando que o dataset atual vai até mês 12 (dezembro)

for category in categories:
    # Filtrar dados da categoria
    category_data = sales_category_month[sales_category_month['Category'] == category]
    
    X = category_data[['Time']]
    y = category_data['Total_Sales']
    
    # Treinar modelo de regressão
    model = LinearRegression()
    model.fit(X, y)
    
    # Prever para os meses futuros
    X_future = np.array(future_months).reshape(-1, 1)
    y_future = model.predict(X_future)
    
    # Salvar previsões
    predictions[category] = {
        'Future_Months': future_months,
        'Predicted_Sales': y_future
    }

# Organizar as previsões em um DataFrame para visualização
predictions_df = pd.DataFrame({
    'Category': [],
    'Month': [],
    'Predicted_Sales': []
})

for category, values in predictions.items():
    temp_df = pd.DataFrame({
        'Category': [category] * len(values['Future_Months']),
        'Month': values['Future_Months'],
        'Predicted_Sales': values['Predicted_Sales']
    })
    predictions_df = pd.concat([predictions_df, temp_df], ignore_index=True)

# Concatenar dados reais + previstos
full_data = pd.concat([
    sales_category_month[['Category', 'Month', 'Total_Sales']].rename(columns={'Total_Sales': 'Sales'}),
    predictions_df.rename(columns={'Predicted_Sales': 'Sales'})
], ignore_index=True)

# Criar o gráfico
fig = px.line(
    full_data,
    x='Month', y='Sales', color='Category',
    markers=True,
    title='📈 Crescimento Real e Projetado das Categorias de Produtos',
    labels={'Month': 'Mês', 'Sales': 'Vendas'}
)

# Exibir gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)

# Observações
st.markdown("""
**Insights da Análise:**  
- A projeção mostra as tendências de crescimento ou queda para cada categoria nos próximos meses.
- Categorias com crescimento acelerado merecem mais investimento e atenção em estoque e marketing.
- Categorias com queda podem indicar sazonalidade ou necessidade de reavaliação de estratégia.

**Nota:**  
Essas projeções assumem que o comportamento passado se manterá semelhante no curto prazo.
""")


# --- CONCLUSÕES E INSIGHTS ---
st.header('📊 Insights da Análise de Vendas')

st.markdown("""
### 📈 Vendas por Mês

- O mês de **Março** destacou-se com o **maior faturamento**, ultrapassando **R$ 140.000** em vendas.
- Essa informação revela uma **sazonalidade clara**, sugerindo que campanhas promocionais devem ser focadas nesse período para maximizar o faturamento.
- **Sugestão Estratégica:** Planejar eventos promocionais especiais em Março para potencializar o aproveitamento do alto fluxo de vendas.

---

### 🏷️ Faturamento por Categoria

- A categoria **Clothing** foi a que **mais faturou**, enquanto a categoria **Electronics** teve o **menor desempenho** em vendas.
- Isso indica que produtos de vestuário têm forte aceitação, enquanto a linha de eletrônicos pode necessitar de ações específicas para melhorar sua performance.
- **Sugestão Estratégica:** Focar esforços de marketing em Clothing e investigar melhorias de produto, preço ou promoção para a categoria Electronics.

---

### 🎯 Produtos Mais Vendidos

- O **Produto 10 (Clothing)** foi o item **mais vendido**.
- O **Produto 16 (Clothing)** foi o **segundo** mais vendido.
- O **Produto 13 (Grocery)** ficou em **terceiro** lugar em vendas.
- Esses produtos mostram uma **clara preferência dos consumidores**.
- **Solução Recomendada:** Garantir estoque robusto para os produtos campeões e avaliar a possibilidade de cross-selling entre as categorias Clothing e Grocery.

---

### 💸 Relação Preço vs Quantidade Vendida

- Observou-se que **produtos com preço abaixo de R$ 100** tiveram um **desempenho superior** em volume de vendas.
- **Interpretação:** Existe uma sensibilidade ao preço entre os consumidores analisados.
- **Estratégia Recomendada:** Implementar **precificação competitiva** para produtos abaixo de R$ 100, e criar promoções ou kits promocionais para estimular ainda mais o volume.

---

### 📈 Previsão de Crescimento por Categorias

- A projeção de vendas entre os meses **6 a 9** aponta uma tendência de **estabilização**.
- **Importante:** Devido à limitação e à incompletude dos dados atuais, a **acurácia da previsão é moderada**.
- Apesar disso, a tendência geral é de **crescimento moderado** nas categorias já consolidadas.
- **Nota:** Para maior precisão em futuras análises, recomenda-se o uso de séries temporais mais completas ou modelos preditivos mais robustos (como ARIMA, Prophet, etc.).

---

# 🔎 Conclusões Gerais

- **Março é o principal mês de vendas** — campanhas e estoques devem ser reforçados nesse período.
- **Categoria Clothing domina o faturamento** — foco em marketing, estoque e novos lançamentos.
- **Produtos campeões** precisam de reforço logístico para não sofrer rupturas de estoque.
- **Preço acessível** (< R$ 100) é um fator decisivo de compra — adaptar ofertas e pricing.
- **Tendência de crescimento moderado** identificada — planejamento antecipado de estoque e vendas recomendado para os próximos meses.

---
""")

