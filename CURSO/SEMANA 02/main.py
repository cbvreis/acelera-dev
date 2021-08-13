import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import random

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main()->None:
    '''
        Função que escreve os componentes do streamlit na tela
    :return:
    '''
    st.title('Acelera Dev - Codenation')
    file = st.file_uploader('Upload do arquivo .csv', type='csv' )
    #if file is not None:
    slider = st.slider('Valores',1,100)
    #df = pd.read_csv(file)
    df = pd.read_csv('data.csv')

    st.dataframe(df.head(slider))
    st.markdown('** Número de linhas e colunas**')
    st.markdown(df.shape)
    st.markdown('** Nome das colunas**')
    st.markdown(df.columns.tolist())
    st.markdown('** Contagem tipos de dados **')
    st.write(df.describe())
    st.write(df.isna().sum())
    btn_select = st.multiselect( 'Quais colunas serão analisadas?',
                    df.columns.tolist(),
                    df.columns.tolist()[2:6])

    if btn_select is not None:
        df_cols = df[btn_select]
        st.markdown('** Analise descritiva **')
        st.write(df.describe())
        st.markdown('** Percentual de dados faltantes das colunas analisadas **')
        st.write(df_cols.isna().sum()*100/df_cols.shape[0])
        btn_radio_cols = st.radio(
                                'Analisar separadamente a coluna',
                                (df_cols.columns)
                            )
        st.write(df[btn_radio_cols].value_counts())

        fig = px.histogram(df_cols,x=btn_radio_cols,title = 'Histograma da variável escolhida')
        st.plotly_chart(fig)





if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
