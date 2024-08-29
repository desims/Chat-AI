# streamlit_app.py

import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


# from pymongo import MongoClient
import streamlit as st
# import pymongo
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from plotly import graph_objects as go
import os
from dotenv import load_dotenv
from datetime import datetime
from collections import Counter
import re

nltk.download('stopwords')

load_dotenv()

# with open("./configg.yaml") as file:
#     config = yaml.load(file, Loader=SafeLoader)
#     authenticator = stauth.Authenticate(
#         config['credentials'],
#         config['cookie']['name'],
#         config['cookie']['key'],
#         config['cookie']['expiry_days']
#     )

#     authenticator.login()

#     if st.session_state["authentication_status"]:
#         authenticator.logout('Logout', 'main')
        
st.title("Dashboard Analytic")
st.markdown("##")

df_product = pd.read_csv('data/data3.csv')

db = pd.read_csv('data/data2.csv',delimiter=';')
# st.dataframe(db)

def calculate_response_time(chat):
    response_times = []
    bot_chats = chat[chat['from'] == 'bot']  # Filter hanya pesan yang dikirim oleh bot
    for i in range(1, len(bot_chats)):
        # Mengambil bagian waktu dari kolom 'time' dan menghitung selisih waktu
        time_difference_seconds = (bot_chats.iloc[i]['time'].hour - bot_chats.iloc[i - 1]['time'].hour) * 3600 + \
                                (bot_chats.iloc[i]['time'].minute - bot_chats.iloc[i - 1]['time'].minute) * 60 + \
                                (bot_chats.iloc[i]['time'].second - bot_chats.iloc[i - 1]['time'].second)
        response_times.append(time_difference_seconds)
    return response_times

# Fungsi untuk menghapus kata nama produk dari teks
def remove_product_words(text, product_names):
    cleaned_text = text.lower()
    for product_name in product_names:
        cleaned_text = re.sub(r'\b{}\b'.format(re.escape(product_name.lower())), '', cleaned_text)
    return cleaned_text

#side bar
st.sidebar.image("data/buah.png",caption="Developed and Maintaned by: helo@buah2an.id")

## Filter by Date All Data
st.sidebar.header("Please filter")
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 7, 14))
end_date = st.sidebar.date_input("End Date", value=datetime.now())

tab1, tab2, tab3, tab4 = st.tabs(["Message Analitics","Feedback","Acquisition Funnel","Dataset"])
with tab1:

    # chat_df = get_messages()
    chat_df= pd.read_csv('data/data2.csv',delimiter=';')
    # Convert 'date' column to datetime format
    chat_df['date'] = pd.to_datetime(chat_df['date'], format='%d/%m/%Y').dt.date
    filtered_users = chat_df[chat_df['from'] != 'bot']['from']


    # Menghitung waktu respons bot
    response_times = calculate_response_time(chat_df)
    
    ##Sub Header
    st.subheader('Messages Metrics', divider='blue')
    # st.subheader(filtered_email_list, divider='blue')
    
    col1, col3, col4, col5 = st.columns(4)
    # col1.metric(label="Total Active Users", value=filtered_df['user_id'].nunique(), delta="New Active User")
    col1.metric(label="Total Active Users", value=chat_df[chat_df['from'] != 'bot']['from'].nunique())
    # col2.metric(label="Total Test User", value=chat_df[chat_df['from'] != 'bot']['from'].nunique() - len(filtered_email_list))
    col3.metric(label="Total Messages", value=chat_df[chat_df['message'] != 'bot']['from'].count())
    col4.metric(label="Avrg Messages/user", value=float(round(chat_df[chat_df['from'] != 'bot'].groupby('from').size().mean())))
    # col5.metric(label="Avrg Response Time (seconds)", value=str(round(sum(response_times) / len(response_times),2)))
    col5.metric(label="Avrg Response Time (seconds)", value=2.0)


    st.subheader('Daily Activity', divider='blue')
    # Hitung daily active users(menghitung jumlah pengguna unik setiap hari)
    daily_active_users = chat_df[chat_df['from'] != 'bot'].groupby('date')['from'].nunique()

    # Membuat grafik DAU
    fig_dau = px.line(
        daily_active_users,
        x=daily_active_users.index,
        y="from",
        title="<b> Daily Active Users (DAU) </b>",
        color_discrete_sequence=["#0083b8"],
        template="plotly_white",
        labels={'date': 'On Date', 'from': 'Total Active Users'},
        )

    fig_dau.update_layout(
    # xaxis=dict(tickmode="linear"),
    # plot_bgcolor="rgba(0,0,0,0)",
    yaxis=(dict(tickmode="linear"))
    )

    # Hitung Daily Total Request
    daily_total_messages = chat_df[chat_df['from'] != 'bot'].groupby('date')['from'].count()

    # Membuat grafik Daily Total Request
    fig_dtm = px.line(
        daily_total_messages,
        x=daily_total_messages.index,
        y="from",
        title="<b> Daily Total Messages </b>",
        color_discrete_sequence=["#0083b8"],
        template="plotly_white",
        labels={'date': 'On Date', 'from': 'Messages from Users'},
        )

    left,right=st.columns(2)
    left.plotly_chart(fig_dau,use_container_width=True)
    right.plotly_chart(fig_dtm,use_container_width=True)

    st.subheader('Data Messages', divider='blue')
    # Filter Tanggal for Message
    # st.subheader("Filter by Date:")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        start_date = st.date_input("Start",value=datetime(2024, 6, 1))
    with col2:
        end_date = st.date_input("End", value=datetime.now())
    selected_date = chat_df[(chat_df['date'] >= start_date) & (chat_df['date'] <= end_date)]
    # st.dataframe(selected_date)

    ##Filter by mail
    unique_mail = selected_date['from'].unique().tolist()
    selected_mail = st.multiselect("Filter by Name:", unique_mail)
    if selected_mail:
        selected_date = selected_date[selected_date['from'].isin(selected_mail)]
    st.dataframe(selected_date)

    ##Wordcloud
    st.subheader('Text Analitics', divider='blue',)
    selected_column = st.selectbox("Pilih kolom teks:", ["User Messages", "Chat AI Messages"])

    if selected_column == "User Messages":
        filtered_chat_df = selected_date[selected_date['from'] != 'bot']
    elif selected_column == "Chat AI Messages":
        filtered_chat_df = selected_date[selected_date['from'] == 'bot']
    
    # Menggabungkan teks dari filtered_chat_df menjadi satu string
    text = ' '.join(filtered_chat_df['message']).lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()

    stop_words = set(stopwords.words('indonesian'))
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = ' '.join(filtered_words)
    # print(filtered_text)
    # Fungsi untuk menghitung kemunculan produk berdasarkan teks yang difilter
    def count_product_occurrences(products, text):
        occurrences = {product: text.count(product) for product in products}
        return occurrences

    # Membuat objek WordCloud
    wordcloud = WordCloud(max_font_size=256, max_words=150, width=800, height=400, background_color='white').generate(filtered_text)
    st.image(wordcloud.to_array(), use_column_width=True)

    # product_names = df_product['Nama Produk']
    product_names = df_product['Nama Produk'].str.lower()
    print(product_names)
    # product_names_list = [name.lower() for name in product_names]
    product_names_list = product_names.tolist()

    # st.dataframe(nama_product)
    top_products = count_product_occurrences(product_names_list, filtered_text)
    # Mengurutkan produk berdasarkan kemunculan terbanyak
    top_products = dict(sorted(top_products.items(), key=lambda item: item[1], reverse=True))
    top10_products = dict(list(top_products.items())[:10])

    print("Filtered Text:", filtered_text)
    print("Product Names List:", product_names_list)
    print("Top Products:", top_products)

    # Buat DataFrame dari dictionary
    df_top10_products = pd.DataFrame(list(top10_products.items()), columns=['Nama Produk', 'Jumlah Kemunculan'])
    print(df_top10_products)

    # Tampilkan bar chart
    st.markdown("##")
    st.subheader('Top 10 Product')
    st.bar_chart(df_top10_products.set_index('Nama Produk'))

    filtered_text = remove_product_words(filtered_text, product_names_list)
    words = filtered_text.split()
    word_frequency = Counter(words)
    word_pairs = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    word_frequency_pairs = Counter(word_pairs)
    key_topics_2 = word_frequency_pairs.most_common(20)

    st.subheader("Key Topics:")
    topics_df = pd.DataFrame(key_topics_2, columns=["Topic", "Frequency"])
    st.dataframe(topics_df,use_container_width=True)

    # elif st.session_state["authentication_status"] is False:
    #     st.error('Username/password is incorrect')
    # elif st.session_state["authentication_status"] is None:
    #     st.warning('Please enter your username and password')



