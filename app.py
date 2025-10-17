# Impor library yang diperlukan
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

# Impor komponen machine learning dari scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity # Untuk mencari kemiripan

# --- KONFIGURASI APLIKASI ---
app = Flask(__name__)

# --- VARIABEL GLOBAL DAN KONFIGURASI ---
model_assets = {}

# Kamus untuk mendeteksi kata kunci ambigu dan pilihan yang ditawarkan
AMBIGUOUS_KEYWORDS = {
    'lemot': ['Internet', 'Device'],
    'lambat': ['Internet', 'Device'],
    'tidak bisa connect': ['Internet', 'Aplikasi'],
    'tidak terhubung': ['Internet', 'Device'],
    'error': ['Aplikasi', 'Sistem Operasi'],
    'mati': ['Device', 'Listrik'],
    'ngehang': ['Aplikasi', 'Device'],
    'macet': ['Aplikasi', 'Device']
}

# --- FUNGSI UTAMA MACHINE LEARNING ---
def train_unsupervised_model():
    """
    Membaca dataset, melatih model K-Means, dan menyiapkan aset
    untuk prediksi kategori serta rekomendasi solusi.
    """
    try:
        # 1. Membaca dataset
        df = pd.read_csv('merged_clean.csv')
        df.dropna(subset=['gangguan', 'solusi'], inplace=True)
        df['gangguan'] = df['gangguan'].astype(str)
        df['solusi'] = df['solusi'].astype(str)
        problems = df['gangguan'].tolist()
        print(f"Dataset 'merged_clean.csv' berhasil dimuat dengan {len(problems)} data.")
    except FileNotFoundError:
        print("Error: File 'merged_clean.csv' tidak ditemukan!")
        return None

    # 2. Vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=['yang', 'di', 'dan', 'tidak', 'saya', 'komputer', 'laptop', 'bisa', 'mau', 'itu', 'ini', 'ke', 'dari']
    )
    tfidf_matrix = vectorizer.fit_transform(problems)
    print("Teks berhasil diubah menjadi vektor TF-IDF.")

    # 3. Clustering
    num_clusters = 5
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    kmeans_model.fit(tfidf_matrix)
    df['cluster'] = kmeans_model.labels_
    print(f"Model K-Means berhasil dilatih dan membuat {num_clusters} klaster.")

    # 4. Memberi Nama Klaster Secara Otomatis
    cluster_names = {}
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :3]]
        cluster_name = ' & '.join(top_terms).replace('_', ' ').title()
        cluster_names[i] = f"Kategori: {cluster_name}"
    print("Nama klaster dinamis berhasil dibuat.")

    # 5. Simpan semua aset
    return {
        'vectorizer': vectorizer,
        'kmeans_model': kmeans_model,
        'cluster_names': cluster_names,
        'dataframe': df
    }

# --- FUNGSI UNTUK ANALISIS AWAL ---
def analyse_initial_problem(description, assets):
    """
    Menganalisis masalah awal. Jika ambigu, tawarkan pilihan.
    Jika tidak, langsung berikan rekomendasi solusi.
    """
    # 1. Cek apakah ada kata kunci ambigu dalam deskripsi masalah
    description_lower = description.lower()
    for keyword, choices in AMBIGUOUS_KEYWORDS.items():
        if keyword in description_lower:
            # Jika ditemukan, kembalikan daftar pilihan, bukan solusi
            return None, [], choices

    # 2. Jika tidak ambigu, lanjutkan dengan logika clustering
    vectorizer = assets['vectorizer']
    kmeans_model = assets['kmeans_model']
    cluster_names = assets['cluster_names']
    full_df = assets['dataframe']

    new_vec = vectorizer.transform([description])
    predicted_cluster_id = kmeans_model.predict(new_vec)[0]
    predicted_category = cluster_names[predicted_cluster_id]

    relevant_df = full_df[full_df['cluster'] == predicted_cluster_id]
    recommended_solutions = []
    
    if not relevant_df.empty:
        relevant_problems_matrix = vectorizer.transform(relevant_df['gangguan'])
        cosine_similarities = cosine_similarity(new_vec, relevant_problems_matrix).flatten()
        top_indices = cosine_similarities.argsort()[::-1][:3]
        solutions = relevant_df.iloc[top_indices]['solusi'].tolist()
        unique_solutions = list(dict.fromkeys(solutions))
        recommended_solutions = unique_solutions

    # Kembalikan kategori, solusi, dan list pilihan kosong
    return predicted_category, recommended_solutions, []

# --- FUNGSI UNTUK MENCARI SOLUSI SPESIFIK BERDASARKAN PILIHAN ---
def get_specific_solutions(original_problem, choice, assets):
    """Mencari solusi yang lebih terfokus setelah user memilih kategori."""
    full_df = assets['dataframe']
    vectorizer = assets['vectorizer']
    
    # Filter dataframe berdasarkan kata kunci dari pilihan user
    search_term = choice.lower()
    filtered_df = full_df[
        full_df['gangguan'].str.contains(search_term, case=False) | 
        full_df['solusi'].str.contains(search_term, case=False)
    ].copy()

    recommended_solutions = []
    if not filtered_df.empty:
        # Hitung kemiripan masalah user dengan data yang sudah difilter
        new_vec = vectorizer.transform([original_problem])
        filtered_problems_matrix = vectorizer.transform(filtered_df['gangguan'])
        cosine_similarities = cosine_similarity(new_vec, filtered_problems_matrix).flatten()
        
        # Ambil 3 solusi teratas
        top_indices = cosine_similarities.argsort()[::-1][:3]
        solutions = filtered_df.iloc[top_indices]['solusi'].tolist()
        recommended_solutions = list(dict.fromkeys(solutions))
        
    return recommended_solutions

# --- ROUTE UNTUK HALAMAN WEB ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    solution_list = []
    choice_list = []
    user_problem = ""
    
    if request.method == 'POST':
        # Cek apakah ini request dari form pilihan atau form awal
        if 'pilihan' in request.form:
            # --- ALUR 2: USER SUDAH MEMILIH KATEGORI ---
            user_problem = request.form.get('masalah_asli', '')
            user_choice = request.form.get('pilihan', '')
            
            prediction_result = f"Kategori Pilihan: {user_choice}"
            solution_list = get_specific_solutions(user_problem, user_choice, model_assets)

        else:
            # --- ALUR 1: USER BARU MENGIRIM MASALAH ---
            user_problem = request.form.get('masalah', '')
            if user_problem.strip():
                prediction_result, solution_list, choice_list = analyse_initial_problem(user_problem, model_assets)
            else:
                prediction_result = "Mohon masukkan deskripsi masalah Anda."
            
    # Kirim semua variabel yang mungkin ke template HTML
    return render_template('index.html', 
                           kategori=prediction_result, 
                           rekomendasi=solution_list,
                           pilihan_list=choice_list,
                           masalah=user_problem)

# --- BLOK EKSEKUSI UTAMA ---
if __name__ == '__main__':
    print("--- Memulai Pelatihan Model Help Desk Cerdas ---")
    model_assets = train_unsupervised_model()
    if model_assets:
        print("--- Pelatihan Model Selesai. Aplikasi Siap Digunakan. ---")
        app.run(debug=True)
    else:
        print("--- Gagal melatih model. Aplikasi tidak dapat dijalankan. ---")

