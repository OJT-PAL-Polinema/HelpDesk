import pandas as pd
from flask import Flask, request, render_template, g
from sklearn.feature_extraction.text import TfidfVectorizer
# --- IMPORT BARU UNTUK CLUSTERING ---
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import glob

app = Flask(__name__)

# --- KONFIGURASI PENTING ---
# Tentukan berapa banyak kategori otomatis yang ingin dibuat oleh mesin.
# Anda bisa bereksperimen dengan angka ini (misalnya 4, 5, atau 6).
K_CLUSTERS = 5

def train_model():
    """Melatih model untuk membuat kategori otomatis (clustering) dari data."""
    path = 'training_data'
    all_files = glob.glob(os.path.join(path, "*.csv"))
    if not all_files:
        print(f"ERROR: Tidak ada file CSV di folder '{path}'!")
        return None

    try:
        df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        print(f"Berhasil menggabungkan {len(all_files)} file CSV.")
    except Exception as e:
        print(f"Error saat membaca file CSV: {e}")
        return None
    
    # Sekarang kita hanya butuh 'gangguan' dan 'solusi'
    df.dropna(subset=['gangguan', 'solusi'], inplace=True)
    if 'gangguan' not in df.columns or 'solusi' not in df.columns:
        print("ERROR: Pastikan semua file CSV memiliki kolom 'gangguan' dan 'solusi'")
        return None

    X_problems = df['gangguan']
    
    # 1. Ubah teks menjadi vektor angka (tetap sama)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=['yang', 'di', 'dan', 'ke', 'ini', 'itu'])
    tfidf_matrix = vectorizer.fit_transform(X_problems)

    # 2. Latih model K-Means untuk membuat grup
    print(f"Memulai training model K-Means dengan {K_CLUSTERS} klaster...")
    kmeans_model = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    kmeans_model.fit(tfidf_matrix)
    print("Model K-Means berhasil dilatih.")

    # Tambahkan label klaster ke dataframe untuk filtering nanti
    df['cluster'] = kmeans_model.labels_
    
    model_assets = {
        'kmeans_model': kmeans_model,
        'vectorizer': vectorizer,
        'dataframe': df
    }
    
    return model_assets

def get_model():
    if 'model_assets' not in g:
        g.model_assets = train_model()
    return g.model_assets

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        model_assets = get_model()
        if model_assets is None:
            return "Error: Model tidak dapat dilatih. Periksa file CSV.", 500
            
        masalah_deskripsi = request.form['masalah']
        if not masalah_deskripsi:
            return render_template('index.html', error="Deskripsi masalah tidak boleh kosong!")
        
        kmeans_model = model_assets['kmeans_model']
        vectorizer = model_assets['vectorizer']
        full_df = model_assets['dataframe']
        
        # 1. Ubah masalah baru menjadi vektor
        new_problem_vector = vectorizer.transform([masalah_deskripsi])
        
        # 2. Prediksi nomor klaster untuk masalah baru
        predicted_cluster = kmeans_model.predict(new_problem_vector)[0]
        
        # 3. Filter DataFrame berdasarkan klaster yang diprediksi
        relevant_df = full_df[full_df['cluster'] == predicted_cluster]
        
        recommended_solutions_list = []
        fallback_message = ""
        
        if not relevant_df.empty:
            # 4. Cari kemiripan di dalam klaster yang relevan
            relevant_problems_matrix = vectorizer.transform(relevant_df['gangguan'])
            cosine_similarities = cosine_similarity(new_problem_vector, relevant_problems_matrix).flatten()
            
            CONFIDENCE_THRESHOLD = 0.05
            relevant_indices = np.where(cosine_similarities > CONFIDENCE_THRESHOLD)[0]
            
            if relevant_indices.size > 0:
                solutions = relevant_df.iloc[relevant_indices]['solusi'].tolist()
                unique_solutions = list(dict.fromkeys(solutions))
                recommended_solutions_list = unique_solutions
        
        if not recommended_solutions_list:
            fallback_message = "Maaf, tidak ditemukan solusi yang relevan di data historis."

        return render_template(
            'result.html', 
            masalah=masalah_deskripsi, 
            kategori=f"Klaster {predicted_cluster} (Dibuat Otomatis)",
            rekomendasi_list=recommended_solutions_list,
            fallback=fallback_message
        )
        
    return render_template('index.html')

if __name__ == '__main__':
    with app.app_context():
        get_model() 
    app.run(debug=True)

