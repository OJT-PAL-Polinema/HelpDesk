import os
import google.generativeai as genai

print("--- Memeriksa model AI yang tersedia untuk Anda ---")

try:
    # 1. Baca API Key dari environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Variabel GOOGLE_API_KEY tidak ditemukan. Pastikan Anda sudah menjalankan perintah yang benar di terminal ini.")
    
    print("API Key ditemukan. Mengkonfigurasi...")
    genai.configure(api_key=api_key)
    
    print("\nBerikut adalah daftar model yang bisa Anda gunakan:")
    
    # 2. Loop melalui semua model yang tersedia
    available_models = []
    for m in genai.list_models():
      # 3. Filter hanya model yang mendukung metode 'generateContent'
      if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")
        available_models.append(m.name)
        
    if not available_models:
        print("\nTidak ada model yang ditemukan. Ini aneh, coba periksa kembali kunci API Anda.")
    else:
        print(f"\n✅ SARAN: Coba gunakan salah satu model di atas, misalnya '{available_models[0]}', di dalam file app.py dan test_api.py Anda.")


except Exception as e:
    print(f"\n❌ GAGAL: Terjadi error.")
    print(f"Detail Error: {e}")
