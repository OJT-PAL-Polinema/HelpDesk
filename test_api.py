import os
import google.generativeai as genai

print("--- Memulai Tes Koneksi API Gemini ---")

try:
    # 1. Baca API Key dari environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Variabel GOOGLE_API_KEY tidak ditemukan. Pastikan Anda sudah menjalankan perintah yang benar di terminal ini.")
    
    print("API Key ditemukan. Mencoba mengkonfigurasi...")
    genai.configure(api_key=api_key)
    
    # 2. Buat model dengan nama yang diperbarui
    # Mengganti 'gemini-pro' dengan model yang lebih baru dan stabil
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    # 3. Kirim prompt sederhana
    print("Mengirim permintaan tes ke Gemini...")
    response = model.generate_content("Hello, world! Balas dalam Bahasa Indonesia.")
    
    # 4. Cetak hasilnya
    print("\n--- HASIL ---")
    print(response.text)
    print("\n✅ Tes koneksi berhasil!")

except Exception as e:
    print(f"\n❌ TES GAGAL: Terjadi error.")
    print(f"Detail Error: {e}")

