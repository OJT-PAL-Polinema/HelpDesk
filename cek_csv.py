import pandas as pd

# GANTI DENGAN NAMA FILE ANDA. PASTIKAN ADA EKSTENSI .csv
nama_file = "helpdesk_2025.csv"

try:
    df = pd.read_csv(nama_file)
    print("✅ File CSV berhasil dibaca!")
    print("\nBerikut adalah nama-nama kolom yang ditemukan di dalam file:")
    # Ini akan mencetak daftar nama kolom persis seperti yang dibaca Python
    print(list(df.columns))

except FileNotFoundError:
    print(f"❌ ERROR: File dengan nama '{nama_file}' TIDAK DITEMUKAN.")
    print("SOLUSI: Pastikan nama file sudah benar dan file tersebut ada di folder yang sama dengan skrip ini.")
except Exception as e:
    print(f"❌ Terjadi error lain: {e}")