import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz  
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pdfkit
from sentence_transformers import SentenceTransformer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# case folding, tokenizing
def preprocess_text(text):
    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    # Menghapus token non-alfanumerik
    clean_tokens = [word for word in word_tokens if word.isalnum()]

    # Lowercase dan Menghapus stop words 
    filtered_text = [word.lower() for word in clean_tokens if word.lower()]

    cleaned_text = " ".join(filtered_text)

    return cleaned_text

def extract_title(text):
    title_pattern = r'\b(Peraturan|Undang-Undang)\b.*?(?=\bdengan rahmat\b)'
    title_match = re.search(title_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if title_match:
        title = title_match.group().strip()
    else:
        title = "Judul Tidak Ditemukan"
    
    return title

def extract_details(text):
    pemrakarsa_pattern = r'\b(?:Presiden|Wakil Presiden|Lembaga Pemerintah Non-Kementerian|Dewan Perwakilan Rakyat(?: \(DPR\))?|Majelis Permusyawaratan Rakyat(?: \(MPR\))?|Mahkamah Agung(?: \(MA\))?|Mahkamah Konstitusi(?: \(MK\))?|Badan Pemeriksa Keuangan(?: \(BPK\))?|Bank Indonesia(?: \(BI\))?|Otoritas Jasa Keuangan(?: \(OJK\))?|Badan Pengawas Pemilihan Umum(?: \(Bawaslu\))?|Komisi Pemilihan Umum(?: \(KPU\))?|Komisi Pemberantasan Korupsi(?: \(KPK\))?|Gubernur|Dewan Perwakilan Rakyat Daerah Provinsi(?: \(DPRD Provinsi\))?|Bupati|Walikota|Dewan Perwakilan Rakyat Daerah Kabupaten/Kota(?: \(DPRD Kabupaten/Kota\))?|Kepala Desa|Lurah|Badan Permusyawaratan Desa(?: \(BPD\))?|Komisi Yudisial(?: \(KY\))?|Lembaga Negara Independen|Komnas HAM|Komisi Informasi|Menteri Dalam Negeri|Menteri Luar Negeri|Menteri Pertahanan|Menteri Hukum dan Hak Asasi Manusia|Menteri Keuangan|Menteri Pendidikan dan Kebudayaan|Menteri Riset dan Teknologi|Menteri Agama|Menteri Ketenagakerjaan|Menteri Energi dan Sumber Daya Mineral|Menteri Perindustrian|Menteri Perdagangan|Menteri Pertanian|Menteri Lingkungan Hidup dan Kehutanan|Menteri Kelautan dan Perikanan|Menteri Desa, Pembangunan Daerah Tertinggal, dan Transmigrasi|Menteri Perencanaan Pembangunan Nasional|Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi|Menteri Pekerjaan Umum dan Perumahan Rakyat|Menteri Kesehatan|Menteri Sosial|Menteri Pariwisata|Menteri Komunikasi dan Informatika|Menteri Koordinator Bidang Politik, Hukum, dan Keamanan|Menteri Koordinator Bidang Perekonomian|Menteri Koordinator Bidang Pembangunan Manusia dan Kebudayaan|Menteri Koordinator Bidang Kemaritiman dan Investasi|Menteri Badan Usaha Milik Negara|Menteri Koperasi dan Usaha Kecil dan Menengah|Menteri Pemuda dan Olahraga|Menteri Perhubungan|Menteri Agraria dan Tata Ruang/Badan Pertanahan Nasional|Menteri Perumahan Rakyat|Menteri Percepatan Pembangunan Daerah Tertinggal|Menteri Perencanaan Pembangunan Nasional/Bappenas|Menteri Sekretaris Negara|Menteri Sekretariat Kabinet|Menteri(?: [A-Za-z]+)*|Kementerian(?: [A-Za-z]+)*)\b'
    pemrakarsa_match = re.search(pemrakarsa_pattern, text, re.IGNORECASE)
    pemrakarsa = pemrakarsa_match.group(0).strip() if pemrakarsa_match else "Pemrakarsa tidak ditemukan"
    
    level_peraturan_pattern = r'\b(?:Undang-Undang Dasar 1945|Ketetapan Majelis Permusyawaratan Rakyat|Undang-Undang|Peraturan Pemerintah Pengganti Undang-Undang|Peraturan Pemerintah|Keputusan Presiden|Peraturan Menteri|Peraturan Gubernur|Peraturan Bupati)\b'
    level_peraturan_match = re.search(level_peraturan_pattern, text, re.IGNORECASE)
    if level_peraturan_match:
        level_peraturan = level_peraturan_match.group(0).strip()
        if level_peraturan.lower() == "peraturan gubernur":
            level_peraturan = "peraturan daerah provinsi"
        elif level_peraturan.lower() == "peraturan bupati":
            level_peraturan = "peraturan daerah kabupaten/Kota"
    else:
        level_peraturan = "Level Peraturan tidak ditemukan"

    penimbang_pattern = r'Menimbang\s*(.*?)(?=Mengingat|$)'
    penimbang_match = re.search(penimbang_pattern, text, re.DOTALL| re.IGNORECASE)
    konten_penimbang = penimbang_match.group(1).strip() if penimbang_match else "Penimbang tidak ditemukan"

    peraturan_terkait_pattern = r'Mengingat\s*(.*?)(?=Memutuskan|$)'
    peraturan_terkait_match = re.search(peraturan_terkait_pattern, text, re.DOTALL| re.IGNORECASE)
    peraturan_terkait = peraturan_terkait_match.group(1).strip() if peraturan_terkait_match else "Peraturan Terkait tidak ditemukan"
    
    konten_peraturan_pattern = r'Memutuskan\s*(.*?)$'
    konten_peraturan_match = re.search(konten_peraturan_pattern, text, re.IGNORECASE | re.DOTALL)
    konten_peraturan = (konten_peraturan_match.group(1)[:255]).strip() if konten_peraturan_match else "Konten peraturan tidak ditemukan"

    if level_peraturan == "Level Peraturan tidak ditemukan":
       kategori_peraturan = "peraturan biasa"
    else:
       kategori_peraturan = "peraturan perundang-undangan"
    

    topik_kata_kunci = {
        "pendidikan": ["sekolah", "kurikulum", "pengajaran", "siswa", "guru", "pendidikan tinggi", "universitas", "beasiswa"],
        "kesehatan": ["rumah sakit", "dokter", "obat-obatan", "penyakit menular", "vaksinasi", "pelayanan kesehatan", "asuransi kesehatan"],
        "lingkungan hidup": ["polusi udara", "polusi air", "limbah", "konservasi", "hutan", "energi terbarukan", "pengelolaan sampah"],
        "pertanian": ["tanaman", "peternakan", "lahan pertanian", "irigasi", "pupuk", "pestisida", "perlindungan tanaman"],
        "ketenagakerjaan": ["tenaga kerja", "upah", "keamanan kerja", "hak-hak pekerja", "serikat pekerja", "perlindungan sosial"],
        "perpajakan": ["pajak penghasilan", "pajak pertambahan nilai", "tarif pajak", "penghindaran pajak", "insentif pajak"],
        "investasi": ["pasar modal", "saham", "obligasi", "regulasi investasi", "perlindungan investor", "modal ventura"],
        "transportasi": ["angkutan","jalan", "transportasi umum", "kendaraan bermotor", "bandara", "pelabuhan", "transportasi massal"],
        "keuangan": ["perbankan", "asuransi", "pasar keuangan", "regulasi keuangan", "inflasi", "suku bunga", "kebijakan moneter"]
    }

    frekuensi_topik = {topik: 0 for topik in topik_kata_kunci}

    for topik, kata_kunci in topik_kata_kunci.items():
        for kata in kata_kunci:
            frekuensi_topik[topik] += text.count(kata)

    topik_peraturan_text = max(frekuensi_topik, key=frekuensi_topik.get)

    if frekuensi_topik[topik_peraturan_text] == 0:
        topik_peraturan_text = "Topik tidak ditemukan"
    
    judul =  extract_title(text)

    struktur_peraturan_pattern = {
    "Judul": judul,
    "Pembukaan": r'(Dengan Rahmat Tuhan Yang Maha Esa.*?)(?=Menimbang)',
    "Batang Tubuh": konten_penimbang + "\n\n" + peraturan_terkait + "\n\n" + konten_peraturan,
    "Penjelasan": r'(Penjelasan atas.*?)(?=Ditetapkan Di)',
    "Penutup": r'(?:Ditetapkan Di).*?(?=Lampiran|$)',
    "Lampiran": r'(Lampiran.*?$)'
}

    struktur_peraturan = {}

    for bagian, pola in struktur_peraturan_pattern.items():
        if bagian == "Batang Tubuh":
            struktur_peraturan[bagian] = pola
        else:
            matches = re.findall(pola, text, re.DOTALL | re.IGNORECASE)
            if matches:
                if bagian == "Lampiran":
                    struktur_peraturan[bagian] = matches[-1].strip()
                else:
                    struktur_peraturan[bagian] = matches[0].strip()
            else:
                struktur_peraturan[bagian] = "Tidak ditemukan"

    struktur_peraturan_mix = "\n\n".join([f"{bagian}: {konten}" for bagian, konten in struktur_peraturan.items()])

    return {
        "Pemrakarsa": pemrakarsa,
        "Level Peraturan": level_peraturan,
        "Konten Penimbang": konten_penimbang,
        "Peraturan Terkait": peraturan_terkait,
        "Konten Peraturan": konten_peraturan,
        "Kategori Peraturan": kategori_peraturan,
        "Topik Peraturan": topik_peraturan_text,
        "Struktur Peraturan": struktur_peraturan_mix
    }
  
  
def calculate_similarity(documents):
    # Menggabungkan konten dari setiap dokumen
    combined_contents = [
        {
            'title': doc.title,
            'pemrakarsa': doc.pemrakarsa,
            'level_peraturan': doc.level_peraturan,
            'konten_penimbang': doc.konten_penimbang,
            'peraturan_terkait': doc.peraturan_terkait,
            'konten_peraturan': doc.konten_peraturan,
            'kategori_peraturan': doc.kategori_peraturan,
            'topik_peraturan': doc.topik_peraturan,
            'struktur_peraturan': doc.struktur_peraturan
        }
        for doc in documents
    ]

    # Load pre-trained word embedding model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Encode setiap field menjadi vektor
    field_vectors = {
        field: model.encode([doc[field] for doc in combined_contents])
        for field in combined_contents[0]
    }

    titles = [doc.title for doc in documents]
    similarity_results = []

    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            detail_similarity = {}
            total_similarity = 0

            # Hitung similarity per field
            for field, vectors in field_vectors.items():
                score = cosine_similarity([vectors[i]], [vectors[j]])[0][0] * 100
                detail_similarity[field] = score
                total_similarity += score

            # Rata-rata similarity dari semua field
            average_similarity = total_similarity / len(field_vectors)

            similarity_results.append({
                'dokumen1': titles[i],
                'dokumen2': titles[j],
                'keterkaitan': f"{average_similarity:.2f}%",
                'detail_similarity': detail_similarity,
                'detail_url': f"/similarity_detail/{i}/{j}/"  # URL untuk melihat detail
            })

    return similarity_results


def perform_clustering(similarity_matrix, num_clusters=2):
    # Menggunakan K-Means untuk clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(similarity_matrix)
    
    # Menghitung silhouette score untuk evaluasi clustering
    silhouette_avg = silhouette_score(similarity_matrix, labels)
    
    return silhouette_avg, labels


from django.shortcuts import render
