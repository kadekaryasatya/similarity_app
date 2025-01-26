from django.shortcuts import render, redirect
from .models import Document
from .utils import (
    extract_title, extract_details,
    calculate_similarity, perform_clustering
)

from .services import get_documents, delete_document  # Import dari services.py
import pandas as pd

from django.http import HttpResponse
import base64
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def delete_document_view(request, doc_id):
    if request.method == 'POST':
        # Menggunakan fungsi delete_document dari services.py
        delete_document(doc_id)
        return redirect('home')  # Redirect ke halaman utama setelah penghapusan
    return redirect('home')

def home(request):
    # Mengambil dokumen dari services.py
    documents = get_documents()
    
    # Cek jika ada request POST untuk penghapusan
    if request.method == 'POST' and 'delete' in request.POST:
        doc_id = request.POST['delete']
        delete_document(doc_id)  # Panggil fungsi delete_document
        return redirect('home')  # Redirect setelah hapus dokumen

    return render(request, 'home.html', {'documents': documents})

def view_document(request, doc_id):
    doc = Document.objects.get(id=doc_id)
    return render(request, "view_document.html", {'doc': doc})

import fitz  # PyMuPDF
from nltk.tokenize import word_tokenize

def pdf_to_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    
    text = ""
    for page in doc:
        text += page.get_text()

    # Tokenisasi (bisa disesuaikan dengan kebutuhan Anda)
    word_tokens = word_tokenize(text)
    filtered_tokens = [word for word in word_tokens if word.isalnum() or '-' in word]
    cleaned_text = " ".join(filtered_tokens)

    return cleaned_text

def add_document(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        
        # Menggunakan pdf_to_text untuk membaca isi file PDF
        content = pdf_to_text(file)
        
        # Proses judul dan konten dokumen
        title = extract_title(content)
        details = extract_details(content)  # Implementasikan sesuai dengan aplikasi Anda
        
        # Simpan dokumen ke database
        Document.objects.create(
            title=title,
            pemrakarsa=details.get('Pemrakarsa', ''),
            level_peraturan=details.get('Level Peraturan', ''),
            konten_penimbang=details.get('Konten Penimbang', ''),
            peraturan_terkait=details.get('Peraturan Terkait', ''),
            konten_peraturan=details.get('Konten Peraturan', ''),
            kategori_peraturan=details.get('Kategori Peraturan', ''),
            topik_peraturan=details.get('Topik Peraturan', ''),
            struktur_peraturan=details.get('Struktur Peraturan', '')
        )
        
        return redirect('home')
    
    return render(request, "add_document.html")


def similarity(request):
    documents = Document.objects.all()
    num_samples = len(documents)

    if num_samples < 2:
        return render(request, "similarity.html", {'error': "Diperlukan setidaknya 2 dokumen untuk menghitung keterkaitan."})

    similarity_results = []

    if request.method == 'POST' and 'calculate_similarity' in request.POST:
        # Memanggil fungsi calculate_similarity
        similarity_results = calculate_similarity(documents)

        context = {
            'documents': documents,
            'similarity_results': similarity_results,
            'clustering_ready': True  # Menandakan bahwa similarity telah dihitung
        }
        return render(request, "similarity.html", context)



    if request.method == 'POST' and 'perform_clustering' in request.POST:
        combined_contents = [
            f"{doc.title} {doc.pemrakarsa} {doc.level_peraturan} {doc.konten_penimbang} "
            f"{doc.peraturan_terkait} {doc.konten_peraturan} {doc.kategori_peraturan} "
            f"{doc.topik_peraturan} {doc.struktur_peraturan}" for doc in documents
        ]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_contents)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters=2)

        cluster_data = {f"Cluster {i + 1}": [] for i in range(2)}
        for i, label in enumerate(labels):
            cluster_data[f"Cluster {label + 1}"].append(documents[i].title)

        max_cluster_length = max(len(cluster) for cluster in cluster_data.values())
        for key in cluster_data:
            cluster_data[key].extend([''] * (max_cluster_length - len(cluster_data[key])))

        cluster_df = pd.DataFrame(cluster_data)

        context = {
            'documents': documents,
            'clusters': cluster_df.to_html(classes="table-simialrity table-striped", index=False),
            'silhouette_avg': silhouette_avg,
            'clustering_ready': True
        }
        return render(request, "similarity.html", context)

    return render(request, "similarity.html", {'documents': documents})


def similarity_detail(request, i, j):
    documents = Document.objects.all()
    if i >= len(documents) or j >= len(documents):
        return ("Dokumen tidak ditemukan")

    # Menggunakan calculate_similarity untuk mendapatkan detail similarity
    similarity_results = calculate_similarity(documents)
    detail = next(
        (result['detail_similarity'] for result in similarity_results if result['detail_url'] == f"/similarity_detail/{i}/{j}/"),
        None
    )

    if not detail:
        return ("Detail keterkaitan tidak ditemukan")

    context = {
        'dokumen1': documents[i].title,
        'dokumen2': documents[j].title,
        'detail_similarity': detail
    }
    return render(request, "similarity_detail.html", context)


