from django.shortcuts import render, redirect
from .models import Document
from .utils import (
    extract_title, extract_details,
    calculate_similarity, perform_clustering
)
import networkx as nx
from django.http import JsonResponse
import networkx as nx
from .models import Document
from .services import get_documents, delete_document  
import pandas as pd
import numpy as np
from django.http import JsonResponse
import fitz  # PyMuPDF
from nltk.tokenize import word_tokenize

def delete_document_view(request, doc_id):
    if request.method == 'POST':
        delete_document(doc_id)
        if 'similarity_results' in request.session:
            del request.session['similarity_results']
        if 'similarity_matrix' in request.session:
            del request.session['similarity_matrix']
        if 'graph_nodes' in request.session:
            del request.session['graph_nodes']
        if 'graph_edges' in request.session:
            del request.session['graph_edges']
        update_similarity_session(request)
        return redirect('home')
    return redirect('home')

def home(request):
    documents = get_documents()
    if request.method == 'POST' and 'delete' in request.POST:
        doc_id = request.POST['delete']
        delete_document(doc_id)
        return redirect('home')
    return render(request, 'home.html', {'documents': documents})

def view_document(request, doc_id):
    doc = Document.objects.get(id=doc_id)
    return render(request, "view_document.html", {'doc': doc})

def pdf_to_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    word_tokens = word_tokenize(text)
    filtered_tokens = [word for word in word_tokens if word.isalnum() or '-' in word]
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text

def add_document(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        content = pdf_to_text(file)
        title = extract_title(content)
        details = extract_details(content)
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
        update_similarity_session(request)
        return redirect('home')
    return render(request, "add_document.html")

def similarity_graph(request):
    if 'graph_nodes' in request.session and 'graph_edges' in request.session:
        nodes = request.session['graph_nodes']
        edges = request.session['graph_edges']
    else:
        nodes, edges = update_similarity_session(request)[2], []
    # print("nodes and edges:", nodes, edges)
    return JsonResponse({"nodes": nodes, "edges": edges})

def update_similarity_session(request):
    documents = Document.objects.all()
    if len(documents) > 1:
        similarity_results, similarity_matrix = calculate_similarity(documents)
        similarity_matrix = similarity_matrix.astype(float)
        request.session['similarity_results'] = similarity_results
        request.session['similarity_matrix'] = similarity_matrix.tolist()
        G = nx.Graph()
        titles = [doc.title for doc in documents]
        for i, title in enumerate(titles):
            G.add_node(i, label=title)
        threshold = 0.1
        edges = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity_score = similarity_matrix[i, j]
                if similarity_score > threshold:
                    similarity_score = min(similarity_score, 100)
                    normalized_width = similarity_score / 10
                    G.add_edge(i, j, weight=similarity_score)
                    edges.append({
                        "from": i,
                        "to": j,
                        "width": normalized_width,
                        "label": f"{similarity_score:.2f}%",
                    })
        request.session['graph_nodes'] = [{"id": i, "label": title} for i, title in enumerate(titles)]
        request.session['graph_edges'] = edges
        return similarity_results, similarity_matrix, edges
    return [], [], []

def similarity(request):
    similarity_results = request.session.get('similarity_results', None)
    # print("Similarity Results:", similarity_results)
    clustering_ready = bool(similarity_results)
    return render(request, "similarity.html", {'clustering_ready': clustering_ready})

def clustering(request):
    documents = Document.objects.all()
    similarity_matrix = request.session.get('similarity_matrix')
    if not similarity_matrix:
        return redirect('similarity')
    similarity_matrix = np.array(similarity_matrix)
    silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters=2)
    cluster_data = {f"Cluster {i + 1}": [] for i in range(2)}
    for i, label in enumerate(labels):
        cluster_data[f"Cluster {label + 1}"].append(documents[i].title)
    max_cluster_length = max(len(cluster) for cluster in cluster_data.values())
    for key in cluster_data:
        cluster_data[key].extend([''] * (max_cluster_length - len(cluster_data[key])))
    cluster_df = pd.DataFrame(cluster_data)
    context = {
        'clusters': cluster_df.to_html(classes="table table-striped", index=False),
        'silhouette_avg': silhouette_avg
    }
    return render(request, "clustering.html", context)

def similarity_detail(request, i, j):
    similarity_results = request.session.get('similarity_results', [])
    if not similarity_results:
        return render(request, "error.html", {"message": "Similarity results not found in session"})
    detail = next(
        (result["detail_similarity"] for result in similarity_results if result["detail_url"] == f"/similarity_detail/{i}/{j}/"),
        None
    )
    if not detail:
        return render(request, "error.html", {"message": "Detail keterkaitan tidak ditemukan"})
    documents = list(Document.objects.all())
    if i >= len(documents) or j >= len(documents):
        return render(request, "error.html", {"message": "Dokumen tidak ditemukan"})
    total_similarity = sum(detail.values()) / len(detail)
    context = {
        "dokumen1": documents[i].title,
        "dokumen2": documents[j].title,
        "detail_similarity": detail,
        "total_similarity": total_similarity,
    }
    return render(request, "similarity_detail.html", context)
