
from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255)
    pemrakarsa = models.TextField()
    level_peraturan = models.TextField()
    konten_penimbang = models.TextField()
    peraturan_terkait = models.TextField()
    konten_peraturan = models.TextField()
    kategori_peraturan = models.TextField()
    topik_peraturan = models.TextField()
    struktur_peraturan = models.TextField()

    def __str__(self):
        return self.title
