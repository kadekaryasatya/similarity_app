# Generated by Django 5.0.6 on 2024-11-12 03:44

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=255)),
                ('pemrakarsa', models.TextField()),
                ('level_peraturan', models.TextField()),
                ('konten_penimbang', models.TextField()),
                ('peraturan_terkait', models.TextField()),
                ('konten_peraturan', models.TextField()),
                ('kategori_peraturan', models.TextField()),
                ('topik_peraturan', models.TextField()),
                ('struktur_peraturan', models.TextField()),
            ],
        ),
    ]
