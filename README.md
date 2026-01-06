# NeuroOCR

Proyecto de OCR con un flujo de inferencia por lote y una interfaz gráfica tipo pizarra para pruebas.

## Requisitos
- Python (recomendado 3.10+)
- Docker + Docker Compose

## Instalación

1) Instalar dependencias de Python:
```bash
pip install -r requirements.txt
```

2) Levantar servicios con Docker:
```bash
docker compose up -d
```

## Ejecución

### Inferencia por lote
Lee todos los archivos dentro de `tests` y guarda los resultados en `out`:
```bash
python -m src.infer
```

### Interfaz gráfica (pizarra)
Abre una GUI para testear el OCR dibujando:
```bash
python -m src.gui
```
