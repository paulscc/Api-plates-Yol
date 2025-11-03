from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import logging
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import torch

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix para PyTorch 2.6+
def patch_torch_load():
    """Parcha torch.load para permitir cargar modelos con PyTorch 2.6+"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, weights_only=None, **kwargs):
        try:
            return original_load(f, map_location=map_location, weights_only=False, **kwargs)
        except Exception as e:
            logger.error(f"Error en patched load: {e}")
            raise
    
    torch.load = patched_load
    logger.info("✅ torch.load parcheado para PyTorch 2.6+")

patch_torch_load()

# Inicializar FastAPI
app = FastAPI(
    title="API Detector de Placas con Blur",
    description="Detecta y aplica blur a placas de vehículos usando YOLO",
    version="1.0.0"
)

# Configuración de Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "processed-images")

# Inicializar cliente Supabase
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("✅ Cliente Supabase inicializado correctamente")
except Exception as e:
    logger.error(f"❌ Error inicializando Supabase: {e}")
    supabase = None

# Cargar modelo YOLO
model = None

def load_model():
    """Carga el modelo YOLO"""
    global model
    try:
        logger.info("Cargando modelo YOLO...")
        model = YOLO('license-plate-finetune-v1s.pt')
        logger.info("✅ Modelo YOLO cargado correctamente")
        logger.info(f"Clases disponibles: {model.names}")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        raise e

def ensure_model_loaded():
    """Asegura que el modelo esté cargado"""
    global model
    if model is None:
        load_model()

def detectar_placas_blur(image_array: np.ndarray, nivel_blur: int = 35):
    """
    Detecta placas y aplica blur limpio (sin cuadros)
    """
    ensure_model_loaded()
    
    image = image_array.copy()
    
    # Asegurar que el kernel sea impar
    if nivel_blur % 2 == 0:
        nivel_blur += 1
    
    placas_detectadas = 0
    
    try:
        results = model(image, conf=0.015, verbose=False)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = box.conf[0].cpu().numpy()
                
                logger.info(f"Placa detectada - Confianza: {conf:.3f}")
                
                # Aplicar blur a la región
                roi = image[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (nivel_blur, nivel_blur), 0)
                image[y1:y2, x1:x2] = blurred_roi
                
                placas_detectadas += 1
    
    except Exception as e:
        logger.error(f"Error en detección: {e}")
        raise e
    
    return image, placas_detectadas

def upload_to_supabase(image: np.ndarray, filename: str, metadata: dict = None):
    """
    Sube imagen a Supabase Storage y guarda metadata en BD
    """
    if supabase is None:
        raise Exception("Cliente Supabase no inicializado")
    
    try:
        # Convertir imagen a bytes
        success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise Exception("Error codificando imagen")
        
        image_bytes = encoded_image.tobytes()
        
        # Subir a Supabase Storage
        logger.info(f"Subiendo {filename} a Supabase Storage...")
        storage_response = supabase.storage.from_(SUPABASE_BUCKET).upload(
            file=image_bytes,
            path=filename,
            file_options={"content-type": "image/jpeg"}
        )
        
        # Obtener URL pública
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(filename)
        logger.info(f"✅ Imagen subida: {public_url}")
        
        # Preparar metadata para la base de datos
        db_metadata = {
            "filename": filename,
            "public_url": public_url,
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_size": len(image_bytes),
            "content_type": "image/jpeg"
        }
        
        # Agregar metadata adicional si se proporciona
        if metadata:
            db_metadata.update(metadata)
        
        # Guardar en la tabla 'processed_images'
        try:
            db_response = supabase.table("processed_images").insert(db_metadata).execute()
            logger.info(f"✅ Metadata guardada en BD: {filename}")
        except Exception as db_error:
            logger.warning(f"⚠️  No se pudo guardar metadata en BD: {db_error}")
        
        return {
            "public_url": public_url,
            "filename": filename,
            "file_size": len(image_bytes)
        }
        
    except Exception as e:
        logger.error(f"❌ Error subiendo a Supabase: {e}")
        raise e

@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API Detector de Placas con Blur",
        "version": "1.0.0",
        "supabase_connected": supabase is not None,
        "endpoints": {
            "health": "/health",
            "detect": "/detect",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado de la API"""
    ensure_model_loaded()
    supabase_status = "connected" if supabase else "disconnected"
    return {
        "status": "healthy",
        "supabase": supabase_status,
        "model_loaded": model is not None,
        "model_classes": model.names if model else None
    }

@app.post("/detect")
async def detect_plates(
    file: UploadFile = File(...),
    blur_level: int = 35
):
    """
    Detecta placas en una imagen y aplica blur
    
    - **file**: Archivo de imagen (JPG, PNG, etc.)
    - **blur_level**: Nivel de blur (por defecto 35, debe ser impar)
    """
    try:
        ensure_model_loaded()
        
        # Validar que sea una imagen
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer imagen
        logger.info(f"Procesando archivo: {file.filename}")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="Error leyendo la imagen")
        
        logger.info(f"Imagen cargada: {original_image.shape}")
        
        # Validar nivel de blur
        if blur_level < 1 or blur_level > 99:
            raise HTTPException(status_code=400, detail="blur_level debe estar entre 1 y 99")
        
        # Detectar y aplicar blur
        imagen_procesada, placas_count = detectar_placas_blur(original_image, blur_level)
        
        # Generar nombre único
        filename = f"plate_detected_{uuid.uuid4().hex}.jpg"
        
        # Subir a Supabase
        upload_result = upload_to_supabase(
            imagen_procesada,
            filename,
            metadata={
                "original_filename": file.filename,
                "placas_detectadas": placas_count,
                "blur_level": blur_level,
                "processing_type": "license_plate_blur"
            }
        )
        
        logger.info(f"✅ Imagen procesada exitosamente: {placas_count} placas detectadas")
        
        return {
            "success": True,
            "filename": filename,
            "public_url": upload_result["public_url"],
            "placas_detectadas": placas_count,
            "blur_level": blur_level,
            "original_filename": file.filename,
            "file_size": upload_result["file_size"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
