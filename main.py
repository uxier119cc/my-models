# 4. FastAPI application
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import os
import gc
import re
import shutil
import threading
import time
import uuid
import numpy as np
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import transformers
from transformers import BlipProcessor, BlipForConditionalGeneration

blip_processor = None
blip_model = None
model_ready = False
device = "cpu"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global blip_processor, blip_model, model_ready, device
    logger.info("Loading BLIP model...")
    device = "cpu"
    try:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32
        )
        blip_model.to(device)
        blip_model.eval()
        logger.info("BLIP model loaded successfully!")
        model_ready = True
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        model_ready = False
    yield
    logger.info("Shutting down...")
    gc.collect()

app = FastAPI(lifespan=lifespan, title="Pet Match API", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_pet_description(image_path: str):
    """Generate pet description using BLIP model with improved parameters"""
    global device
    try:
        with Image.open(image_path).convert("RGB") as img:
            # Use optimized prompts for pet detection
            prompts = [
                "a photo of a",
                "this is a",
                "image of a",
                "photograph of a",
                "picture of a"
            ]
            
            best_description = ""
            for prompt in prompts:
                try:
                    inputs = blip_processor(img, text=prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = blip_model.generate(
                            **inputs, 
                            max_length=30,  # Shorter to avoid repetition
                            num_beams=2,    # Fewer beams for faster generation
                            temperature=0.8,
                            do_sample=True,
                            no_repeat_ngram_size=2,
                            early_stopping=True
                        )
                    description = blip_processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean the description
                    description = clean_description(description, prompt)
                    
                    # Accept any description that's not empty and not just repeating
                    if description and len(description) > 5 and not has_excessive_repetition(description):
                        best_description = description
                        break  # Use the first good description
                        
                except Exception as e:
                    continue
            
            # If no description found, use a fallback with different parameters
            if not best_description:
                try:
                    inputs = blip_processor(img, return_tensors="pt").to(device)  # No prompt
                    with torch.no_grad():
                        outputs = blip_model.generate(
                            **inputs, 
                            max_length=20,
                            num_beams=1,  # Greedy decoding
                            do_sample=False,
                            early_stopping=True
                        )
                    best_description = blip_processor.decode(outputs[0], skip_special_tokens=True)
                    if not best_description:
                        best_description = "a domestic animal"
                except Exception as e:
                    best_description = "a pet"
            
            # Extract features from the description
            species, breed, color = extract_pet_features(best_description)
            
            return {
                "description": best_description,
                "species": species,
                "breed": breed,
                "color": color
            }
            
    except Exception as e:
        logger.error(f"Error in generate_pet_description: {str(e)}")
        return {
            "description": "a domestic animal",
            "species": "unknown",
            "breed": "unknown",
            "color": "unknown"
        }

def clean_description(description: str, prompt: str) -> str:
    """Clean up the generated description"""
    if not description:
        return ""
    
    # Remove the prompt if it appears at the start
    prompt_lower = prompt.lower()
    desc_lower = description.lower()
    
    if desc_lower.startswith(prompt_lower):
        description = description[len(prompt):].strip()
    
    # Remove any special tokens or unusual characters
    description = re.sub(r'<[^>]+>', '', description)  # Remove HTML-like tags
    description = re.sub(r'[^\w\s.,!?]', '', description)  # Keep only text characters
    
    # Capitalize first letter
    if description:
        description = description[0].upper() + description[1:]
    
    return description.strip()

def has_excessive_repetition(text: str, max_repeats: int = 2) -> bool:
    """Check if text has excessive word repetition"""
    words = text.lower().split()
    if len(words) <= 3:
        return False
    
    # Check for consecutive repetition
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return True
    
    # Check for overall repetition
    word_counts = {}
    for word in words:
        if len(word) > 2:  # Only consider words longer than 2 characters
            word_counts[word] = word_counts.get(word, 0) + 1
            if word_counts[word] > max_repeats:
                return True
    return False

def extract_pet_features(description: str):
    """Extract species, breed, and color from description with improved logic"""
    desc_lower = description.lower()
    
    # Species detection - more comprehensive
    species = "unknown"
    
    # Dog indicators
    dog_indicators = ["dog", "puppy", "pup", "canine", "retriever", "shepherd", "terrier", 
                      "labrador", "bulldog", "beagle", "husky", "poodle", "chihuahua", 
                      "dachshund", "boxer", "rottweiler", "greyhound", "collie", "maltese"]
    
    # Cat indicators  
    cat_indicators = ["cat", "kitten", "kitty", "feline", "tabby", "siamese", "persian",
                      "maine coon", "bengal", "ragdoll", "sphynx", "russian blue", "scottish fold"]
    
    if any(indicator in desc_lower for indicator in dog_indicators):
        species = "dog"
    elif any(indicator in desc_lower for indicator in cat_indicators):
        species = "cat"
    
    # Breed detection
    breed = "unknown"
    breed_keywords = {
        "golden retriever": ["golden retriever", "golden", "retriever"],
        "labrador retriever": ["labrador", "lab", "lab retriever"],
        "german shepherd": ["german shepherd", "alsatian", "shepherd"],
        "bulldog": ["bulldog", "bull dog"],
        "poodle": ["poodle"],
        "beagle": ["beagle"],
        "chihuahua": ["chihuahua"],
        "siberian husky": ["husky", "siberian husky"],
        "dachshund": ["dachshund"],
        "pug": ["pug"],
        "rottweiler": ["rottweiler"],
        "boxer": ["boxer"],
        "yorkshire terrier": ["yorkshire", "yorkie"],
        "siamese": ["siamese"],
        "persian": ["persian"],
        "maine coon": ["maine coon"],
        "bengal": ["bengal"],
        "ragdoll": ["ragdoll"],
        "sphynx": ["sphynx", "hairless"],
        "british shorthair": ["british shorthair"],
        "scottish fold": ["scottish fold"],
        "russian blue": ["russian blue"],
        "tabby": ["tabby"],
        "tuxedo": ["tuxedo"]
    }
    
    for breed_name, keywords in breed_keywords.items():
        if any(keyword in desc_lower for keyword in keywords):
            breed = breed_name
            break
    
    # Color detection
    color = "unknown"
    colors = [
        "black", "white", "brown", "gray", "grey", "orange", "yellow", 
        "golden", "silver", "cream", "tan", "beige", "chocolate", "fawn",
        "red", "blue", "green", "brindle", "spotted", "striped"
    ]
    for c in colors:
        if c in desc_lower:
            color = c
            break
    
    return species, breed, color

def calculate_similarity(img1_path, img2_path):
    """Calculate image similarity using color histograms"""
    try:
        with Image.open(img1_path).convert('RGB') as img1, Image.open(img2_path).convert('RGB') as img2:
            img1 = img1.resize((200, 200))
            img2 = img2.resize((200, 200))
            
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Calculate histograms for each channel
            hist1_r = np.histogram(arr1[:,:,0], bins=32, range=(0, 255))[0]
            hist1_g = np.histogram(arr1[:,:,1], bins=32, range=(0, 255))[0]
            hist1_b = np.histogram(arr1[:,:,2], bins=32, range=(0, 255))[0]
            
            hist2_r = np.histogram(arr2[:,:,0], bins=32, range=(0, 255))[0]
            hist2_g = np.histogram(arr2[:,:,1], bins=32, range=(0, 255))[0]
            hist2_b = np.histogram(arr2[:,:,2], bins=32, range=(0, 255))[0]
            
            # Normalize
            hist1_r = hist1_r / hist1_r.sum()
            hist1_g = hist1_g / hist1_g.sum()
            hist1_b = hist1_b / hist1_b.sum()
            
            hist2_r = hist2_r / hist2_r.sum()
            hist2_g = hist2_g / hist2_g.sum()
            hist2_b = hist2_b / hist2_b.sum()
            
            # Calculate similarities
            similarities = []
            for h1, h2 in [(hist1_r, hist2_r), (hist1_g, hist2_g), (hist1_b, hist2_b)]:
                corr = np.corrcoef(h1, h2)[0, 1]
                if not np.isnan(corr):
                    similarities.append(corr)
            
            if not similarities:
                return 0.0
                
            avg_similarity = np.mean(similarities)
            similarity_percentage = max(0, min((avg_similarity + 1) / 2 * 100, 100))
            
            return similarity_percentage
            
    except Exception as e:
        logger.error(f"Error in calculate_similarity: {str(e)}")
        return 0.0

def safe_filename(filename: str) -> str:
    """Generate a safe filename"""
    base_name = os.path.basename(filename)
    name, ext = os.path.splitext(base_name)
    safe_name = re.sub(r'[^\w\.-]', '_', name)
    return f"{safe_name}_{uuid.uuid4().hex[:8]}{ext}"

@app.post("/generate-description")
async def generate_description(
    image: UploadFile = File(..., description="Pet image file"),
    owner_breed: str = Form(..., description="Expected breed from owner")
):
    """Generate description for a single pet image"""
    temp_path = None
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not owner_breed.strip():
            raise HTTPException(status_code=400, detail="Owner breed is required")
        
        if not model_ready:
            raise HTTPException(status_code=503, detail="Model is not ready yet")

        safe_name = safe_filename(image.filename)
        temp_path = f"temp_{safe_name}"
        
        with open(temp_path, "wb") as f:
            content = await image.read()
            f.write(content)
            
        result = generate_pet_description(temp_path)
        
        # Improved matching logic
        normalized_owner_breed = re.sub(r'\s+', ' ', owner_breed.strip().lower())
        normalized_detected_breed = re.sub(r'\s+', ' ', result['breed'].lower())
        description_text = result['description'].lower()
        
        # Multiple matching strategies
        exact_match = (normalized_owner_breed == normalized_detected_breed)
        partial_match = (normalized_owner_breed in normalized_detected_breed or 
                         normalized_detected_breed in normalized_owner_breed or
                         normalized_owner_breed in description_text)
        
        # Word-based matching for cases like "golden" vs "golden retriever"
        owner_words = set(normalized_owner_breed.split())
        detected_words = set(normalized_detected_breed.split())
        description_words = set(description_text.split())
        
        word_match = len(owner_words.intersection(detected_words)) > 0 or \
                     len(owner_words.intersection(description_words)) > 0
        
        match = exact_match or partial_match or word_match
        
        return {
            "success": True, 
            "description": result['description'],
            "detected_species": result['species'],
            "detected_breed": result['breed'],
            "detected_color": result['color'],
            "match": match,
            "match_type": "exact" if exact_match else "partial" if partial_match else "word" if word_match else "none"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_description: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.post("/match-and-describe")
async def match_and_describe(
    owner_image: UploadFile = File(..., description="Owner's pet image"),
    finder_image: UploadFile = File(..., description="Found pet image"), 
    owner_breed: str = Form(None, description="Optional: Owner's pet breed")
):
    """Compare two pet images and generate descriptions"""
    owner_temp_path = None
    finder_temp_path = None
    
    try:
        if not owner_image.content_type.startswith('image/') or not finder_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
        
        if not model_ready:
            raise HTTPException(status_code=503, detail="Model is not ready yet")

        owner_safe_name = safe_filename(owner_image.filename)
        finder_safe_name = safe_filename(finder_image.filename)
        owner_temp_path = f"temp_owner_{owner_safe_name}"
        finder_temp_path = f"temp_finder_{finder_safe_name}"

        # Save images
        with open(owner_temp_path, "wb") as f:
            f.write(await owner_image.read())
        with open(finder_temp_path, "wb") as f:
            f.write(await finder_image.read())

        # Calculate similarity
        similarity_percentage = calculate_similarity(owner_temp_path, finder_temp_path)

        # Generate descriptions
        owner_desc = generate_pet_description(owner_temp_path)
        finder_desc = generate_pet_description(finder_temp_path)

        # Enhanced matching logic
        match = False
        match_reason = "no match"
        match_type = "none"
        
        if owner_breed:
            normalized_owner_breed = re.sub(r'\s+', ' ', owner_breed.strip().lower())
            finder_desc_lower = finder_desc['description'].lower()
            finder_breed_lower = finder_desc['breed'].lower()
            
            # Multiple matching strategies
            if normalized_owner_breed == finder_breed_lower:
                match = True
                match_reason = "exact breed match"
                match_type = "breed_exact"
            elif normalized_owner_breed in finder_breed_lower or finder_breed_lower in normalized_owner_breed:
                match = True
                match_reason = "partial breed match"
                match_type = "breed_partial"
            elif normalized_owner_breed in finder_desc_lower:
                match = True
                match_reason = "breed mentioned in description"
                match_type = "breed_description"
            else:
                # Word-based matching
                owner_words = set(normalized_owner_breed.split())
                finder_words = set(finder_desc_lower.split())
                if owner_words.intersection(finder_words):
                    match = True
                    match_reason = "keyword match in description"
                    match_type = "keyword"
        else:
            # Match based on multiple factors
            species_match = owner_desc['species'] == finder_desc['species'] and owner_desc['species'] != "unknown"
            breed_match = owner_desc['breed'] == finder_desc['breed'] and owner_desc['breed'] != "unknown"
            color_match = owner_desc['color'] == finder_desc['color'] and owner_desc['color'] != "unknown"
            
            if species_match and breed_match:
                match = True
                match_reason = "species and breed match"
                match_type = "species_breed"
            elif species_match and similarity_percentage > 60:
                match = True
                match_reason = "species match with good similarity"
                match_type = "species_similarity"
            elif species_match and color_match:
                match = True
                match_reason = "species and color match"
                match_type = "species_color"

        return {
            "success": True,
            "finder_description": finder_desc['description'],
            "owner_description": owner_desc['description'],
            "match": match,
            "match_reason": match_reason,
            "match_type": match_type,
            "similarity_percentage": round(similarity_percentage, 2),
            "species": finder_desc['species'],
            "breed": finder_desc['breed'],
            "color": finder_desc['color'],
            "species_match": owner_desc['species'] == finder_desc['species'],
            "breed_match": owner_desc['breed'] == finder_desc['breed'],
            "color_match": owner_desc['color'] == finder_desc['color']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in match_and_describe: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        for path in [owner_temp_path, finder_temp_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model_ready else "loading",
        "model_ready": model_ready,
        "device": device
    }

@app.get("/")
async def root():
    return {
        "message": "Pet Match API v2.0 - Fixed Descriptions",
        "status": "operational" if model_ready else "initializing",
        "features": ["Reliable descriptions", "Better feature extraction", "Improved matching"]
    }



if __name__ == "__main__":
    import uvicorn
    # This print statement helps confirm execution started
    print("Loading BLIP model and starting Uvicorn server...") 
    # The lifespan context manager will handle model loading before accepting requests
    # Changed host to 0.0.0.0 to allow connections from physical devices on the network
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")