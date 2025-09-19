import os
import glob
from flask import Flask, request, jsonify
import numpy as np
from deepface import DeepFace
from PIL import Image
import shutil  # For potential file operations if needed

app = Flask(__name__)

# Base directory for user images
IMAGES_BASE_DIR = "images"

class SimpleFacerec:
    def __init__(self, model_name="ArcFace"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.model_name = model_name

    def _safe_path(self, img_path):
        if img_path.lower().endswith(".webp"):
            temp_path = img_path.replace(".webp", "_temp.jpg")
            Image.open(img_path).convert("RGB").save(temp_path)
            return temp_path
        return img_path

    def load_encoding_images(self, images_path):
        # Clear previous encodings
        self.known_face_encodings = []
        self.known_face_names = []
        
        images = glob.glob(os.path.join(images_path, "*"))
        print(f"{len(images)} images found in {images_path}")

        for img_path in images:
            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            try:
                safe_img = self._safe_path(img_path)
                embedding = DeepFace.represent(
                    img_path=safe_img,
                    model_name=self.model_name,
                    enforce_detection=False
                )[0]["embedding"]

                embedding = np.array(embedding) 

                self.known_face_encodings.append(embedding)
                self.known_face_names.append(filename)
                print(f"[INFO] Loaded encoding for {filename}")
                
                # Clean up temp file if created
                if safe_img != img_path:
                    os.remove(safe_img)
            except Exception as e:
                print(f"[WARNING] Could not encode {filename}: {e}")

    def match_image(self, query_img_path, tolerance=0.45):
        try:
            safe_img = self._safe_path(query_img_path)
            embedding = DeepFace.represent(
                img_path=safe_img,
                model_name=self.model_name,
                enforce_detection=False
            )[0]["embedding"]

            query_encoding = np.array(embedding)
            
            # Clean up temp file if created
            if safe_img != query_img_path:
                os.remove(safe_img)

        except Exception as e:
            print(f"[ERROR] Could not process query image: {e}")
            return "Unknown"

        if not self.known_face_encodings:
            print("[ERROR] No known faces loaded")
            return "Unknown"

        similarities = [
            np.dot(query_encoding, known) / (
                np.linalg.norm(query_encoding) * np.linalg.norm(known)
            )
            for known in self.known_face_encodings
        ]

        best_match_index = np.argmax(similarities)
        best_score = similarities[best_match_index]

        print(f"[DEBUG] Similarities: {similarities}")
        print(f"[DEBUG] Best score: {best_score:.4f}")

        if best_score >= (1 - tolerance):
            return self.known_face_names[best_match_index]
        else:
            return "Unknown"

# Helper function to ensure user directory exists
def ensure_user_dir(username):
    user_dir = os.path.join(IMAGES_BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

# Endpoint to add a person (store image)
@app.route('/add_person', methods=['POST'])
def add_person():
    username = request.form.get('username')
    person_name = request.form.get('person_name')  # The name to save as filename (e.g., "Alice")
    if not username or not person_name:
        return jsonify({"error": "Username and person_name required"}), 400
    
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "Image required"}), 400
    
    # Ensure user directory exists
    user_dir = ensure_user_dir(username)
    
    # Save image as {person_name}.jpg in user folder
    image_path = os.path.join(user_dir, f"{person_name}.jpg")
    
    # Handle if it's webp or other, convert to jpg
    if image.filename.lower().endswith(".webp"):
        pil_image = Image.open(image.stream).convert("RGB")
        pil_image.save(image_path)
    else:
        image.save(image_path)
    
    # Optionally, you can trigger reloading encodings here, but better to load on-demand in recognize
    return jsonify({
        "success": True, 
        "message": f"Image for {person_name} added to {username}'s folder",
        "image_path": image_path
    })

# Endpoint to delete a person (remove image)
@app.route('/delete_person', methods=['DELETE'])
def delete_person():
    username = request.form.get('username')  # For DELETE, use form data or JSON
    person_name = request.form.get('person_name')
    if not username or not person_name:
        return jsonify({"error": "Username and person_name required"}), 400
    
    user_dir = os.path.join(IMAGES_BASE_DIR, username)
    if not os.path.exists(user_dir):
        return jsonify({"error": "User folder not found"}), 404
    
    image_path = os.path.join(user_dir, f"{person_name}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)
        return jsonify({
            "success": True, 
            "message": f"Image for {person_name} deleted from {username}'s folder"
        })
    else:
        return jsonify({"error": "Image not found"}), 404

# Endpoint for recognition (detection)
@app.route('/recognize', methods=['POST'])
def recognize():
    username = request.form.get('username')
    if not username:
        return jsonify({"error": "Username required"}), 400
    
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "Image required"}), 400

    # Ensure user directory exists (though it should for recognition)
    user_dir = ensure_user_dir(username)
    
    # Save temp query image
    temp_image_path = f"temp_{username}.jpg"
    if image.filename.lower().endswith(".webp"):
        pil_image = Image.open(image.stream).convert("RGB")
        pil_image.save(temp_image_path)
    else:
        image.save(temp_image_path)

    # Load encodings from user's folder
    sfr = SimpleFacerec()
    sfr.load_encoding_images(user_dir)
    
    # Match
    match = sfr.match_image(temp_image_path, tolerance=0.45)
    
    # Clean up temp
    os.remove(temp_image_path)
    
    if match == "Unknown":
        return jsonify({
            "match": match, 
            "message": "No relation found",
            "confidence": None  # Could add confidence if needed
        })
    else:
        return jsonify({
            "match": match, 
            "message": f"Matched {match}",
            "confidence": 0.85  # Placeholder; calculate actual best_score if needed
        })

if __name__ == '__main__':
    # Ensure base dir exists
    os.makedirs(IMAGES_BASE_DIR, exist_ok=True)
    app.run(debug=True)