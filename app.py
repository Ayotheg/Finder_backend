# finder_backend.py
import os
import io
import base64
import traceback
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import requests

load_dotenv()

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__, static_folder="public")
CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
ALLOWED = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED


# ---------------------------
# Roboflow Workflow Config
# ---------------------------
RF_KEY = os.getenv("ROBOFLOW_API_KEY")
RF_WS = os.getenv("ROBOFLOW_WORKSPACE", "the-bunker")
RF_WFID = os.getenv("ROBOFLOW_WORKFLOW_ID", "sam3-with-prompts")

WORKFLOW_URL = f"https://serverless.roboflow.com/{RF_WS}/workflows/{RF_WFID}"

print("\n============================")
print("✓ Using Roboflow Workflow URL:", WORKFLOW_URL)
print("============================\n")


# ---------------------------
# Image compression
# ---------------------------
def compress(path, max_dim=1920, quality=85):
    try:
        img = Image.open(path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return buf.read()
    except:
        return open(path, "rb").read()


# ---------------------------
# Health route
# ---------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "workflow": RF_WFID})


# ---------------------------
# Main analyze route
# ---------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
    uploaded_file_path = None
    try:
        # -----------------------
        # File validation
        # -----------------------
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # -----------------------
        # Prompt handling
        # -----------------------
        prompt = (request.form.get("prompt") or "").strip()
        if not prompt:
            prompt = "all objects"

        # -----------------------
        # Save + compress
        # -----------------------
        fname = secure_filename(file.filename)
        uploaded_file_path = os.path.join(UPLOAD_FOLDER, fname)
        file.save(uploaded_file_path)
        img_bytes = compress(uploaded_file_path)
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        # -----------------------
        # Roboflow payload
        # -----------------------
        payload = {
            "inputs": {
                "image": b64_img,
                "text_prompt": prompt
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {RF_KEY}"   # ✅ send API key in header
        }

        # -----------------------
        # Call Roboflow
        # -----------------------
        res = requests.post(WORKFLOW_URL, json=payload, headers=headers, timeout=60)

        if not res.ok:
            return jsonify({
                "error": "Roboflow request failed",
                "status": res.status_code,
                "details": res.text
            }), res.status_code

        data = res.json()
        outputs = data.get("outputs", [])
        annotated = None
        detections = []

        for block in outputs:
            if not isinstance(block, dict):
                continue

            if "mask_visualization" in block:
                mv = block["mask_visualization"]
                if isinstance(mv, dict):
                    mv = mv.get("value") or mv.get("image")
                if isinstance(mv, str):
                    annotated = mv

            if "predictions" in block:
                preds = block["predictions"]
                if isinstance(preds, list):
                    for p in preds:
                        detections.append({
                            "class": p.get("class") or p.get("class_name"),
                            "confidence": p.get("confidence", 0)
                        })

        return jsonify({
            "message": "Detection successful",
            "annotated_image": annotated,
            "detections": detections,
            "total_detections": len(detections)
        })

    except Exception as e:
        return jsonify({
            "error": "Server crashed",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500

    finally:
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            os.remove(uploaded_file_path)


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"✓ Backend running on port {port}")
    app.run(host="0.0.0.0", port=port)
