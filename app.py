# finder_backend.py
import os
import io
import json
import base64
import traceback
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Load environment variables from .env into os.environ
load_dotenv()  # <-- ensures ROBOFLOW_API_KEY (and others) are loaded

# --- App config ---
app = Flask(__name__, static_folder="public")

CORS(app, resources={r"/api/*": {"origins": "*"}})

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# --- Roboflow config (loaded from env) ---
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "the-bunker")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "sam3-with-prompts")
ROBOFLOW_BASE = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")

ROBOFLOW_WORKFLOW_URL = f"{ROBOFLOW_BASE}/{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW_ID}"

# Startup checks
if not ROBOFLOW_API_KEY:
    print("⚠️  WARNING: ROBOFLOW_API_KEY not set in environment (check your .env)")
else:
    print(f"✓ Roboflow API Key configured (ending with: ...{ROBOFLOW_API_KEY[-4:]})")
print(f"✓ Roboflow workflow URL: {ROBOFLOW_WORKFLOW_URL}")

# --- Helpers ---
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compress_image(file_path: str, quality: int = 85, max_dim: int = 1920) -> bytes:
    """Open, resize if huge, convert to JPEG and return bytes."""
    try:
        img = Image.open(file_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        # Resize if too large
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"  Image resized to: {new_size}")
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=quality, optimize=True)
        output.seek(0)
        data = output.getvalue()
        print(f"  Compressed size: {len(data) / 1024:.2f} KB")
        return data
    except Exception as e:
        print(f"  Image compression failed: {e}; falling back to raw bytes")
        with open(file_path, "rb") as f:
            return f.read()


# --- Routes ---
@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    uploaded_file_path = None
    try:
        print("\n" + "=" * 50)
        print("NEW REQUEST RECEIVED")
        print("=" * 50)

        # 1️⃣ Validate file
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return jsonify(
                {"error": "Invalid file type. Allowed: PNG, JPG, JPEG, GIF"}
            ), 400

        # 2️⃣ Get prompt
        prompt = (request.form.get("prompt") or "").strip()
        if not prompt:
            print("⚠️  No prompt provided - using default 'all objects'")
            prompt = "all objects"

        # 3️⃣ Save temporarily
        filename = secure_filename(file.filename)
        uploaded_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(uploaded_file_path)
        print(f"📁 File saved: {filename}")

        # 4️⃣ Compress & encode image as PNG (preferred for mask visualizations)
        try:
            img = Image.open(uploaded_file_path)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            # Resize if too large
            max_dim = 1920
            if max(img.size) > max_dim:
                ratio = max_dim / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"  Image resized to: {new_size}")

            output = io.BytesIO()
            img.save(output, format="PNG", optimize=True)
            output.seek(0)
            encoded_image = base64.b64encode(output.getvalue()).decode("utf-8")
            print(f"✓ Image encoded as PNG (base64 length: {len(encoded_image)} chars)")
        except Exception as e:
            print(f"⚠️ Image processing failed: {e}; using raw bytes")
            with open(uploaded_file_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")

        # 5️⃣ Build Roboflow payload (matches SAM3 expected schema)
        payload = {
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {
        "image": encoded_image,       # base64 image
        "text_prompt": prompt         # string prompt
        }
        }

        print(f"🚀 Sending request to Roboflow workflow: {ROBOFLOW_WORKFLOW_URL}")
        roboflow_resp = requests.post(
            ROBOFLOW_WORKFLOW_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        print(f"📥 Roboflow response status: {roboflow_resp.status_code}")

        if not roboflow_resp.ok:
            try:
                err_json = roboflow_resp.json()
            except Exception:
                err_json = roboflow_resp.text
            print("❌ ROBOFLOW ERROR:", err_json)
            return (
                jsonify(
                    {
                        "error": f"Roboflow API error: {roboflow_resp.status_code}",
                        "details": err_json,
                        "message": "Workflow may require 'input_image' and 'text_prompt'.",
                    }
                ),
                500,
            )

        # 6️⃣ Parse Roboflow response
        roboflow_data = roboflow_resp.json()
        outputs = roboflow_data.get("outputs", [])

        if not outputs:
            print("❌ No outputs from Roboflow workflow")
            return jsonify({"error": "No outputs from Roboflow", "raw": roboflow_data}), 500

        # 7️⃣ Extract predictions & visualization
        all_predictions = []
        annotated_image_base64 = None

        for out_block in outputs:
            if not isinstance(out_block, dict):
                continue

            # Check for SAM predictions
            for key in out_block.keys():
                lower = key.lower()
                if lower in ("sam_3", "sam3", "sam"):
                    sam_data = out_block.get(key)
                    if isinstance(sam_data, dict):
                        preds = sam_data.get("predictions") or sam_data.get("masks") or []
                        all_predictions.extend(preds)

            # Check for mask visualization
            for vk in ["mask_visualization", "visualization", "annotated_image", "image"]:
                if vk in out_block:
                    val = out_block[vk]
                    candidate = None
                    if isinstance(val, dict):
                        candidate = val.get("value") or val.get("image") or val.get("data")
                    elif isinstance(val, str):
                        candidate = val
                    if candidate and candidate.strip():
                        annotated_image_base64 = candidate
                        break
            if annotated_image_base64:
                break

        if annotated_image_base64 and annotated_image_base64.startswith("data:"):
            annotated_image_base64 = annotated_image_base64.split(",", 1)[1]

        # Normalize detections
        normalized_detections = []
        for pred in all_predictions:
            if not isinstance(pred, dict):
                continue
            normalized_detections.append(
                {
                    "class": pred.get("class") or pred.get("label") or prompt,
                    "confidence": pred.get("confidence", pred.get("score", 1.0)),
                    "x": pred.get("x", 0),
                    "y": pred.get("y", 0),
                    "width": pred.get("width", pred.get("w", 0)),
                    "height": pred.get("height", pred.get("h", 0)),
                    "mask": pred.get("mask") or pred.get("encoded_mask") or pred.get("rle"),
                }
            )

        # 8️⃣ Build response
        response_data = {
            "success": True,
            "prompt": prompt,
            "annotated_image": annotated_image_base64,
            "detections": normalized_detections,
            "total_detections": len(normalized_detections),
            "message": f"Found {len(normalized_detections)} '{prompt}' objects"
            if normalized_detections
            else f"No '{prompt}' found",
        }

        print("✓ Response prepared successfully")
        return jsonify(response_data)

    except requests.Timeout:
        return jsonify({"error": "Request timeout", "details": "The request took too long."}), 504

    except requests.RequestException as e:
        return jsonify({"error": "Network error", "details": str(e)}), 500

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    finally:
        # Clean up temp file
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                print(f"🗑️  Temp file deleted: {uploaded_file_path}")
            except Exception as e:
                print("⚠️  Failed to delete temp file:", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*50}")
    print(f"🚀 Starting Finder AI Backend on port {port}")
    print(f"{'='*50}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
