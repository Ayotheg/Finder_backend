# finder_backend.py
import os
import io
import json
import base64
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
app = Flask(__name__)
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

WORKFLOW_URL = (
    f"https://serverless.roboflow.com/{RF_WS}/workflows/{RF_WFID}?api_key={RF_KEY}"
)

print("\n============================")
print("✓ Using Roboflow Workflow URL:")
print(WORKFLOW_URL)
print("============================\n")


# ---------------------------
# Image compression helper
# ---------------------------
def compress(path, max_dim=1920, quality=85):
    try:
        img = Image.open(path)
        if img.mode == "RGBA":
            img = img.convert("RGB")

        if max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return buffer.read()
    except:
        return open(path, "rb").read()


# ---------------------------
# Health route
# ---------------------------
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "workflow": RF_WFID})


# ---------------------------
# MAIN: Analyze route
# ---------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():
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
        if prompt == "":
            prompt = "all objects"

        # workflow requires text_prompt
        text_prompt = prompt

        # -----------------------
        # Save + Compress
        # -----------------------
        fname = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, fname)
        file.save(path)

        img_bytes = compress(path)
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        # -----------------------
        # CORRECT SAM 3 PAYLOAD
        # -----------------------
        payload = {
            "inputs": {
                "image": b64_img,         # some workflows use input_image – yours uses image
                "text_prompt": text_prompt
            }
        }

        print("📤 Sending Payload to Roboflow:")
        print(json.dumps(payload)[:500], "...\n")

        # -----------------------
        # Call Roboflow
        # -----------------------
        res = requests.post(
            WORKFLOW_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        if not res.ok:
            print("❌ Roboflow error:", res.text)
            return jsonify({
                "error": "Roboflow request failed",
                "status": res.status_code,
                "details": res.json()
            }), 400

        data = res.json()

        # -----------------------
        # Extract outputs
        # -----------------------
        outputs = data.get("outputs", [])
        annotated = None
        detections = []

        for block in outputs:
            if not isinstance(block, dict):
                continue

            # mask visualization
            if "mask_visualization" in block:
                mv = block["mask_visualization"]
                if isinstance(mv, dict):
                    mv = mv.get("value") or mv.get("image")
                if isinstance(mv, str):
                    annotated = mv

            # prediction block
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
        print("❌ SERVER ERROR:", str(e))
        return jsonify({
            "error": "Server crashed",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500


# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"✓ Backend running on port {port}")
    app.run(host="0.0.0.0", port=port)
