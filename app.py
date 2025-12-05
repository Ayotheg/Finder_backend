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
@app.route("/")
def index():
    return jsonify(
        {
            "name": "Finder AI Backend",
            "version": "2.0",
            "endpoints": {"health": "/api/health", "analyze": "/api/analyze"},
            "features": ["SAM 3 Text Prompts", "Mask Visualization"],
        }
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "message": "Finder AI Backend is running",
            "roboflow_configured": bool(ROBOFLOW_API_KEY),
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    uploaded_file_path = None
    try:
        print("\n" + "=" * 50)
        print("NEW REQUEST RECEIVED")
        print("=" * 50)

        # Validate file
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        if not allowed_file(file.filename):
            return (
                jsonify({"error": "Invalid file type. Allowed: PNG, JPG, JPEG, GIF"}),
                400,
            )

        # Prompt
        prompt = (request.form.get("prompt") or "").strip()
        if not prompt:
            print("⚠️  No prompt provided - using default 'all objects'")
            prompt = "all objects"

        # Save temporarily
        filename = secure_filename(file.filename)
        uploaded_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(uploaded_file_path)
        print(f"📁 File saved: {filename}")

        # Compress & encode
        compressed_bytes = compress_image(uploaded_file_path)
        encoded_image = base64.b64encode(compressed_bytes).decode("utf-8")
        print(f"✓ Image encoded (base64 length: {len(encoded_image)} chars)")
        prompts_array = [p.strip() for p in prompt.split(",")] if "," in prompt else [prompt]

        # Build payload
        payload = {
            "api_key": ROBOFLOW_API_KEY,
            "inputs": {"image": encoded_image, "prompts": prompts_array},
        }

        print(f"🚀 Calling Roboflow workflow at: {ROBOFLOW_WORKFLOW_URL}")
        # Best practice: send JSON; serverless workflows expect JSON for base64 image + prompts
        roboflow_resp = requests.post(
            ROBOFLOW_WORKFLOW_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60
        )

        print(f"📥 Roboflow response status: {roboflow_resp.status_code}")

        if not roboflow_resp.ok:
            # Try to include JSON-decoded error if available
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
                        "message": "Workflow configuration error. Ensure the workflow accepts 'prompts' (text array) and outputs mask_visualization.",
                    }
                ),
                500,
            )

        # Parse response
        try:
            roboflow_data = roboflow_resp.json()
        except Exception as e:
            print("❌ Failed to parse Roboflow JSON:", e)
            return jsonify({"error": "Invalid JSON from Roboflow", "details": str(e)}), 500

        # Extract outputs safely (must do this BEFORE any outputs-dependent logs)
        outputs = roboflow_data.get("outputs", [])
        print("✓ Roboflow response received successfully")
        print("=" * 50)
        print("FULL ROBOFLOW RESPONSE SNIPPET:")
        # Print a small snippet so logs don't explode
        try:
            print(json.dumps(roboflow_data if len(json.dumps(roboflow_data)) < 2000 else {"outputs": "[big]"}, indent=2) )
        except Exception:
            print("(response too large to pretty print)")

        # Validate outputs
        if not outputs:
            print("❌ No outputs in Roboflow response")
            return jsonify({"error": "No outputs from Roboflow", "raw": roboflow_data}), 500

        print(f"📦 Number of outputs blocks: {len(outputs)}")
        print(f"📦 First output keys: {list(outputs[0].keys())}")

        # Attempt to locate SAM predictions & visualizations across outputs
        all_predictions = []
        annotated_image_base64 = None

        # Search every output block (some workflows put SAM block first, viz second)
        for i, out_block in enumerate(outputs):
            # 1) Try to find SAM output (predictions / masks)
            if isinstance(out_block, dict):
                # common keys to check
                for key in out_block.keys():
                    lower = key.lower()
                    if lower in ("sam_3", "sam3", "sam"):
                        sam_data = out_block.get(key)
                        if isinstance(sam_data, dict):
                            preds = sam_data.get("predictions") or sam_data.get("masks") or []
                            if preds:
                                all_predictions.extend(preds)
                                print(f"📊 Found SAM predictions in outputs[{i}]['{key}'] (count={len(preds)})")
                    # generic prediction/segment keys
                    if "prediction" in lower or "segment" in lower or "masks" in lower:
                        val = out_block.get(key)
                        if isinstance(val, list):
                            all_predictions.extend(val)
                            print(f"📊 Found list predictions in outputs[{i}]['{key}'] (count={len(val)})")
                        elif isinstance(val, dict):
                            preds = val.get("predictions") or val.get("masks") or []
                            if preds:
                                all_predictions.extend(preds)
                                print(f"📊 Found dict predictions in outputs[{i}]['{key}'] (count={len(preds)})")

                # 2) Try to find a visualization image (mask_visualization or similar)
                viz_keys = ["mask_visualization", "visualization", "annotated_image", "image"]
                for vk in viz_keys:
                    if vk in out_block:
                        viz_val = out_block.get(vk)
                        if isinstance(viz_val, dict):
                            # many blocks use {'value': 'data:image/png;base64,...'} or {'value': '<base64>'}
                            candidate = viz_val.get("value") or viz_val.get("image") or viz_val.get("data")
                        else:
                            candidate = viz_val
                        if isinstance(candidate, str) and candidate.strip():
                            annotated_image_base64 = candidate
                            print(f"🎨 Found visualization in outputs[{i}]['{vk}']")
                            break
                if annotated_image_base64:
                    # stop searching viz once found
                    break

        # If annotated_image looks like a data URL, strip prefix
        if annotated_image_base64 and annotated_image_base64.startswith("data:"):
            annotated_image_base64 = annotated_image_base64.split(",", 1)[1]

        print(f"📊 Total detections found (len): {len(all_predictions)}")
        print(f"🎨 Visualization found: {'yes' if annotated_image_base64 else 'no'}")

        # If there are no predictions and no visualization, return helpful message (200 with empty results)
        if not annotated_image_base64:
            # If SAM produced predictions but viz missing, return predictions but indicate missing viz
            if all_predictions:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Visualization not generated",
                            "message": "Workflow returned predictions but no visualization image. Check Mask Visualization block.",
                            "prompt": prompt,
                            "total_detections": len(all_predictions),
                            "detections": all_predictions,
                            "annotated_image": None,
                        }
                    ),
                    200,
                )
            else:
                # No detections
                return (
                    jsonify(
                        {
                            "success": True,
                            "message": f"No objects found for prompt '{prompt}'",
                            "prompt": prompt,
                            "total_detections": 0,
                            "detections": [],
                            "annotated_image": None,
                        }
                    ),
                    200,
                )

        # Prepare detection list to return (normalize fields if possible)
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

        response_data = {
            "success": True,
            "prompt": prompt,
            "annotated_image": annotated_image_base64,
            "detections": normalized_detections,
            "total_detections": len(normalized_detections),
            "message": f"Found {len(normalized_detections)} '{prompt}' objects" if normalized_detections else f"No '{prompt}' found",
        }

        print("✓ Response prepared")
        return jsonify(response_data)

    except requests.Timeout:
        print("❌ Roboflow API timeout")
        return jsonify({"error": "Request timeout", "details": "The request took too long."}), 504

    except requests.RequestException as e:
        print("❌ Network error:", e)
        traceback.print_exc()
        return jsonify({"error": "Network error", "details": str(e)}), 500

    except Exception as e:
        print("❌ Unexpected error:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

    finally:
        # cleanup
        if uploaded_file_path and os.path.exists(uploaded_file_path):
            try:
                os.remove(uploaded_file_path)
                print(f"🗑️  Cleaned up: {uploaded_file_path}")
            except Exception as e:
                print("⚠️  Failed to delete temp file:", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*50}")
    print(f"🚀 Starting Finder AI Backend on port {port}")
    print(f"{'='*50}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
