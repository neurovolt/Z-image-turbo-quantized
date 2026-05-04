from flask import Flask, request, jsonify, render_template_string
import requests
import json
import uuid
import time
import base64
import random
import os

app = Flask(__name__)

COMFYUI_URL = os.environ.get("COMFYUI_URL", "").rstrip("/")


def queue_prompt(prompt_workflow):
    try:
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={
                "prompt": prompt_workflow,
                "client_id": str(uuid.uuid4())
            },
            headers={
                "ngrok-skip-browser-warning": "true"
            },
            timeout=30
        )

        print("STATUS:", response.status_code)
        print("RAW RESPONSE:", response.text[:500])

        try:
            return response.json()
        except:
            return {"error": "Invalid JSON from ComfyUI", "raw": response.text[:200]}

    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def get_history(prompt_id):
    try:
        response = requests.get(
            f"{COMFYUI_URL}/history/{prompt_id}",
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_image(filename, subfolder, folder_type):
    params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(
        f"{COMFYUI_URL}/view",
        params=params,
        headers={"ngrok-skip-browser-warning": "true"}
    )
    return response.content


def build_workflow(prompt, negative_prompt, width, height, steps, cfg, seed):
    if seed == -1:
        seed = random.randint(0, 999999999)

    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "z_image_turbo-Q5_K_S.gguf",
                "weight_dtype": "default"
            }
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "Qwen3-4B-Q5_K_M.gguf",
                "type": "lumina2",
                "device": "default"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "ae.safetensors"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": prompt
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["2", 0],
                "text": negative_prompt
            }
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "beta",
                "denoise": 1.0
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["3", 0]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "zimage_output"
            }
        }
    }
    return workflow


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json

    try:
        workflow = build_workflow(
            prompt=data['prompt'],
            negative_prompt=data.get('negative_prompt', ''),
            width=data.get('width', 768),
            height=data.get('height', 1024),
            steps=data.get('steps', 9),
            cfg=data.get('cfg', 1.0),
            seed=data.get('seed', -1)
        )

        result = queue_prompt(workflow)

        # 🔥 FIX: handle missing prompt_id
        if 'prompt_id' not in result:
            return jsonify({
                'success': False,
                'error': f"ComfyUI error: {result}"
            })

        prompt_id = result['prompt_id']
        used_seed = workflow['7']['inputs']['seed']

        # Polling
        max_wait = 300
        waited = 0

        while waited < max_wait:
            time.sleep(2)
            waited += 2

            history = get_history(prompt_id)

            if prompt_id in history:
                outputs = history[prompt_id]['outputs']

                for node_id, output in outputs.items():
                    if 'images' in output:
                        img_info = output['images'][0]
                        img_data = get_image(
                            img_info['filename'],
                            img_info['subfolder'],
                            img_info['type']
                        )

                        img_b64 = base64.b64encode(img_data).decode('utf-8')

                        return jsonify({
                            'success': True,
                            'image': img_b64,
                            'seed': used_seed
                        })

        return jsonify({
            'success': False,
            'error': 'Timeout — generation took too long'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health')
def health():
    try:
        r = requests.get(
            f"{COMFYUI_URL}/system_stats",
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=5
        )
        return jsonify({'status': 'ok', 'comfyui': r.status_code == 200})
    except:
        return jsonify({'status': 'ok', 'comfyui': False})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
