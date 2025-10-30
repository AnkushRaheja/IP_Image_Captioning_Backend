from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import pickle
import io
import os
import sys

# Import your model architecture
from model import EncoderCNN, DecoderRNN
from vocabulary import Vocabulary

app = Flask(__name__)

# Configure CORS to allow requests from anywhere
CORS(app, resources={r"/*": {"origins": "*"}})

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters (kept as constants for lazy loading)
EMBED_SIZE = 256
HIDDEN_SIZE = 512

# Model paths
VOCAB_PATH = 'vocab.pkl'
ENCODER_PATH = 'models/encoder-3.pkl'
DECODER_PATH = 'models/decoder-3.pkl'

# Global variables for models
vocab = None
encoder = None
decoder = None
models_loaded = False

def check_model_files(paths):
    """Check that each path in `paths` exists. If any are missing, print
    a helpful message with tips and raise FileNotFoundError.
    """
    missing = [p for p in paths if not os.path.exists(p)]
    if not missing:
        return

    tip_lines = [
        "One or more required model files are missing:",
    ]
    tip_lines += [f" - {p}" for p in missing]
    tip_lines += [
        "\nTips to resolve:",
        " - If you keep models in the repo, ensure they've been pushed and that Git LFS files were pulled:",
        "     git lfs install && git lfs pull",
        " - On Render set the service Root Directory to 'backend' so these files are present at build time.",
        " - You can also override paths with environment variables: VOCAB_PATH, ENCODER_PATH, DECODER_PATH",
        " - If models are large or builds fail, consider hosting them in external storage (S3) and downloading at startup.",
    ]

    message = "\n".join(tip_lines)
    # Print to stderr so Render build logs show the message
    print(message, file=sys.stderr)
    raise FileNotFoundError(message)

def load_models():
    global vocab, encoder, decoder, models_loaded
    if models_loaded:
        return
    
    print("Loading vocabulary from vocab.pkl...")
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    
    print("Initializing models on device: cpu")
    encoder = EncoderCNN(EMBED_SIZE).to(device)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(device)
    
    print("Loading trained weights...")
    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device))
    
    encoder.eval()
    decoder.eval()
    
    models_loaded = True
    print("✅ Models loaded successfully!")

# Check for required model files and provide clearer, actionable errors if they are missing
check_model_files([VOCAB_PATH, ENCODER_PATH, DECODER_PATH])

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def generate_caption(image, max_length=20):
    """
    Generate caption for the given image
    
    Args:
        image: PIL Image object
        max_length: Maximum caption length
    
    Returns:
        str: Generated caption
    """
    load_models()  # Load models if not already loaded
    
    with torch.no_grad():
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract features using encoder
        features = encoder(image_tensor)
        
        # Reshape features for decoder: (batch_size, embed_size) -> (batch_size, 1, embed_size)
        features = features.unsqueeze(1)
        
        # Generate caption (returns a list of word IDs)
        sampled_ids = decoder.sample(features, max_len=max_length)
        
        # Convert word ids to words
        caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            if word == '<end>':
                break
            if word != '<start>':
                caption.append(word)
        
        # Join words to form caption
        caption_text = ' '.join(caption)
        
        # Capitalize first letter
        if caption_text:
            caption_text = caption_text[0].upper() + caption_text[1:]
        
        return caption_text

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Image Captioning API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            'POST /caption': 'Generate caption for an image',
            'GET /health': 'Health check',
            'GET /info': 'Model information'
        }
    })

@app.route('/caption', methods=['POST'])
def caption_image():
    """
    API endpoint to generate caption for uploaded image
    
    Expects:
        - multipart/form-data with 'image' file
    
    Returns:
        JSON with success status and generated caption
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided. Please upload an image file.'
            }), 400
        
        file = request.files['image']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename. Please select a valid image.'
            }), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Generate caption
        caption = generate_caption(image)
        
        return jsonify({
            'success': True,
            'caption': caption,
            'image_size': image.size
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'ready to load',
        'device': str(device),
        'models_loaded': models_loaded
    })

@app.route('/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not models_loaded:
        load_models()
    return jsonify({
        'architecture': {
            'encoder': 'ResNet-50 (pre-trained on ImageNet)',
            'decoder': 'LSTM',
            'embed_size': EMBED_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'vocabulary_size': len(vocab)
        },
        'device': str(device),
        'status': 'ready'
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Image Captioning Backend Server")
    print("="*60)
    print(f"✅ Device: {device}")
    # vocab may not be loaded yet due to lazy loading; show status safely
    try:
        vocab_size_display = len(vocab) if vocab is not None else 'not loaded'
    except Exception:
        vocab_size_display = 'unknown'
    print(f"✅ Vocabulary size: {vocab_size_display}")
    print(f"✅ Embed size: {EMBED_SIZE}")
    print(f"✅ Hidden size: {HIDDEN_SIZE}")
    print(f"✅ Server running on http://localhost:5000")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
