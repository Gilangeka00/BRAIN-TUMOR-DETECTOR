import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from timm import create_model
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ThreadTimeout

# ======== Konfigurasi ========
dir_path = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(dir_path, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'your_secret_key'  # Ganti dengan env var di produksi

# ======== Load Model ViT ========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
state_dict = torch.load(
    os.path.join(dir_path, 'model', 'vit_model_brain60.pth'),
    map_location=device
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ======== Transformasi Gambar ========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ======== Utility ========
def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def is_brain_mri(pil_img: Image.Image,
                 dp: float = 1.2,
                 min_dist_ratio: float = 8,
                 param1: int = 50,
                 param2: int = 30,
                 min_radius_ratio: float = 4,
                 max_radius_ratio: float = 2) -> bool:
    gray = np.array(pil_img.convert('L'))
    blurred = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=h / min_dist_ratio,
        param1=param1,
        param2=param2,
        minRadius=int(h / min_radius_ratio),
        maxRadius=int(h / max_radius_ratio)
    )
    return circles is not None

# ======== Routes ========
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('Silakan pilih file terlebih dahulu.', 'error')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Tipe file tidak diizinkan. (png/jpg/jpeg)', 'error')
            return redirect(request.url)

        pil_img = Image.open(file.stream).convert('RGB')
        if not is_brain_mri(pil_img):
            flash('Mohon unggah gambar scan MRI otak yang valid.', 'error')
            return redirect(request.url)

        # Simpan file
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pil_img.save(filepath)

        # Persiapan input model
        x = transform(pil_img).unsqueeze(0).to(device)

        # Prediksi dengan timeout
        def predict():
            with torch.no_grad():
                return model(x)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(predict)

        try:
            logits = future.result(timeout=5)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(probs.argmax())
            confidence = probs[idx]
            label = 'No Tumor' if idx == 1 else 'Tumor Detected'

            return render_template(
                'index.html',
                filename=filename,
                label=label,
                probability=f"{confidence * 100:.2f}%",
                probability_float=confidence
            )
        except ThreadTimeout:
            flash('Proses prediksi gagal (timeout).', 'error')
            return redirect(request.url)

    return render_template('index.html')


# Redirect ke static/uploads jika diperlukan
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename=f'uploads/{filename}'), code=301)


# Route untuk menampilkan halaman contoh-gambar.html
@app.route('/contoh-gambar')
def contoh_gambar():
    # `contoh-gambar.html` harus berada di folder `templates/`
    return render_template('contoh-gambar.html')


# ======== Main ========
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
