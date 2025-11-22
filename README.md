Startup Instructions
1. Start the Backend (Port 8000)
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py

2. Start the Frontend (Port 5173)
cd frontend
npm run dev

3. Access the Platform

Open your browser and visit:
http://localhost:5173

Features

Three attack methods: FGSM, PGD, DCT

Configurable parameters: epsilon, alpha, steps, freq_threshold

GPU acceleration: automatically detects and uses CUDA

Model stays in memory: loaded once at startup to avoid repeated loading

Real-time results: prediction comparison (original vs adversarial), attack success state, perturbation statistics

Drag-and-drop upload: supports JPG/PNG image uploads

Responsive design: mobile-friendly
