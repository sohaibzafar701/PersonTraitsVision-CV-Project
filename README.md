# Gender, Glasses, and Shirt Color Detection

This is a Flask web application that detects gender, glasses, and shirt color from an uploaded image using YOLOv11 models and HSV-based color analysis with MediaPipe. The app provides a user-friendly web interface for uploading images and displays an annotated output image with bounding boxes for gender (green), glasses (blue), and shirt region (red), along with a color patch for the detected shirt color.

## Features
- **Image Upload**: Upload images via a sleek web interface.
- **Gender Detection**: Uses a YOLOv11 model to classify gender (e.g., male, female).
- **Glasses Detection**: Uses a YOLOv11 model to detect glasses (e.g., glasses, sunglasses, no glasses).
- **Shirt Color Detection**: Analyzes shirt color using HSV color space and MediaPipe pose estimation.
- **Annotated Output**: Displays a single image with bounding boxes and detection results.
- **Responsive Design**: Modern, Tailwind CSS-styled interface for desktop and mobile.

## Project Structure
```
flask-detection-app/
├── app.py                    # Main Flask application
├── templates/
│   └── index.html            # HTML template for the web interface
├── uploads/                  # Temporary folder for uploaded images
├── static/
│   └── outputs/              # Temporary folder for output images
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Prerequisites
- **Python**: Version 3.10 or higher.
- **Trained YOLOv11 Models**: Pre-trained models for gender and glasses detection.
- **System Requirements**: At least 4GB RAM and a modern CPU for running YOLOv11 models. GPU is optional but recommended for faster inference.

## Setup Instructions
Follow these steps to set up and run the Flask app locally.

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/yourusername/flask-detection-app.git
cd flask-detection-app
```

### 2. Create a Virtual Environment
Create and activate a Python virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Note**: Installation may take time due to large dependencies like `mediapipe` and `ultralytics`. Ensure a stable internet connection.

### 4. Create Required Folders
Ensure the `uploads` and `static/outputs` folders exist for storing temporary files:
```bash
mkdir -p uploads static/outputs
```

### 5. Run the Flask App
Start the Flask development server:
```bash
python app.py
```
The app will run on `http://localhost:5000`. Open this URL in a web browser.

## Usage
1. **Access the Web Interface**:
   - Navigate to `http://localhost:5000` in your browser.
   - You’ll see a modern interface with an upload form styled using Tailwind CSS.

2. **Upload an Image**:
   - Click the file input to select an image (`.jpg`, `.png`, etc.).
   - Click **Upload and Analyze** to process the image.

3. **View Results**:
   - The app displays:
     - An annotated image with bounding boxes: green for gender, blue for glasses, red for shirt region.
     - A color patch in the bottom-right corner showing the detected shirt color.
     - Text results for gender (e.g., "Male"), glasses (e.g., "Glasses"), and shirt color (e.g., "Blue").

4. **Troubleshooting**:
   - If no results appear, ensure the image contains a person with visible shoulders and hips (required for shirt color detection).
   - Check the console for errors if models fail to load or processing crashes.

## Expected Model Outputs
- **Gender Model**: Should output classes like `male`, `female`, or similar.
- **Glasses Model**: Should output classes like `glasses`, `sunglasses`, `no_glasses`, or similar.
- **Shirt Color**: Outputs colors like `red`, `blue`, `green`, `white`, `black`, etc., based on HSV analysis.

## Troubleshooting
- **Model Loading Errors**:
  - Ensure YOLOv11 model files exist and paths in `app.py` are correct.
  - Verify models are compatible with the `ultralytics` library (YOLOv11 format).
- **Dependency Installation Issues**:
  - If installation is slow, try installing packages individually:
    ```bash
    pip install flask pillow
    pip install opencv-python-headless
    pip install mediapipe
    pip install ultralytics
    ```
- **Image Processing Errors**:
  - Ensure uploaded images are valid (`.jpg`, `.png`) and contain a person.
  - Check console logs for OpenCV or MediaPipe errors.
- **Server Not Starting**:
  - Ensure port `5000` is free. If not, kill conflicting processes or change the port in `app.py`:
    ```python
    app.run(host='0.0.0.0', port=8000)
    ```
- **Blank or No Results**:
  - Verify MediaPipe detects pose landmarks (shoulders, hips).
  - Ensure YOLOv11 models are trained correctly and output expected classes.

## Performance Tips
- **Image Size**: Large images may slow down processing. Consider resizing images in `app.py` (see commented code for resizing).
- **Model Optimization**: Use lightweight YOLOv11 models (e.g., YOLOv11n) for faster inference.
- **Memory Usage**: Monitor system resources, as YOLOv11 and MediaPipe can be memory-intensive.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **YOLOv11**: For gender and glasses detection models.
- **MediaPipe**: For pose estimation in shirt color detection.
- **Flask**: For the web framework.
- **Tailwind CSS**: For the responsive, modern interface.

---
*Built with ❤️ by Sohaib*  
*Last updated: June 2025*
