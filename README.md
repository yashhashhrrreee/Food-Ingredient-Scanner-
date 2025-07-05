# Food Ingredient Scanner

## Overview
The Food Ingredient Scanner is an AI-powered web application that helps users analyze food ingredients and recipes through image processing. It combines computer vision and natural language processing to extract ingredient information from food packaging and recipe images.

## Features
- Image-based ingredient detection and analysis
- Text extraction from food packaging images
- Recipe information processing
- Web-based user interface for easy access
- Image highlighting for detected ingredients
- Search functionality for related food images
- Video generation with extracted information

## Tech Stack
### Frontend
- HTML/CSS
- Django Templates
- Static file handling

### Backend
- Python 3.11
- Django 5.1.3
- PyTorch (for image processing)
- Beautiful Soup (for web scraping)
- gTTS (Google Text-to-Speech)
- MoviePy (for video processing)

### Machine Learning
- ResNet model (stored as resnet_model.joblib)
- PyTorch vision transformations
- Image processing libraries (PIL)

## Project Structure
```
├── Food_Ingredient_Scanner/    # Main Django project directory
│   ├── Web_App/               # Django application
│   │   ├── migrations/        # Database migrations
│   │   ├── templates/         # HTML templates
│   │   ├── static/           # Static files (CSS, JS)
│   │   └── admin.py          # Admin configuration
├── Model/                     # ML model directory
│   ├── Code.ipynb            # Model training notebook
│   └── resnet_model.joblib   # Trained model
├── data/                     # Training data directory
│   ├── Fast food/            # Food image datasets
│   ├── Flavour/
│   └── Instant food/
└── myenv/                    # Python virtual environment
```

## Database Schema
```
HighlightedImage
- id (BigAutoField, primary key)
- highlighted_image (ImageField)
- created_at (DateTimeField)
- uploaded_image (ForeignKey to UploadedImage)

UploadedImage (implied from migration)
- Contains original uploaded images
```

## Installation and Setup
1. Clone the repository
```bash
git clone https://github.com/yashhashhrrreee/Food-Ingredient-Scanner-.git
cd Food-Ingredient-Scanner-
```

2. Create and activate virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Linux/Mac
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Apply database migrations
```bash
python manage.py migrate
```

5. Run the development server
```bash
python manage.py runserver
```

## Usage Steps
1. Access the web application through your browser at `http://localhost:8000`
2. Navigate to the Ingredient Scanner page
3. Upload a food package or recipe image
4. Wait for the AI model to process the image
5. View the extracted ingredients and related information
6. Optionally generate a video summary of the results

## Collaborators
- **Devanshu Katiyar** - ML Modeling and Data Pipeline
  - GitHub: [https://github.com/Devanshuk2004]

- **Yashashree Bedmutha** - Backend Framework and Final Production
  - GitHub: [https://github.com/yashhashhrrreee]

- **Jayraj Khivensara** - Frontend Design

## License
This project is licensed under the terms of the license included in the repository.
