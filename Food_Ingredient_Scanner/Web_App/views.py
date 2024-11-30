from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageUploadForm
from .models import UploadedImage ,HighlightedImage, Audio, Video, MergedVideo
from django.shortcuts import get_object_or_404
from bs4 import BeautifulSoup
from django.conf import settings
from gtts import gTTS
from PIL import Image, ImageFilter
import pytesseract
import language_tool_python
import numpy as np
import enchant
import subprocess
import re
import os
import cv2
import pytesseract
import requests

def home(request):
    return render(request, 'home.html')

def ingredient_scanner(request):
    return render(request, 'ingredient_scanner.html')

def about(request):
    return render(request, 'about.html')

def features(request):
    return render(request, 'features.html')

def contact(request):
    return render(request, 'contact.html')

def analysis_ingredients(request):
    return render(request, 'analysis_ingredients.html')

def generate_video(request):
    return render(request, 'generate_video.html')

def ingredient_scanner(request):
    """View function for the ingredient scanner page.
    
    This function handles the image upload form and saves the uploaded image to the database.
    
    Args:
        request (HttpRequest): The HTTP request object.
        
    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == 'POST' and request.FILES['image']:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()  # Save the uploaded image to the database
            return redirect('upload_success')  # Redirect to a success page
    else:
        form = ImageUploadForm()

    return render(request, 'ingredient_scanner.html', {'form': form})

def upload_success(request):
    """ 
    View function to display the uploaded image and its URL.
    
    This function retrieves the most recent uploaded image from the database and passes the image URL and name to the template.
    
    Args:
        request (HttpRequest): The HTTP request object.
        
    Returns:
        HttpResponse: The HTTP response object.
    """
    uploaded_image = UploadedImage.objects.last()  # Get the most recent uploaded image

    if uploaded_image:  # Ensure there is an image
        image_url = uploaded_image.image.url  # Get the image URL to display
        image_name = uploaded_image.image.name  # Get the image name (filename)
    else:
        image_url = None
        image_name = None

    # Pass the image URL and name to the template
    return render(request, 'upload_success.html', {
        'uploaded_image': uploaded_image,
        'image_url': image_url,
        'image_name': image_name,
    })

# WHO guidelines URL
WHO_URL = "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"

def fetch_unhealthy_nutrients():
    """ 
    Function to fetch unhealthy nutrients from the WHO guidelines.
    
    This function fetches the WHO guidelines webpage and extracts the unhealthy nutrients to limit or reduce.
    
    Returns:
        list: A list of unhealthy nutrients to limit or reduce.
    """
    try:
        response = requests.get(WHO_URL)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        unhealthy_nutrients = []
        for li in soup.find_all('li'):
            if "limit" in li.get_text() or "reduce" in li.get_text():
                nutrient_match = re.search(r'(sugar|sodium|saturated fat|trans fat)', li.get_text(), re.IGNORECASE)
                if nutrient_match:
                    unhealthy_nutrients.append(nutrient_match.group(0).lower())

        return list(set(unhealthy_nutrients))

    except requests.exceptions.RequestException as e:
        print(f"Error fetching WHO guidelines: {e}")
        return []

def extract_text(image):
    """ 
    Function to extract text from an image.
    
    This function uses the Tesseract OCR library to extract text from an image.
    
    Args:
        image (numpy.ndarray): The image to extract text from.
        
    Returns:
        str: The extracted text from the image.
    """
    # model = "D:\Study\Sem VI\Project Phase 2\Food-Ingredient-Scanner-\Model\resnet_model.h5"
    # model.eval()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

def highlight_unhealthy_nutrients(image, text, unhealthy_nutrients, uploaded_image_instance):
    """ 
    Function to highlight unhealthy nutrients in an image.
    
    This function highlights the area of the image where unhealthy nutrients are mentioned in the text.
    
    Args:
        image (numpy.ndarray): The image to highlight.
        text (str): The text extracted from the image.
        unhealthy_nutrients (list): A list of unhealthy nutrients to highlight.
        uploaded_image_instance (UploadedImage): The UploadedImage instance.
        
    Returns:
        HighlightedImage: The HighlightedImage instance with the highlighted image path.
    """
    lines = text.splitlines()
    height, width, _ = image.shape

    # Create a copy of the image to highlight nutrients
    highlighted_image = image.copy()

    for line in lines:
        for nutrient in unhealthy_nutrients:
            if nutrient in line.lower():
                color = (0, 0, 255)  # Highlight color in red
                # Highlight the area of the image (you can adjust the rectangle coordinates)
                cv2.rectangle(highlighted_image, (0, height // 2), (width, height // 2 + 50), color, 2)

    # Save the highlighted image in the 'highlighted_images/' folder
    # Correct the filename to avoid including the original image path in the filename
    highlighted_image_filename = f"highlighted_{os.path.basename(uploaded_image_instance.image.name)}"  # Use the base filename only
    highlighted_image_path = os.path.join(settings.MEDIA_ROOT, 'highlighted_images', highlighted_image_filename)

    # Create the 'highlighted_images' folder if it doesn't exist
    os.makedirs(os.path.dirname(highlighted_image_path), exist_ok=True)

    # Save the image to disk
    cv2.imwrite(highlighted_image_path, highlighted_image)
    print(f"Highlighted image saved at: {highlighted_image_path}")

    # Save the highlighted image path to the database
    highlighted_image_instance = HighlightedImage.objects.create(
        uploaded_image=uploaded_image_instance,  # Link it to the uploaded image
        highlighted_image=highlighted_image_path  # The path to the saved image
    )

    return highlighted_image_instance

def analyse_ingredients(request):
    """
    View function to analyze the uploaded image and extract unhealthy nutrients.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The HTTP response object.
    """
    if request.method == 'POST':
        # Get the uploaded image ID from the form
        image_id = request.POST.get('image_id')

        if image_id:
            # Retrieve the uploaded image using its ID
            uploaded_image = get_object_or_404(UploadedImage, id=image_id)

            if uploaded_image:
                # Process the image
                image = cv2.imread(uploaded_image.image.path)
                if image is not None:
                    text = extract_text(image)  # This function extracts the text from the image
                    unhealthy_nutrients = fetch_unhealthy_nutrients()  # This function gets the unhealthy nutrients

                    # Analyze healthiness
                    health_status = "Good for Health"
                    if any(nutrient in text.lower() for nutrient in unhealthy_nutrients):
                        health_status = "Not Good for Health"

                    # Highlight unhealthy nutrients in the image
                    highlighted_image_instance = highlight_unhealthy_nutrients(image, text, unhealthy_nutrients, uploaded_image)
                    
                    # Prepare context for rendering
                    context = {
                        'uploaded_image': uploaded_image,
                        'image_name': uploaded_image.image.name,
                        'health_status': health_status,
                        'highlighted_image_url': highlighted_image_instance.highlighted_image.url,
                        'unhealthy_nutrients': unhealthy_nutrients,
                    }

                    # Render the analysis results page with the context
                    return render(request, 'analyse_ingredients.html', context)
                else:
                    return HttpResponse("Error processing the image")
            else:
                return HttpResponse("Image not found")
        else:
            return HttpResponse("No image ID provided")
    else:
        return HttpResponse("Invalid request")

def extract_text_from_image(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Convert image to grayscale (helps with text extraction)
    gray_img = img.convert('L')

    # Apply a sharpening filter or thresholding to clean up the image (Optional, try adjusting)
    processed_img = gray_img.filter(ImageFilter.SHARPEN)

    # Open the image with OpenCV for better control
    cv_img = cv2.imread(image_path)
    gray_cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray_cv_img, 150, 255, cv2.THRESH_BINARY)

    # Use pytesseract to extract text from the processed image
    extracted_text = pytesseract.image_to_string(thresh_img)
    return extracted_text

def correct_text_with_tool(extracted_text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(extracted_text)
    corrected_text = language_tool_python.utils.correct(extracted_text, matches)
    return corrected_text

def clean_and_fix_text(text):
    dictionary = enchant.Dict("en_US")
    
    # Remove unwanted symbols and keep only letters, numbers, spaces, and basic punctuation
    text = re.sub(r"[^\w\s.,;'-]", "", text)
    
    # Split text into words
    words = text.split()
    
    # Filter out non-dictionary words or gibberish based on simple length and dictionary lookup
    cleaned_words = [word for word in words if dictionary.check(word) and len(word) > 1]
    
    # Reassemble the cleaned words into a single string
    cleaned_text = " ".join(cleaned_words)
    
    return cleaned_text

def text_to_audio(text, output_file):
    tts = gTTS(text=text, lang='en')  # You can change 'en' to your preferred language code
    tts.save(output_file)
    print(f"Audio saved as {output_file}")
    
def create_video_from_text(text, output_video, font_size=1, width=1280, height=720, fps=30, duration_per_line=30, max_line_length=40):
    """
    Create a video from text with automatic line breaks to fit the screen width.
    """
    # Create a blank frame (black background)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Set the text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)  # White text
    thickness = 2
    y0, dy = 50, 50  # Starting position for the text
    
    # Function to split the text into multiple lines with a max line length
    def split_text_into_lines(text, max_length):
        words = text.split()
        lines = []
        line = ""
        for word in words:
            if len(line + " " + word) <= max_length:
                line += " " + word
            else:
                lines.append(line.strip())
                line = word
        if line:
            lines.append(line.strip())
        return lines
    
    # Split the text into lines that fit the width of the video
    lines = split_text_into_lines(text, max_line_length)
    
    # Open a VideoWriter to create the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Write frames with the text
    for line in lines:
        # For each line, create multiple frames (to simulate time on screen)
        for _ in range(duration_per_line):  # Repeat the line for a set number of frames
            frame = np.zeros((height, width, 3), dtype=np.uint8)  # Clear the frame each time
            cv2.putText(frame, line, (50, y0), font, font_size, color, thickness, lineType=cv2.LINE_AA)
            out.write(frame)
        
        y0 += dy  # Move the text down for the next line
    
    out.release()  # Save the video
    print(f"Video saved as {output_video}")
    
def merge_audio_video_ffmpeg(video_file, audio_file, output_file):
    command = [
        'ffmpeg', '-i', video_file, '-i', audio_file, '-c:v', 'libx264', '-c:a', 'aac', '-strict', 'experimental', '-shortest', output_file
    ]
    subprocess.run(command, check=True)
    print(f"Final video saved as {output_file}")

def generate_video(request):
    if request.method == 'POST':
        # Get the uploaded image ID from the form
        image_id = request.POST.get('image_id')

        if image_id:
            # Retrieve the uploaded image using its ID
            uploaded_image = get_object_or_404(UploadedImage, id=image_id)

            if uploaded_image:
                # Process the image file
                image = cv2.imread(uploaded_image.image.path)
                if image is None:
                    return HttpResponse("Error processing the image.", status=400)

                # Extract text from the uploaded image
                extracted_text = extract_text_from_image(uploaded_image.image.path)
                if not extracted_text:
                    return HttpResponse("No text extracted from the image.", status=400)

                # Step 1: Generate audio from the extracted text
                audio_output_file = os.path.join(settings.MEDIA_ROOT, 'audio/generated_audio.mp3')
                text_to_audio(extracted_text, audio_output_file)

                # Save the audio file in the database
                audio = Audio.objects.create(
                    text=extracted_text,
                    audio_file='audio/generated_audio.mp3'
                )

                # Step 2: Generate video from the extracted text
                video_output_file = os.path.join(settings.MEDIA_ROOT, 'video/generated_video.mp4')
                create_video_from_text(extracted_text, video_output_file)

                # Save the video file in the database
                video = Video.objects.create(
                    text=extracted_text,
                    video_file='video/generated_video.mp4'
                )

                # Step 3: Merge audio and video
                merged_video_output = os.path.join(settings.MEDIA_ROOT, 'merged_video/merged_video.mp4')
                merge_audio_video_ffmpeg(
                    audio.audio_file.path,  # Use .path to get the full file path
                    video.video_file.path,  # Use .path to get the full file path
                    merged_video_output
                )

                # Save the merged video in the database, including the extracted text
                merged_video = MergedVideo.objects.create(
                    video=video,
                    audio=audio,
                    merged_video_file='merged_video/merged_video.mp4',
                    text=extracted_text
                )

                # Render the template with the merged video details
                return render(request, 'generate_video.html', {
                    'text': extracted_text,
                    'merged_video_url': merged_video.merged_video_file,
                })
            else:
                return HttpResponse("Image not found", status=404)
        else:
            return HttpResponse("No image ID provided", status=400)

    return render(request, 'generate_video.html')


