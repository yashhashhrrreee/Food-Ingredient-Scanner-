from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.image.name

class HighlightedImage(models.Model):
    uploaded_image = models.ForeignKey(UploadedImage, on_delete=models.CASCADE)  # Link to the original uploaded image
    highlighted_image = models.ImageField(upload_to='highlighted_images/')  # Path to the highlighted image
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Highlighted Image for {self.uploaded_image.image.name}"

from django.db import models

# Model to store audio files
class Audio(models.Model):
    text = models.TextField()  
    audio_file = models.FileField(upload_to='audio/')  # Path to the generated audio file
    created_at = models.DateTimeField(auto_now_add=True)  

    def __str__(self):
        return f"Audio for text: {self.text}..."


# Model to store video files
class Video(models.Model):
    text = models.TextField() 
    video_file = models.FileField(upload_to='video/')  # Path to the generated video file
    created_at = models.DateTimeField(auto_now_add=True) 

    def __str__(self):
        return f"Video for text: {self.text}..."  


# Model to store merged video files (audio + video)
class MergedVideo(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)  # Associated video
    audio = models.ForeignKey(Audio, on_delete=models.CASCADE)  # Associated audio
    merged_video_file = models.FileField(upload_to='merged_video/') 
    text = models.TextField()  
    created_at = models.DateTimeField(auto_now_add=True)  

    def __str__(self):
        return f"Merged Video for text: {self.text}..."  

