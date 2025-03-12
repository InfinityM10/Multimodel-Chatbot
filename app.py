import streamlit as st
import requests
import os
import tempfile
import base64
from PIL import Image
import io
import json
import time
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

# Set page configuration
st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Define the available Ollama models that are actually available
MODELS = {
    "Text": "llama3",
    "Image": "llava",
    "Audio": "sentence-transformers",  # Using sentence-transformers for audio
    "Video": "llava"  # Use llava for video frame analysis (upload only)
}

# Initialize FAISS and sentence transformer models
@st.cache_resource
def load_sentence_transformer():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

@st.cache_resource
def initialize_faiss_index():
    # Create an empty FAISS index
    dimension = 384  # Depends on the sentence-transformer model used
    index = faiss.IndexFlatL2(dimension)
    return index

# Load models on startup
sentence_transformer = load_sentence_transformer()
faiss_index = initialize_faiss_index()

# Initialize session state to store chat history and media content
if "messages" not in st.session_state:
    st.session_state.messages = []

# In-memory storage for media content
if "media_store" not in st.session_state:
    st.session_state.media_store = {
        "image": None,
        "audio": None,
        "video": None,
        "image_description": "",
        "audio_transcription": "",
        "video_description": "",
        "last_media_type": None
    }

# In-memory storage for embeddings
if "stored_embeddings" not in st.session_state:
    st.session_state.stored_embeddings = []
    st.session_state.stored_texts = []
    st.session_state.prompt_history = []

def process_text(text, model="llama3"):
    """Process text input using the specified model."""
    try:
        # Add context about previous media if available
        context = ""
        if st.session_state.media_store["last_media_type"] == "image" and st.session_state.media_store["image_description"]:
            context += f"\nContext from the uploaded image: {st.session_state.media_store['image_description']}"
        elif st.session_state.media_store["last_media_type"] == "audio" and st.session_state.media_store["audio_transcription"]:
            context += f"\nContext from the uploaded audio: {st.session_state.media_store['audio_transcription']}"
        elif st.session_state.media_store["last_media_type"] == "video" and st.session_state.media_store["video_description"]:
            context += f"\nContext from the uploaded video: {st.session_state.media_store['video_description']}"
        
        # Add prompt history if requested
        if "previous prompts" in text.lower() or "previous questions" in text.lower() or "prompt history" in text.lower():
            if st.session_state.prompt_history:
                context += "\nPrevious prompts/questions:\n" + "\n".join([f"- {prompt}" for prompt in st.session_state.prompt_history])
            else:
                context += "\nNo previous prompts or questions found."
        
        # Prepare the final prompt with context
        full_prompt = text
        if context:
            full_prompt += f"\n\n{context}"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": full_prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error processing text: {str(e)}"

def process_image(image_file, prompt, model="llava", store=True):
    """Process image input using the specified model."""
    try:
        # Convert the image to base64
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()["response"]
        
        # Store image and description if needed
        if store:
            # Reset the file pointer to read again
            image_file.seek(0)
            st.session_state.media_store["image"] = image_file.read()
            st.session_state.media_store["image_description"] = result
            st.session_state.media_store["last_media_type"] = "image"
            
        return result
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_audio(audio_file, prompt="Transcribe this audio", model="sentence-transformers", store=True):
    """Process audio input using whisper for transcription and FAISS for semantic search."""
    try:
        # Import whisper here to avoid loading it unless needed
        import whisper
        
        # Save the uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        
        # Load whisper model
        whisper_model_size = "base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner(f"Loading Whisper model ({whisper_model_size}) on {device}..."):
            whisper_model = whisper.load_model(whisper_model_size, device=device)
        
        # Transcribe with Whisper
        with st.spinner("Transcribing audio with Whisper..."):
            transcription_result = whisper_model.transcribe(tmp_path)
            transcription = transcription_result["text"]
        
        # Create embedding for the transcription
        with st.spinner("Creating embedding with sentence-transformers..."):
            embedding = sentence_transformer.encode(transcription)
            
            # Store embedding in FAISS index and in memory
            faiss_index.add(np.array([embedding]).astype('float32'))
            st.session_state.stored_embeddings.append(embedding)
            st.session_state.stored_texts.append(transcription)
            
            # If we have a specific question/prompt, perform semantic search
            if prompt != "Transcribe this audio":
                # Get embedding for the question
                query_embedding = sentence_transformer.encode(prompt)
                
                # Search in FAISS index
                k = min(3, len(st.session_state.stored_texts))  # Get top k results
                if k > 0:
                    D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
                    
                    # Get the most relevant transcripts
                    relevant_transcripts = [st.session_state.stored_texts[i] for i in I[0]]
                    
                    # Use LLM to generate response based on relevant transcripts
                    context = "\n".join(relevant_transcripts)
                    llm_prompt = f"{prompt}\n\nRelevant transcriptions:\n{context}"
                    
                    process_response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3",
                            "prompt": llm_prompt,
                            "stream": False
                        }
                    )
                    process_response.raise_for_status()
                    final_response = process_response.json()["response"]
                    result = f"Transcription: {transcription}\n\nResponse (based on semantic search): {final_response}"
                else:
                    result = f"Transcription: {transcription}\n\nNo previous transcriptions for semantic search."
            else:
                result = f"Transcription: {transcription}"
            
        # Clean up the temporary file
        os.unlink(tmp_path)
        
        # Store audio transcription if needed
        if store:
            # Reset the file pointer to read again
            audio_file.seek(0)
            st.session_state.media_store["audio"] = audio_file.read()
            st.session_state.media_store["audio_transcription"] = transcription
            st.session_state.media_store["last_media_type"] = "audio"
        
        return result
    except ImportError:
        return "Error: Please install the required libraries with 'pip install -U openai-whisper sentence-transformers faiss-cpu'"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def process_video(video_file, prompt="Describe what's happening in this video", model="llava", store=True):
    """Process video input by extracting frames and audio, then analyzing both."""
    try:
        import cv2
        import whisper
        import tempfile
        import subprocess
        import os
        
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        
        try:
            # Create a temporary file for extracted audio
            audio_path = tmp_path + ".wav"
            
            # Extract audio using ffmpeg (requires ffmpeg to be installed)
            try:
                with st.spinner("Extracting audio from video..."):
                    # Use ffmpeg to extract audio
                    subprocess.run([
                        'ffmpeg', '-i', tmp_path, 
                        '-q:a', '0', '-map', 'a', 
                        '-y',  # Overwrite output file if it exists
                        audio_path
                    ], check=True, capture_output=True)
                
                # Load whisper model for audio transcription
                whisper_model_size = "base"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                with st.spinner(f"Loading Whisper model ({whisper_model_size}) on {device}..."):
                    whisper_model = whisper.load_model(whisper_model_size, device=device)
                
                # Transcribe audio with Whisper
                with st.spinner("Transcribing audio from video..."):
                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                        transcription_result = whisper_model.transcribe(audio_path)
                        audio_transcription = transcription_result["text"]
                    else:
                        audio_transcription = "No audio could be extracted or audio is silent."
            except Exception as audio_error:
                audio_transcription = f"Audio extraction or transcription failed: {str(audio_error)}"
                st.warning(f"Audio processing issue: {audio_transcription}")
            
            # Extract frames from the video
            vidcap = cv2.VideoCapture(tmp_path)
            
            # Get video information
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # For videos longer than 30 seconds, sample frames
            max_frames = 5
            frame_indices = []
            
            if duration > 30 and frame_count > max_frames:
                # Sample frames evenly throughout the video
                step = frame_count // max_frames
                for i in range(0, frame_count, step):
                    if len(frame_indices) < max_frames:
                        frame_indices.append(i)
            else:
                # For shorter videos, take frames at 1-second intervals
                step = int(fps) if fps > 0 else 1
                for i in range(0, frame_count, step):
                    if len(frame_indices) < max_frames:
                        frame_indices.append(i)
            
            # Extract the selected frames
            frames = []
            for idx in frame_indices:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, image = vidcap.read()
                if success:
                    frames.append(image)
            
            # Make sure to release the video capture object before trying to delete the file
            vidcap.release()
            
            if not frames:
                return "Could not extract frames from video file."
            
            # Process each frame and collect results
            frame_results = []
            
            for i, frame in enumerate(frames):
                # Save frame to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp:
                    img_path = img_tmp.name
                    cv2.imwrite(img_path, frame)
                
                try:
                    # Process the frame with the image model
                    with open(img_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    # For the first frame, use the original prompt
                    if i == 0:
                        frame_prompt = f"{prompt} (This is frame {i+1} of {len(frames)} from the video)"
                    else:
                        # For subsequent frames, add context
                        frame_prompt = f"This is frame {i+1} of {len(frames)} from the same video. Describe what's new or different in this frame."
                    
                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": frame_prompt,
                            "images": [image_data],
                            "stream": False
                        }
                    )
                    
                    if response.status_code == 200:
                        frame_results.append(response.json()["response"])
                finally:
                    # Clean up image file
                    try:
                        os.unlink(img_path)
                    except Exception as e:
                        st.warning(f"Could not delete temporary image file: {e}")
            
            # Summarize the results, combining visual and audio analysis
            combined_prompt = f"""I've analyzed a video both visually and audibly.

VISUAL ANALYSIS:
I've analyzed {len(frames)} frames from the video. Here's what I saw in each frame:

{chr(10).join([f"Frame {i+1}: {result}" for i, result in enumerate(frame_results)])}

AUDIO TRANSCRIPTION:
{audio_transcription}

Based on both the visual frames and audio transcription, please provide a comprehensive summary of what's happening in the entire video. Include both what can be seen and what can be heard.
"""
            
            summary_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",  # Use a text model for summarization
                    "prompt": combined_prompt,
                    "stream": False
                }
            )
            
            if summary_response.status_code == 200:
                result = summary_response.json()["response"]
            else:
                # If summarization fails, return the individual analyses
                result = f"""Video Analysis Results:

VISUAL ANALYSIS:
{chr(10).join([f"Frame {i+1}: {res}" for i, res in enumerate(frame_results)])}

AUDIO TRANSCRIPTION:
{audio_transcription}"""

            # Store video and description if needed
            if store:
                # Reset the file pointer to read again
                video_file.seek(0)
                st.session_state.media_store["video"] = video_file.read()
                st.session_state.media_store["video_description"] = result
                st.session_state.media_store["last_media_type"] = "video"
                
            return result
                
        finally:
            # Clean up files in a finally block
            try:
                # Make sure we're no longer accessing the files
                time.sleep(0.5)
                
                # Delete the temporary video file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
                # Delete the temporary audio file
                if 'audio_path' in locals() and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception as e:
                st.warning(f"Could not delete temporary files: {e}")
        
    except ImportError as ie:
        missing_lib = str(ie).split("'")[-2] if "'" in str(ie) else "required libraries"
        return f"Error: Please install {missing_lib}. For video processing with audio, you need: opencv-python, openai-whisper, and ffmpeg."
    except Exception as e:
        return f"Error processing video: {str(e)}"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["type"] == "text":
            st.write(message["content"])
        elif message["type"] == "image":
            st.image(message["image"], caption=message["content"])
        elif message["type"] == "audio":
            # Only show transcription/response, not the audio player
            st.write(message["content"])
        elif message["type"] == "video":
            # Don't display video, just show content
            st.write(message["content"])

# Sidebar for model selection and options
with st.sidebar:
    st.title("Model Settings")
    
    text_model = st.selectbox(
        "Text Model",
        ["llama3", "mistral", "mixtral", "gemma", "phi3"],
        index=0
    )
    
    image_model = st.selectbox(
        "Image Model",
        ["llava", "bakllava", "llava:13b"],
        index=0
    )
    
    # Display the current media status
    st.divider()
    st.subheader("Current Media")
    
    if st.session_state.media_store["last_media_type"] == "image":
        st.success("✓ Image available for follow-up questions")
    elif st.session_state.media_store["last_media_type"] == "audio":
        st.success("✓ Audio available for follow-up questions")
    elif st.session_state.media_store["last_media_type"] == "video":
        st.success("✓ Video available for follow-up questions")
    else:
        st.info("No media uploaded yet")
    
    if st.button("Clear Media"):
        st.session_state.media_store = {
            "image": None,
            "audio": None,
            "video": None,
            "image_description": "",
            "audio_transcription": "",
            "video_description": "",
            "last_media_type": None
        }
        st.success("Media cleared!")
    
    st.divider()
    
    st.markdown("""
    ### Required Python packages
    - streamlit
    - requests
    - Pillow
    - openai-whisper
    - opencv-python (for video)
    - faiss-cpu (or faiss-gpu)
    - sentence-transformers
    - torch
    - ffmpeg (system dependency)
    """)
    
    # FAISS Index Statistics
    st.subheader("FAISS Index Stats")
    st.write(f"Stored embeddings: {len(st.session_state.stored_embeddings)}")
    
    if st.button("Clear Embeddings"):
        st.session_state.stored_embeddings = []
        st.session_state.stored_texts = []
        # Reset FAISS index
        dimension = 384
        st.session_state.faiss_index = faiss.IndexFlatL2(dimension)
        st.success("Embeddings cleared!")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.prompt_history = []
        st.rerun()

# Media upload or text input section
# Only show media upload options if no media is currently available
if st.session_state.media_store["last_media_type"] is None:
    # Input area
    input_mode = st.radio(
        "Input Type",
        ["Text", "Image", "Audio", "Video"],
        horizontal=True
    )

    if input_mode == "Text":
        prompt = st.text_input("Your prompt:", placeholder="Enter your message here...")
        
        if st.button("Send") and prompt:
            # Store prompt in history
            st.session_state.prompt_history.append(prompt)
            
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "type": "text",
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = process_text(prompt, model=text_model)
                    st.write(response)
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": response
            })
            
            # Rerun to clear the input field
            st.rerun()

    elif input_mode == "Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        prompt = st.text_input("Your prompt about the image:", placeholder="What can you tell me about this image?")
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image")
            
            if st.button("Process") and prompt:
                # Store prompt in history
                st.session_state.prompt_history.append(prompt)
                
                # Reset the file pointer to the beginning
                uploaded_image.seek(0)
                
                # Add user message to chat
                img_bytes = uploaded_image.getvalue()
                st.session_state.messages.append({
                    "role": "user",
                    "type": "image",
                    "content": prompt,
                    "image": img_bytes
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.image(uploaded_image, caption=prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Processing image..."):
                        # Reset the file pointer again
                        uploaded_image.seek(0)
                        response = process_image(uploaded_image, prompt, model=image_model)
                        st.write(response)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": response
                })
                
                # Rerun to update the UI
                st.rerun()

    elif input_mode == "Audio":
        uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
        prompt = st.text_input("Your prompt about the audio:", placeholder="What is being said in this audio?")
        
        if uploaded_audio is not None:
            # Add a processing button
            if st.button("Process"):
                # Store prompt in history
                if prompt:
                    st.session_state.prompt_history.append(prompt)
                else:
                    st.session_state.prompt_history.append("Transcribe this audio")
                
                # Reset the file pointer to the beginning
                uploaded_audio.seek(0)
                
                # Add user message to chat
                st.session_state.messages.append({
                    "role": "user",
                    "type": "audio",
                    "content": prompt if prompt else "Transcribe this audio"
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt if prompt else "Transcribe this audio")
                
                # Get AI response
                with st.chat_message("assistant"):
                    # Reset the file pointer again
                    uploaded_audio.seek(0)
                    response = process_audio(uploaded_audio, prompt if prompt else "Transcribe this audio")
                    st.write(response)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": response
                })
                
                # Rerun to update the UI
                st.rerun()

    elif input_mode == "Video":
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
        prompt = st.text_input("Your prompt about the video:", placeholder="What is happening in this video?")
        
        if uploaded_video is not None:
            # Show upload confirmation
            st.success("Video uploaded successfully. Click 'Process' to analyze.")
            
            if st.button("Process"):
                # Store prompt in history
                if prompt:
                    st.session_state.prompt_history.append(prompt)
                else:
                    st.session_state.prompt_history.append("Describe this video")
                
                # Reset the file pointer to the beginning
                uploaded_video.seek(0)
                
                # Add user message to chat
                st.session_state.messages.append({
                    "role": "user",
                    "type": "video",
                    "content": prompt if prompt else "Describe this video"
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt if prompt else "Describe this video")
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Processing video..."):
                        # Reset the file pointer again
                        uploaded_video.seek(0)
                        response = process_video(uploaded_video, prompt if prompt else "Describe what's happening in this video", model=image_model)
                        st.write(response)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "text",
                    "content": response
                })
                
                # Rerun to update the UI
                st.rerun()
else:
    # Media is already available, just show the message input
    message_input = st.chat_input("Ask a follow-up question about the uploaded media...")
    
    # Display info about the current media
    media_type = st.session_state.media_store["last_media_type"]
    if media_type:
        st.info(f"You have a {media_type} uploaded. Ask questions about it or type 'clear media' to remove it.")
    
    # If we have a message input, process it
    if message_input:
        # Special case for clearing media
        if message_input.lower() == "clear media":
            st.session_state.media_store = {
                "image": None,
                "audio": None,
                "video": None,
                "image_description": "",
                "audio_transcription": "",
                "video_description": "",
                "last_media_type": None
            }
            
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "type": "text",
                "content": message_input
            })
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": "Media cleared. You can now upload new media."
            })
            
            # Rerun to update the UI
            st.rerun()
        else:
            # Store prompt in history
            st.session_state.prompt_history.append(message_input)
            
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "type": "text",
                "content": message_input
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(message_input)
            
            # Get AI response based on the last media type
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = process_text(message_input, model=text_model)
                    st.write(response)
            
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "type": "text",
                "content": response
            })