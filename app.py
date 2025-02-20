import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import speech_recognition as sr
import requests
import json
from PIL import Image
import asyncio
import pygetwindow as gw
import base64
import logging
import re
import ctypes
import win32gui
import win32ui
import win32con
import numpy as np
import cv2
from ctypes import windll
import pygetwindow as gw
import numpy as np
from PIL import ImageGrab
import cv2
from typing import Optional, Tuple
from dotenv import load_dotenv
import json
from datetime import datetime
from typing import Dict, List, Optional
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from typing import List
import uvicorn
from PIL import ImageGrab
from meeting_report_emailer import MeetingReportEmailer
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
#CARD_STREAM_URL = os.getenv('CARD_STREAM_URL', 'rtsp://metaverse911:hellomoto123@192.168.1.106:554/stream1')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found in environment variables!")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMPANY_CONTEXT = """
Virtual Reality (VR) is an immersive technology that simulates a digital environment, allowing users to experience and interact with a computer-generated world. It typically requires a VR headset, motion controllers, and sometimes additional accessories to enhance the experience. By using advanced graphics, 3D audio, and motion tracking, VR can create highly realistic simulations for various applications, including gaming, education, training, and healthcare. The technology provides a sense of presence, making users feel as if they are truly inside the virtual environment rather than just observing it on a screen.
The applications of VR extend beyond entertainment, playing a crucial role in industries like medicine, military training, architecture, and mental health therapy. For example, medical professionals use VR for surgical simulations, while businesses employ it for virtual meetings and product design. Despite its advantages, challenges such as high costs, motion sickness, and the need for powerful hardware still exist. However, with ongoing advancements, VR continues to evolve, making it more accessible and improving its capabilities.

"""
class TextRequest(BaseModel):
    text: str
# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize speech recognition
recognizer = sr.Recognizer()

# Track detection states
face_detected = False
card_detected = False


# Card stream URL from environment variable


class MeetingTracker:
  
    def __init__(self, storage_dir: str = "meeting_logs"):
        self.storage_dir = storage_dir
        self.current_meeting = {
            "start_time": None,
            "end_time": None,
            "participant_name": None,
            "participant_email": "NA",
            "participant_phone": "NA",
            "participant_company": "NA",
            "questions": [],
            "responses": [],
            "discussion_overview": None
        }
        
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    async def generate_discussion_overview(self) -> str:
        """Generate a discussion overview using GPT API"""
        if not self.current_meeting["questions"]:
            return "No discussion took place."

        # Prepare conversation history for GPT
        conversation = ""
        for q, r in zip(self.current_meeting["questions"], self.current_meeting["responses"]):
            conversation += f"Q: {q}\nA: {r}\n\n"

        prompt = (
            "Based on the following conversation, provide a concise overview of what was discussed. "
            "Focus on the main points, key decisions, and important information shared. "
            "Keep it to 2-3 sentences.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {OPENAI_API_KEY}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'gpt-3.5-turbo',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0.3
                    }
                )
            )
            
            overview = response.json()['choices'][0]['message']['content'].strip()
            self.current_meeting["discussion_overview"] = overview
            return overview
        except Exception as e:
            logger.error(f"Error generating discussion overview: {str(e)}")
            return "Error generating discussion overview."

    def start_meeting(self, participant_name: Optional[str] = None):
        """Start a new meeting session"""
        self.current_meeting["start_time"] = datetime.now().isoformat()
        self.current_meeting["participant_name"] = participant_name
        self.current_meeting["questions"] = []
        self.current_meeting["responses"] = []
        self.current_meeting["topics_discussed"] = set()

    def update_contact_info(self, email: str, phone: str, company: str):
        """Update participant contact information"""
        self.current_meeting["participant_email"] = email
        self.current_meeting["participant_phone"] = phone
        self.current_meeting["participant_company"] = company

    def _save_meeting_log(self, summary: Dict) -> str:
        """Save meeting summary to a file with contact information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        participant = self.current_meeting["participant_name"] or "anonymous"
        filename = f"{self.storage_dir}/meeting_{participant}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Meeting Summary ===\n\n")
            f.write(f"Date: {summary['meeting_date']}\n")
            f.write(f"Time: {summary['start_time']} - {summary['end_time']}\n")
            f.write(f"Duration: {summary['duration_minutes']:.1f} minutes\n")
            f.write(f"Participant: {summary['participant_name']}\n")
            f.write(f"Email: {self.current_meeting['participant_email']}\n")
            f.write(f"Phone: {self.current_meeting['participant_phone']}\n")
            f.write(f"Company: {self.current_meeting['participant_company']}\n")
            f.write(f"Total Questions Asked: {summary['total_questions']}\n\n")
            
            f.write("=== Discussion Overview ===\n")
            f.write(f"{summary['discussion_overview']}\n\n")
            
            f.write("=== Questions and Responses ===\n\n")
            for i, qa in enumerate(summary['questions_and_responses'], 1):
                f.write(f"Q{i}: {qa['Q']}\n")
                f.write(f"A{i}: {qa['A']}\n\n")
        
        return filename

    def add_interaction(self, question: str, response: str):
        """Record a Q&A interaction"""
        self.current_meeting["questions"].append(question)
        self.current_meeting["responses"].append(response)
        
        # Extract potential topics from question and response
        # This is a simple implementation - could be enhanced with NLP
        words = set((question + " " + response).lower().split())
        self.current_meeting["topics_discussed"].update(words)

    async def end_meeting(self) -> str:
        """End the meeting and save the summary"""
        if not self.current_meeting["start_time"]:
            return "No active meeting to end"

        self.current_meeting["end_time"] = datetime.now().isoformat()
        
        # Generate discussion overview
        await self.generate_discussion_overview()
        
        # Generate summary
        summary = self._generate_summary()
        
        # Save to file
        filename = self._save_meeting_log(summary)
        
        # Reset current meeting
        self.current_meeting = {
            "start_time": None,
            "end_time": None,
            "participant_name": None,
            "participant_email": "NA",
            "participant_phone": "NA",
            "participant_company": "NA",
            "questions": [],
            "responses": [],
            "discussion_overview": None
        }
        
        return filename

    async def generate_discussion_overview(self) -> str:
        """Generate a discussion overview using GPT API"""
        if not self.current_meeting["questions"]:
            return "No discussion took place."

        # Prepare conversation history for GPT
        conversation = ""
        for q, r in zip(self.current_meeting["questions"], self.current_meeting["responses"]):
            conversation += f"Q: {q}\nA: {r}\n\n"

        prompt = (
            "Based on the following conversation, provide a concise overview of what was discussed. "
            "Focus on the main points, key decisions, and important information shared. "
            "Keep it to 2-3 sentences.\n\n"
            f"Conversation:\n{conversation}"
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {OPENAI_API_KEY}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': 'gpt-3.5-turbo',
                        'messages': [{'role': 'user', 'content': prompt}],
                        'temperature': 0.3
                    }
                )
            )
            
            overview = response.json()['choices'][0]['message']['content'].strip()
            self.current_meeting["discussion_overview"] = overview
            return overview
        except Exception as e:
            logger.error(f"Error generating discussion overview: {str(e)}")
            return "Error generating discussion overview."
            
    def _generate_summary(self) -> Dict:
        """Generate a meeting summary"""
        start_time = datetime.fromisoformat(self.current_meeting["start_time"])
        end_time = datetime.fromisoformat(self.current_meeting["end_time"])
        duration = end_time - start_time
        
        summary = {
            "meeting_date": start_time.strftime("%Y-%m-%d"),
            "start_time": start_time.strftime("%H:%M:%S"),
            "end_time": end_time.strftime("%H:%M:%S"),
            "duration_minutes": duration.total_seconds() / 60,
            "participant_name": self.current_meeting["participant_name"],
            "total_questions": len(self.current_meeting["questions"]),
            "questions_and_responses": [
                {"Q": q, "A": r} for q, r in zip(
                    self.current_meeting["questions"],
                    self.current_meeting["responses"]
                )
            ],
            "discussion_overview": self.current_meeting["discussion_overview"]
        }
        
        return summary
    




async def process_frame(frame, face_cascade):
    """Process a single frame for face detection"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return len(faces) > 0
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return False


async def get_ai_response(text: str, name: Optional[str] = None) -> str:
    """Get response using single context, limited to 50 words"""
    try:
        system_context = COMPANY_CONTEXT + "\nIMPORTANT: All responses must be 50 words or less."
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [
                        {"role": "system", "content": system_context},
                        {"role": "user", "content": f"Provide a concise response (maximum 50 words) to: {text}"}
                    ],
                    'temperature': 0.5,
                    'max_tokens': 100  # Limiting tokens to encourage shorter responses
                }
            )
        )
        
        response_text = response.json()['choices'][0]['message']['content'].strip()
        
        # Ensure response is exactly 50 words or less
        words = response_text.split()
        if len(words) > 50:
            response_text = ' '.join(words[:50]) + ','
        
        # Format response with name if provided
        if name:
            response_text = f"{name}, {response_text}"
            
        # Replace periods with commas
        response_text = response_text.replace('.', ',')
            
        return response_text
        
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return "Error processing your question, Please try again,"
    
face_detected = False
card_detected = False

# Card stream URL from environment variable
CARD_STREAM_URL = os.getenv('CARD_STREAM_URL', 'rtsp://metaverse911:hellomoto123@192.168.1.56:554/stream1')

async def validate_contact_info(text: str) -> tuple[bool, Optional[Dict]]:
    """Validate if text contains valid name and extract contact information"""
    try:
        prompt = (
            "Extract name and contact information from the text. If a piece of information "
            "is not found, use 'NA'. Respond in this exact JSON format:\n"
            "{\n"
            "  'is_valid': true/false,\n"
            "  'name': 'extracted name or NA',\n"
            "  'email': 'email or NA',\n"
            "  'phone': 'phone number or NA',\n"
            "  'company': 'company name or NA'\n"
            "}\n\n"
            f"Text to analyze: {text}"
        )
        
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {OPENAI_API_KEY}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.1
                }
            )
        )
        
        contact_info = json.loads(response.json()['choices'][0]['message']['content'].strip())
        return contact_info['is_valid'], contact_info
    except Exception as e:
        logger.error(f"Error in contact info validation: {str(e)}")
        return False, None
    
async def get_frame_from_window(window_name="BlueStacks App Player"):
    """Capture frame from a named window using Win32 API"""
    try:
        # Find the window handle
        hwnd = win32gui.FindWindow(None, window_name)
        if not hwnd:
            logger.error(f"Window '{window_name}' not found")
            return None

        # Get window dimensions - need to handle different window states
        if win32gui.IsIconic(hwnd):  # If minimized
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            
        # Get the window's client area dimensions
        client_rect = win32gui.GetClientRect(hwnd)
        client_width = client_rect[2] - client_rect[0]
        client_height = client_rect[3] - client_rect[1]
        
        # Get the window's full dimensions including borders
        window_rect = win32gui.GetWindowRect(hwnd)
        window_width = window_rect[2] - window_rect[0]
        window_height = window_rect[3] - window_rect[1]

        # Use the larger of the two dimensions
        width = max(client_width, window_width)
        height = max(client_height, window_height)

        # Account for DPI scaling
        scale_factor = windll.shcore.GetScaleFactorForDevice(0) / 100
        width = int(width * scale_factor)
        height = int(height * scale_factor)

        # Get window DC and create compatible DC
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create bitmap object
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(save_bitmap)

        # Copy window contents to bitmap
        # Use PW_RENDERFULLCONTENT (3) to capture the entire window including DirectX content
        result = windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
        if result != 1:
            logger.error("PrintWindow failed")
            return None

        # Convert bitmap to numpy array
        bitmap_bits = save_bitmap.GetBitmapBits(True)
        img_array = np.frombuffer(bitmap_bits, dtype='uint8')
        
        try:
            img = img_array.reshape((height, width, 4))  # RGBA format
        except ValueError as e:
            logger.error(f"Reshape error: {e}. Array size: {img_array.size}, Expected: {height}x{width}x4")
            return None

        # Convert RGBA to BGR for OpenCV
        frame = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Clean up resources
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        return frame

    except Exception as e:
        logger.error(f"Error capturing window frame: {str(e)}")
        return None
async def check_card_stream(websocket: WebSocket):
    """Check card stream from window with optimized settings"""
    logger.info("Starting optimized card stream check from window")
    
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    try:
        previous_frame = None
        last_screenshot_time = 0
        
        while True:
            current_time = asyncio.get_event_loop().time()
            
            # Take screenshot every 3 seconds
            if current_time - last_screenshot_time >= 3:
                frame = await get_frame_from_window()
                
                if frame is not None:
                    # Resize frame for faster processing
                    frame = cv2.resize(frame, (640, 480))
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(screenshots_dir, f"capture_{timestamp}.jpg")
                    
                    # Use optimized image encoding parameters
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    
                    # Save screenshot with compression
                    cv2.imwrite(screenshot_path, frame, encode_param)
                    
                    # Optimize frame encoding for websocket
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send frame asynchronously
                    await websocket.send_text(f"frame:{frame_base64}")
                    
                    if previous_frame is not None:
                        # Use optimized frame comparison
                        has_changed = await compare_frames_optimized(previous_frame, frame)
                        
                        if has_changed:
                            logger.info("Change detected - processing with GPT")
                            
                            # Process with Vision API using compressed frame
                            success, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                            if success:
                                text = await process_image_with_vision_api(encoded_frame.tobytes())
                                if text:
                                    is_valid, contact_info = await validate_contact_info(text)
                                    if is_valid and contact_info['name'] != 'NA':
                                        valid_card_path = os.path.join(screenshots_dir, f"valid_card_{timestamp}.jpg")
                                        cv2.imwrite(valid_card_path, frame, encode_param)
                                        return contact_info
                    
                    previous_frame = frame.copy()
                    last_screenshot_time = current_time
                
                # Shorter sleep interval
                await asyncio.sleep(0.05)
            else:
                # Very short sleep when not processing
                await asyncio.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error in card stream check: {str(e)}")
        return None

async def process_card_stream(websocket: WebSocket):
    """Main function to process card stream frames from window at 1 frame per 3 seconds"""
    logger.info("Starting card stream processing from window")
    
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    
    try:
        previous_frame = None
        last_frame_time = 0
        
        while True:
            current_time = asyncio.get_event_loop().time()
            
            # Only process a frame every 3 seconds
            if current_time - last_frame_time >= 3.0:
                frame = await get_frame_from_window()
                
                if frame is not None:
                    # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(screenshots_dir, f"capture_{timestamp}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    
                    # Send frame to websocket
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    await websocket.send_text(f"frame:{frame_base64}")
                    
                    # Process frame if it's different from previous
                    if previous_frame is None or await compare_frames(previous_frame, frame):
                        # Process frame with Vision API
                        success, encoded_frame = cv2.imencode('.jpg', frame)
                        if success:
                            text = await process_image_with_vision_api(encoded_frame.tobytes())
                            if text:
                                is_valid, contact_info = await validate_contact_info(text)
                                if is_valid and contact_info['name'] != 'NA':
                                    logger.info("Valid card detected")
                                    return contact_info
                    
                    previous_frame = frame.copy()
                    last_frame_time = current_time
            
            # Short sleep to prevent CPU overload while waiting for next 3-second interval
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Error in card stream processing: {str(e)}")
        return None
async def compare_frames_optimized(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.1) -> bool:
    """Optimized frame comparison function"""
    try:
        # Convert to grayscale for faster processing
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate percentage of changed pixels
        changed_pixels = np.count_nonzero(diff > 25)  # Threshold for pixel difference
        total_pixels = diff.size
        change_ratio = changed_pixels / total_pixels
        
        return change_ratio > threshold
        
    except Exception as e:
        logger.error(f"Error in frame comparison: {str(e)}")
        return False

GOOGLE_VISION_API_KEY = os.getenv('GOOGLE_VISION_API_KEY')

async def compare_frames(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.1) -> bool:
    """
    Compare two frames to detect significant changes using structural similarity
    Returns True if significant change detected
    """
    if frame1 is None or frame2 is None:
        return True
        
    # Resize frames for faster processing
    frame1_resized = cv2.resize(frame1, (320, 240))
    frame2_resized = cv2.resize(frame2, (320, 240))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
    
    # Calculate structural similarity index
    score = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
    
    # Return True if frames are significantly different
    return score < (1.0 - threshold)

async def process_image_with_vision_api(image_bytes: bytes) -> str:
    """Process image using Google Cloud Vision API"""
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        vision_api_url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
        payload = {
            "requests": [{
                "image": {"content": image_b64},
                "features": [{"type": "TEXT_DETECTION"}]
            }]
        }

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: requests.post(
                vision_api_url,
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
        )
        response.raise_for_status()

        result = response.json()
        if 'responses' in result and result['responses']:
            text_annotation = result['responses'][0].get('textAnnotations', [])
            if text_annotation:
                return text_annotation[0].get('description', '')
        return ''
    except Exception as e:
        logger.error(f"Error in Vision API processing: {str(e)}")
        return ''


async def init_card_stream():
    """Initialize card stream with minimal buffering"""
    try:
        logger.info(f"Attempting to connect to RTSP stream at: {CARD_STREAM_URL}")
        
        stream = cv2.VideoCapture(CARD_STREAM_URL, cv2.CAP_FFMPEG)
        if not stream.isOpened():
            logger.error(f"Failed to open RTSP stream: {CARD_STREAM_URL}")
            return None
            
        # Configure stream properties
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Minimum buffer size
        
        # Test stream
        ret, frame = stream.read()
        if not ret:
            logger.error("Failed to read first frame from stream")
            return None
            
        logger.info("Successfully initialized card stream")
        return stream
        
    except Exception as e:
        logger.error(f"Error initializing card stream: {str(e)}")
        return None
    


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Initialize meeting tracker and emailer
    meeting_tracker = MeetingTracker()
    emailer = MeetingReportEmailer()
    meeting_tracker.start_meeting()
    
    try:
        # Start card stream processing
        contact_info = await process_card_stream(websocket)
        
        if contact_info:
            # Update meeting with contact info
            meeting_tracker.update_contact_info(
                email=contact_info['email'],
                phone=contact_info['phone'],
                company=contact_info['company']
            )
            
            # Send welcome message
            await websocket.send_text(f"card_detected:{json.dumps(contact_info)}")
            welcome_message = f"Welcome {contact_info['name']}! How can I assist you today?"
            await websocket.send_text(f"ai_response:{welcome_message}")
            
            # Handle Q&A
            last_message_time = asyncio.get_event_loop().time()
            
            while True:
                try:
                    # Check for timeout
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_message_time > 300:  # 5 minutes timeout
                        break
                    
                    # Wait for questions
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    last_message_time = current_time
                    
                    # Check for session end
                    if isinstance(data, str) and "thankyou" in data.lower() and "done" in data.lower():
                        break
                    
                    # Handle questions
                    if data.startswith('question:'):
                        question = data[9:]
                        response = await get_ai_response(question, contact_info['name'])
                        meeting_tracker.add_interaction(question, response)
                        await websocket.send_text(f"ai_response:{response}")
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    break
        else:
            await websocket.send_text("ai_response:No valid business card detected. Please try again.")
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # End meeting and cleanup
        await meeting_tracker.end_meeting()
        await websocket.close()

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)