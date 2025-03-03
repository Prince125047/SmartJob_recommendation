import os
import random
import re
import time
import base64
import pickle
import requests
import spacy
import pdfplumber
from PIL import Image
import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text
import io
import subprocess
import importlib.util
# ===============================
# Helper Functions
# ===============================

def install_spacy_model(model_name):
    """Ensure SpaCy model is installed."""
    if not spacy.util.is_package(model_name):
        os.system(f"python -m spacy download {model_name}")

# Install SpaCy model if not already installed
install_spacy_model("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

def switch_page(new_page):
    """Switch to a new page."""
    st.session_state.page = new_page

# Load ML models
try:
    with open('ml-models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('ml-models/job_recommendation.pkl', 'rb') as f:
        job_recommendation_model = pickle.load(f)
except FileNotFoundError:
    st.error("ML model files are missing. Please upload `vectorizer.pkl` and `job_recommendation.pkl`.")
    st.stop()

# Function to read PDF files
def pdf_reader(file_path):
    return extract_text(file_path)

# Function to process uploaded PDF files
def process_uploaded_pdf(pdf_file):
    binary_buffer = io.BytesIO(pdf_file.read())
    save_image_path = os.path.join("./Uploaded_Resumes", pdf_file.name)
    os.makedirs("./Uploaded_Resumes", exist_ok=True)
    with open(save_image_path, "wb") as f:
        f.write(binary_buffer.getvalue())
    resume_text = pdf_reader(save_image_path)
    return resume_text, save_image_path

# Function to display PDF in Streamlit
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Name Parsing Function
def extract_name_with_font_info(file_path):
    name_keywords = ["name:", "full name:", "applicant:", "candidate:"]
    potential_names = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text_details = page.extract_words()
            for word in text_details:
                text = word["text"].strip()
                font_size = word["size"]
                font_name = word["fontname"]

                if "bold" in font_name.lower() or font_size > 14:
                    potential_names.append(text)
                for keyword in name_keywords:
                    if text.lower().startswith(keyword):
                        name = text[len(keyword):].strip()
                        if name and name[0].isupper():
                            return name

    if potential_names:
        return " ".join(potential_names[:2])
    return None

def extract_name_from_resume(resume_text, file_path=None):
    if file_path:
        name_from_font = extract_name_with_font_info(file_path)
        if name_from_font:
            return name_from_font
    lines = resume_text.split("\n")
    name_keywords = ["name:", "full name:", "applicant:", "candidate:"]

    def clean_text(text):
        return text.strip().lower()
    for line in lines[:10]:
        line = line.strip()
        for keyword in name_keywords:
            if clean_text(line).startswith(keyword):
                name = line[len(keyword):].strip()
                if name and name[0].isupper():
                    return name
        if line and line[0].isupper() and len(line.split()) >= 2:
            return line
    return None

# Extract Skills
def extract_skills_section(text):
    text = text.lower()

    skills_patterns = [
        r"\bskills\b[:\n]",
        r"\btechnical skills\b[:\n]",
        r"\bkey skills\b[:\n]",
        r"\bcore competencies\b[:\n]",
        r"\bareas of expertise\b[:\n]",
        r"\bprofessional skills\b[:\n]",
        r"\bexpertise\b[:\n]",
    ]

    start_index = None
    for pattern in skills_patterns:
        match = re.search(pattern, text)
        if match:
            start_index = match.end()
            break

    if start_index is None:
        return text

    stop_patterns = [
        r"\nexperience",
        r"\neducation",
        r"\nprojects",
        r"\ncertifications",
        r"\nsummary",
        r"\ncontact",
        r"\npersonal details",
        r"\nacademic details",
        r"\nprofile summary",
        r"\npositions of responsibility",
        r"\nextra-curricular achievements",
        r"\nsoft skills",
        r"\nachievements",
    ]
    end_index = len(text)
    for stop_pattern in stop_patterns:
        stop_match = re.search(stop_pattern, text[start_index:])
        if stop_match:
            end_index = start_index + stop_match.start()
            break

    skills_section = text[start_index:end_index].strip()
    return skills_section

# Match Skills
def match_skills(skills_section, skill_list):
    if not skills_section:
        return []

    skills_found = []
    for skill in skill_list:
        if re.search(rf"\b{re.escape(skill)}\b", skills_section, re.IGNORECASE):
            skills_found.append(skill)
    return list(set(skills_found))

# YouTube API Integration
API_KEY = "AIzaSyDbvYU855ZHRzu4SmqZG9OpQKXDJNsOeU0"
ALLOWED_CHANNELS = {
    "FreeCodeCamp": "UC8butISFwT-Wl7EV0hUK0BQ",
    "Chai with Code": "UCs6nmQViDpUw0nuIx9c_WvA"
}

def get_youtube_recommendations(skills_to_learn, num_recommendations=5):
    recommendations = {}
    selected_skills = skills_to_learn[:num_recommendations]

    for skill in selected_skills:
        for channel_name, channel_id in ALLOWED_CHANNELS.items():
            query = f"{skill} tutorial for beginners"
            url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&maxResults=1&type=video&channelId={channel_id}&key={API_KEY}"

            try:
                response = requests.get(url).json()
                if "items" in response and response["items"]:
                    video_id = response["items"][0]["id"]["videoId"]
                    video_title = response["items"][0]["snippet"]["title"]
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    recommendations[skill] = {
                        "channel": channel_name,
                        "title": video_title,
                        "url": video_url
                    }
                    break
            except Exception as e:
                st.error(f"Error fetching YouTube recommendations: {e}")
                continue

    return recommendations

# Course Recommender Function
def course_recommender(course_list, no_of_reco=5):
    random.shuffle(course_list)
    return course_list[:no_of_reco]

# Chatbot Button
def chatbot_button():
    st.subheader("Need Further Assistance?")
    chatbot_link = "https://cdn.botpress.cloud/webchat/v2.2/shareable.html?configUrl=https://files.bpcontent.cloud/2025/02/04/08/20250204081537-I2WZDKEN.json"
    st.markdown(
        f'<a href="{chatbot_link}" target="_blank" style="text-decoration:none;"><button style="background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; border: none;">Chat with our Bot</button></a>',
        unsafe_allow_html=True
    )

# Home Page
def home_page():
    try:
        img = Image.open("./Logo/logo2.png")
        st.image(img, use_container_width=True)
    except FileNotFoundError:
        st.error("Logo file not found. Please ensure 'logo2.png' is placed in the './Logo/' folder.")
        st.stop()

    st.title("SMART JOB RECOMMENDATION MODEL")
    st.markdown("Upload your resume and get smart recommendations.", unsafe_allow_html=True)

    pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
    if pdf_file is not None:
        with st.spinner("Uploading and Processing your Resume..."):
            time.sleep(2)
            try:
                resume_text, save_image_path = process_uploaded_pdf(pdf_file)
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")
                st.stop()

            doc = nlp(resume_text)
            email = None
            mobile_number = None

            for ent in doc.ents:
                if ent.label_ == "EMAIL":
                    email = ent.text
                elif ent.label_ == "PHONE":
                    mobile_number = ent.text

            # Extract skills section and match skills
            skills_section = extract_skills_section(resume_text)
            matched_skills = match_skills(skills_section, skill_list)

            # Vectorize the matched skills
            feature_vector = vectorizer.transform([" ".join(matched_skills)])
            vectorized_skills = vectorizer.get_feature_names_out()
            resume_skills = [skill for skill in vectorized_skills if feature_vector[0, vectorizer.vocabulary_[skill]] > 0]

        st.header("**Resume Analysis Results**")

        # Match skills
        st.subheader("**Your Skills**")
        if matched_skills:
            selected_skills = st.multiselect(
                "Select/Deselect your skills:",
                options=[skill.capitalize() for skill in matched_skills],
                default=[skill.capitalize() for skill in matched_skills]
            )
            matched_skills = [skill.lower() for skill in selected_skills]

            if not matched_skills:
                st.markdown("No skills selected.", unsafe_allow_html=True)
        else:
            st.markdown("No skills detected.", unsafe_allow_html=True)

        # Recommended Job Role
        if matched_skills:
            feature_vector = vectorizer.transform([" ".join(matched_skills)])
            predicted_jobs = job_recommendation_model.predict(feature_vector)[0]
            predicted_jobs_list = [job.strip() for job in predicted_jobs.split(",")]
        else:
            predicted_jobs_list = []

        st.subheader("**Recommended Job Role (ML Model)**")
        if predicted_jobs_list:
            st.markdown(
                f"Predicted Job Role: {predicted_jobs_list[0]}",
                unsafe_allow_html=True
            )
        else:
            st.markdown("No job roles predicted.", unsafe_allow_html=True)

        # Recommended Additional Skills
        st.subheader("**Recommended Additional Skills (ML Model)**")
        all_skills = set(vectorized_skills)
        skills_to_learn = list(all_skills - set(resume_skills))
        num_skills_to_learn = min(len(skills_to_learn), 5)
        if skills_to_learn:
            skill_boxes = "".join(
                [f"✅ {skill.capitalize()}     " for skill in skills_to_learn[:num_skills_to_learn]]
            )
            st.markdown(skill_boxes, unsafe_allow_html=True)
        else:
            st.markdown("No additional skills recommended.", unsafe_allow_html=True)

        # YouTube Video Recommendations
        st.subheader("**YouTube Video Recommendations**")
        no_of_reco = st.slider("Choose Number of Video Recommendations:", 1, 5, 2)

        if skills_to_learn:
            videos = get_youtube_recommendations(skills_to_learn, no_of_reco)
            for skill, video in videos.items():
                st.markdown(
                    f"**{skill.capitalize()} ({video['channel']}):** [{video['title']}]({video['url']})",
                    unsafe_allow_html=True
                )
        else:
            st.markdown("No skills to recommend videos for.", unsafe_allow_html=True)

        # Course Recommendations
        st.subheader("**Courses & Certificates Recommendations 🎓**")
        no_of_reco = st.slider("Choose Number of Course Recommendations:", 1, 5, 2)

        all_courses = (
            ds_course + web_course + android_course + ios_course + uiux_course +
            mern_course + graphics_course + video_editing_course +
            digital_marketing_course + copywriting_course + sales_course +
            project_management_course + forex_trading_course + foreign_language_course +
            dropshipping_course + digital_marketing_course + funnel_design_course +
            influencer_marketing_course
        )
        recommended_courses = course_recommender(all_courses, no_of_reco)
        for i, (c_name, c_link) in enumerate(recommended_courses, start=1):
            st.markdown(f"🎓 [{c_name}]({c_link})", unsafe_allow_html=True)

    chatbot_button()

# Main Execution
if __name__ == "__main__":
    if st.session_state.page == "home":
        home_page()
