import streamlit as st
import numpy as np
import joblib
import spacy
import pandas as pd
import os
import io
import requests
from pdfminer.high_level import extract_text
import tensorflow as tf
from PIL import Image
import re
import subprocess
import importlib.util

# Load models
def install_spacy_model(model_name):
    if not importlib.util.find_spec(model_name):
        subprocess.run(
            ["python", "-m", "spacy", "download", model_name], check=True)

model_name = "en_core_web_sm"
install_spacy_model(model_name)
nlp = spacy.load(model_name)

# Load dataset
df = pd.read_csv("final_dataset.csv")  

# Extract all unique skills from dataset
all_skills = set()
for skills in df["Skills"].dropna():
    all_skills.update([s.strip().lower() for s in skills.split(",")])

# Home Page
def home_page():
    try:
        img = Image.open("logo2.png")
        st.image(img, use_container_width=True)
    except FileNotFoundError:
        st.error(
            "Logo file not found. Please ensure 'logo2.png' is placed in the './Logo/' folder."
        )
        st.stop()

# Chatbot Button
def chatbot_button():
    st.subheader("Need Further Assistance?")
    chatbot_link = "https://cdn.botpress.cloud/webchat/v2.2/shareable.html?configUrl=https://files.bpcontent.cloud/2025/02/04/08/20250204081537-I2WZDKEN.json"
    st.markdown(
        f'<a href="{chatbot_link}" target="_blank" style="text-decoration:none;"><button style="background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 5px; border: none;">Chat with our Bot</button></a>',
        unsafe_allow_html=True
    )

# Function to reset session state when a new resume is uploaded
def reset_session():
    st.session_state.clear()

# Function to reset jobs and everything after that when skills are edited
def reset_after_skills():
    keys_to_remove = ["recommended_jobs", "selected_job", "missing_skills", "youtube_videos", "num_videos"]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

# Function to reset steps after job selection
def reset_after_jobs():
    keys_to_remove = ["selected_job", "missing_skills", "youtube_videos", "num_videos"]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

# Function to process uploaded PDF
def process_uploaded_pdf(pdf_file):
    binary_buffer = io.BytesIO(pdf_file.read())
    resume_text = extract_text(binary_buffer)
    return resume_text

# Extract skills section from resume
def extract_skills_section(text):
    text = text.lower()
    skills_patterns = [r"\bskills\b[:\n]", r"\btechnical skills\b[:\n]", r"\bkey skills\b[:\n]"]
    start_index = None
    for pattern in skills_patterns:
        match = re.search(pattern, text)
        if match:
            start_index = match.end()
            break
    return text[start_index:].strip() if start_index else text

# Match extracted skills to dataset-based skills
def match_skills(skills_section, skill_list):
    return [skill for skill in skill_list if re.search(rf"\b{re.escape(skill)}\b", skills_section, re.IGNORECASE)]

# Extract name from resume
def extract_name(resume_text):
    lines = resume_text.split("\n")
    for line in lines[:5]:  
        words = line.split()
        if len(words) >= 2 and words[0][0].isupper() and words[1][0].isupper():
            return line.strip()
    return None

# Recommend 3 jobs using Student Model
def recommend_jobs(user_skills):
    user_vectorized = vectorizer.transform([" ".join(user_skills)]).toarray()
    job_probs = student_model.predict(user_vectorized)
    top_n_indices = np.argsort(job_probs[0])[::-1][:3]  
    top_jobs = encoder.inverse_transform(top_n_indices)
    return list(top_jobs)

# Recommend missing skills based on selected job
def find_missing_skills(job, user_skills):
    job_skills_list = df[df["Job"] == job]["Skills"].dropna().values
    if len(job_skills_list) == 0:
        return ["No skill data available"]
    
    job_skills = {skill.strip().lower() for skill in job_skills_list[0].split(",")}
    user_skills_set = {skill.strip().lower() for skill in user_skills}
    
    missing_skills = job_skills - user_skills_set
    return list(missing_skills)

# Hardcoded YouTube API Key
YOUTUBE_API_KEY = "AIzaSyDbvYU855ZHRzu4SmqZG9OpQKXDJNsOeU0"

# Get YouTube Video Recommendations
def get_youtube_recommendations(skills_to_learn, num_recommendations=3):
    recommendations = {}
    preferred_channels = ["freecodecamp", "Programming with Mosh", "Traversy Media", "CS Dojo"]
    
    for skill in skills_to_learn[:num_recommendations]:  
        query = f'{skill} tutorial for beginners ({" OR ".join(preferred_channels)})'
        url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&maxResults=1&type=video&key={YOUTUBE_API_KEY}"

        response = requests.get(url).json()
        if "items" in response and response["items"]:
            video_id = response["items"][0]["id"]["videoId"]
            video_title = response["items"][0]["snippet"]["title"]
            thumbnail_url = response["items"][0]["snippet"]["thumbnails"]["medium"]["url"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            recommendations[skill] = {
                "title": video_title, "url": video_url, "thumbnail": thumbnail_url
            }
    
    return recommendations


# Streamlit UI
home_page()
st.title("🚀 Smart Job Recommendation System")

# Step 1: Upload Resume (Reset results if new resume uploaded)
pdf_file = st.file_uploader("📄 Upload your Resume", type=["pdf"], on_change=reset_session)
if pdf_file:
    resume_text = process_uploaded_pdf(pdf_file)

    # Extract name & skills
    user_name = extract_name(resume_text)
    skills_section = extract_skills_section(resume_text)
    matched_skills = match_skills(skills_section, all_skills)  

    # Greeting Message
    if user_name:
        st.subheader(f"👋 Hello, {user_name}!")

    # Step 2: Let Users Modify Extracted Skills
    if matched_skills:
        st.subheader("✅ Extracted Skills (Edit if Needed):")
        selected_skills = st.multiselect("Modify Your Skills:", sorted(all_skills), default=matched_skills, on_change=reset_after_skills)

        if not selected_skills:
            st.warning("⚠️ Please select at least one skill to proceed.")
        else:
            # Step 3: Button to Recommend Jobs
            if st.button("🔍 Recommend Jobs"):
                reset_after_jobs()
                st.session_state.recommended_jobs = recommend_jobs(selected_skills)
                st.session_state.selected_skills = selected_skills  

# Step 4: Show Recommended Jobs in Blocks
if "recommended_jobs" in st.session_state:
    st.subheader("🎯 Recommended Jobs:")
    cols = st.columns(3)

    for i, job in enumerate(st.session_state.recommended_jobs):
        with cols[i]:
            if st.button(job, key=job):
                reset_after_jobs()
                st.session_state.selected_job = job  

    if "selected_job" in st.session_state:
        st.success(f"✔ Selected Job: {st.session_state.selected_job}")

        if "missing_skills" not in st.session_state:
            st.session_state.missing_skills = find_missing_skills(st.session_state.selected_job, st.session_state.selected_skills)

        st.subheader(f"📌 More Skills to Add for {st.session_state.selected_job}:")
        skill_html = " ".join([f"<span style='background-color:#ffeb99; padding:5px; border-radius:5px; margin:2px;'>{skill}</span>" for skill in st.session_state.missing_skills])
        st.markdown(f"<div style='padding:10px;'>{skill_html}</div>", unsafe_allow_html=True)

        # YouTube Video Slider
        num_videos = st.slider("🎥 Select Number of Video Recommendations:", 1, 5, 3)

        if "youtube_videos" not in st.session_state or st.session_state.num_videos != num_videos:
            st.session_state.youtube_videos = get_youtube_recommendations(st.session_state.missing_skills, num_videos)
            st.session_state.num_videos = num_videos  

        st.subheader("📺 YouTube Video Recommendations")
        if st.session_state.youtube_videos:
            # Display videos in a grid format
            cols = st.columns(len(st.session_state.youtube_videos))
            for i, (skill, video) in enumerate(st.session_state.youtube_videos.items()):
                with cols[i % len(cols)]:  # Use modulo to wrap around columns
                    st.markdown(f"[![{video['title']}]({video['thumbnail']})]({video['url']})", unsafe_allow_html=True)
                    st.markdown(f"**{skill.capitalize()} - [{video['title']}]({video['url']})**")
        else:
            st.warning("⚠️ No YouTube videos available.")

        # Chatbot Button
        chatbot_button()
