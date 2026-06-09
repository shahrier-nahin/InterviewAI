import streamlit as st
import google.generativeai as genai
from serpapi import GoogleSearch
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os
import re

load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash-lite')
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")


# ========== PHASE 1: RESUME ANALYSIS ==========
def extract_resume_text(uploaded_file):
    """Extract text from uploaded PDF resume"""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def analyze_resume(resume_text):
    """Extract skills, experience, and key info from resume"""
    prompt = f"""
    Analyze this resume and extract:
    1. Technical skills (languages, frameworks, tools) - list only skills
    2. Years of experience (total)
    3. Current/previous job titles

    Resume:
    {resume_text[:4000]}

    Format your response as:
    SKILLS: skill1, skill2, skill3
    EXPERIENCE: X years
    TITLES: title1, title2
    """
    response = model.generate_content(prompt)
    return response.text


# ========== PHASE 2: JOB SEARCH & QUESTION GENERATION ==========
def search_job_requirements(job_title):
    """Fetch job requirements using SerpAPI"""
    try:
        search = GoogleSearch({
            "q": f"{job_title} job description requirements",
            "api_key": SERPAPI_KEY,
            "num": 2
        })
        results = search.get_dict()

        job_text = ""
        for result in results.get("organic_results", []):
            if result.get("snippet"):
                job_text += result["snippet"] + "\n"

        return job_text if job_text else "No job description found. Using generic requirements."
    except Exception as e:
        return f"Error fetching job: {str(e)}"


def generate_questions(resume_analysis, job_requirements, num_questions=5):
    """Generate interview questions based on resume + job requirements"""
    prompt = f"""
    You are a technical interviewer. Generate {num_questions} interview questions.

    Candidate's Resume Skills: {resume_analysis}
    Job Requirements: {job_requirements}

    Rules:
    - Create questions that test if candidate matches job requirements
    - Include questions that probe their claimed skills
    - Mix of technical and behavioral questions

    Return ONLY the questions, one per line, numbered 1 to {num_questions}.
    """
    response = model.generate_content(prompt)
    questions_text = response.text

    # Parse questions
    questions = []
    for line in questions_text.split('\n'):
        line = line.strip()
        if line and re.match(r'^\d+\.', line):
            # Remove the number prefix
            question = re.sub(r'^\d+\.\s*', '', line)
            questions.append(question)

    return questions[:num_questions]


# ========== PHASE 3: INTERVIEW & JUDGING ==========
def evaluate_answer(question, user_answer, resume_analysis):
    """Judge the answer and return score (1-10) with feedback"""
    prompt = f"""
    You are an interviewer evaluating a candidate's answer.

    Question: {question}

    Candidate's Answer: {user_answer}

    Candidate's Resume Claims: {resume_analysis}

    Evaluate on:
    - Relevance to question (0-4 points)
    - Specificity and clarity (0-3 points)  
    - Alignment with resume claims (0-3 points)

    Return EXACTLY this format:
    SCORE: X/10
    FEEDBACK: (1-2 line feedback)
    """
    response = model.generate_content(prompt)
    result_text = response.text

    # Extract score
    score_match = re.search(r'SCORE:\s*(\d+)', result_text)
    score = int(score_match.group(1)) if score_match else 5

    # Extract feedback
    feedback_match = re.search(r'FEEDBACK:\s*(.*?)(?:\n|$)', result_text, re.IGNORECASE)
    feedback = feedback_match.group(1) if feedback_match else "No feedback provided"

    return score, feedback


# ========== STREAMLIT UI ==========
st.set_page_config(page_title="InterviewAI", page_icon="🎯")
st.title("🎯 InterviewAI - Mock Interviewer")
st.markdown("---")

# Initialize session state
if 'phase' not in st.session_state:
    st.session_state.phase = "upload"  # upload, interview, complete
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'feedbacks' not in st.session_state:
    st.session_state.feedbacks = []


# ===== PHASE 1: Resume Upload & Analysis =====
if st.session_state.phase == "upload":
    st.subheader("📄 Step 1: Upload Your Resume")

    uploaded_file = st.file_uploader("Choose PDF file", type="pdf")

    if uploaded_file:
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_resume_text(uploaded_file)
            resume_analysis = analyze_resume(resume_text)

            # Store in session
            st.session_state.resume_analysis = resume_analysis
            st.session_state.resume_text = resume_text

            st.success("Resume analyzed successfully!")

            # Display analysis summary
            with st.expander("View Resume Analysis"):
                st.write(resume_analysis)

            st.subheader("💼 Step 2: Job Details")
            job_title = st.text_input("Which position are you applying for?",
                                      placeholder="e.g., Data Scientist, Full Stack Developer")

            if st.button("Start Interview Preparation"):
                if job_title:
                    with st.spinner("Fetching job requirements and generating questions..."):
                        # Fetch job requirements using SerpAPI
                        job_requirements = search_job_requirements(job_title)
                        st.session_state.job_requirements = job_requirements

                        # Generate questions
                        questions = generate_questions(resume_analysis, job_requirements)
                        st.session_state.questions = questions

                        # Move to interview phase
                        st.session_state.phase = "interview"
                        st.rerun()
                else:
                    st.warning("Please enter a job title")


# ===== PHASE 2: Conduct Interview =====
elif st.session_state.phase == "interview":
    st.subheader("🎤 Interview in Progress")

    # Show progress
    if len(st.session_state.questions) > 0:
        progress = st.session_state.current_q / len(st.session_state.questions)
        st.progress(progress)
        st.write(f"Question {st.session_state.current_q + 1} of {len(st.session_state.questions)}")

    # Show previous Q&A if any
    for i in range(st.session_state.current_q):
        with st.expander(f"Question {i + 1} - Score: {st.session_state.scores[i]}/10"):
            st.write(f"**Q:** {st.session_state.questions[i]}")
            st.write(f"**A:** {st.session_state.answers[i]}")
            st.write(f"**Feedback:** {st.session_state.feedbacks[i]}")

    # Current question
    if st.session_state.current_q < len(st.session_state.questions):
        current_question = st.session_state.questions[st.session_state.current_q]

        st.markdown(f"### Question {st.session_state.current_q + 1}")
        st.write(current_question)

        user_answer = st.text_area("Your answer:", height=150,
                                   placeholder="Type your answer here...")

        if st.button("Submit Answer"):
            if user_answer.strip():
                with st.spinner("Evaluating your answer..."):
                    score, feedback = evaluate_answer(
                        current_question,
                        user_answer,
                        st.session_state.resume_analysis
                    )

                    # Store
                    st.session_state.answers.append(user_answer)
                    st.session_state.scores.append(score)
                    st.session_state.feedbacks.append(feedback)
                    st.session_state.current_q += 1
                    st.rerun()
            else:
                st.warning("Please provide an answer")

    # Interview complete
    if st.session_state.current_q >= len(st.session_state.questions) and len(st.session_state.questions) > 0:
        st.session_state.phase = "complete"
        st.rerun()


# ===== PHASE 3: Results & Report Card =====
elif st.session_state.phase == "complete":
    st.subheader("📊 Interview Complete!")

    # Calculate total score
    total_score = sum(st.session_state.scores)
    max_possible = len(st.session_state.questions) * 10
    percentage = (total_score / max_possible) * 100

    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Score", f"{total_score}/{max_possible}")
    with col2:
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        avg_score = total_score / len(st.session_state.questions)
        st.metric("Average per Question", f"{avg_score:.1f}/10")

    st.markdown("---")

    # Report card
    st.subheader("📝 Detailed Report Card")

    for i, (question, answer, score, feedback) in enumerate(zip(
            st.session_state.questions,
            st.session_state.answers,
            st.session_state.scores,
            st.session_state.feedbacks
    )):
        with st.expander(f"Question {i + 1} - Score: {score}/10"):
            st.write(f"**Question:** {question}")
            st.write(f"**Your Answer:** {answer}")
            st.write(f"**Feedback:** {feedback}")

            # Visual indicator
            if score >= 8:
                st.success("Excellent! 🎯")
            elif score >= 6:
                st.info("Good 👍")
            else:
                st.warning("Needs improvement 📚")

    # Recommendation based on performance
    st.markdown("---")
    st.subheader("💡 Overall Recommendation")

    if percentage >= 80:
        st.success("Strong candidate! Ready for the actual interview.")
    elif percentage >= 60:
        st.info("Good potential. Review the weak areas mentioned above.")
    else:
        st.warning("Consider gaining more experience in the areas where scores were low.")

    if st.button("Start New Interview"):
        # Reset all session state
        keys_to_reset = ['phase', 'questions', 'current_q', 'answers', 'scores', 'feedbacks',
                         'resume_analysis', 'resume_text', 'job_requirements']
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.phase = "upload"
        st.rerun()