import os
import time
import json
import io
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
import datetime
import pandas as pd
import streamlit as st
import sqlite3
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq
from google import genai
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
# ---------------- DATABASE LAYER (SQLite) ----------------
DB_PATH = "candidates.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            filename TEXT,
            email TEXT,
            match_score INTEGER,
            key_strengths TEXT,
            weaknesses TEXT,
            quick_verdict TEXT,
            compatibility_reasons TEXT,
            rank INTEGER,
            is_top_candidate INTEGER DEFAULT 0,
            job_description TEXT,
            created_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_email TEXT,
            candidate_name TEXT,
            subject TEXT,
            status TEXT,
            template TEXT,
            sent_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_candidates_to_db(all_summaries: list, top_candidates: list, job_desc: str):
    """Persist every analyzed candidate (and their top-5 standing, if any) to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_lookup = {c.get("filename"): c for c in top_candidates}

    for summary in all_summaries:
        filename = summary.get("filename", "")
        top_match = top_lookup.get(filename)
        cursor.execute("""
            INSERT INTO candidates (
                name, filename, email, match_score, key_strengths,
                weaknesses, quick_verdict, compatibility_reasons, rank,
                is_top_candidate, job_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.get("name", ""),
            filename,
            summary.get("email", ""),
            summary.get("match_score", 0),
            json.dumps(summary.get("key_strengths", [])),
            json.dumps(summary.get("weaknesses", [])),
            summary.get("quick_verdict", ""),
            top_match.get("compatibility_reasons", "") if top_match else "",
            top_match.get("rank") if top_match else None,
            1 if top_match else 0,
            job_desc,
            now_str
        ))
    conn.commit()
    conn.close()

def log_email_to_db(candidate_email: str, candidate_name: str, subject: str, status: str, template: str, sent_at: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO email_log (candidate_email, candidate_name, subject, status, template, sent_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (candidate_email, candidate_name, subject, status, template, sent_at))
    conn.commit()
    conn.close()

def load_email_tracker_from_db() -> dict:
    """Rebuild the in-memory email_tracker from persisted logs on every app start."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT candidate_email, status, template, sent_at FROM email_log ORDER BY sent_at ASC")
    rows = cursor.fetchall()
    conn.close()
    return {email: {"status": status, "template": template, "timestamp": sent_at} for email, status, template, sent_at in rows}

def search_candidates_db(query: str = "", min_score: int = 0) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = """
        SELECT name, filename, email, match_score, quick_verdict, job_description, created_at
        FROM candidates
        WHERE match_score >= ?
    """
    params = [min_score]
    if query:
        like_query = f"%{query}%"
        sql += " AND (name LIKE ? OR email LIKE ? OR quick_verdict LIKE ? OR job_description LIKE ?)"
        params.extend([like_query, like_query, like_query, like_query])
    sql += " ORDER BY created_at DESC"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    columns = ["Name", "File Name", "Email", "Match Score", "Quick Verdict", "Job Description", "Analyzed On"]
    return pd.DataFrame(rows, columns=columns)

init_db()
if "email_tracker" not in st.session_state:
    st.session_state.email_tracker = load_email_tracker_from_db()
# -----------------------------------------------------------

# Page Configuration
st.set_page_config(page_title="AI Resume Leaderboard", page_icon="🏆", layout="wide")
st.title("🏆 AI Candidate Ranker & Excel Dashboard")
with st.expander("🔍 Search Past Candidates (Historical Database)"):
    st.write("Search across every candidate ever analyzed — for any job posting — without re-uploading resumes.")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input("Search by name, email, verdict, or job keyword:", key="db_search_query")
    with search_col2:
        min_score_filter = st.number_input("Min Match Score:", min_value=0, max_value=100, value=0, key="db_min_score")

    if st.button("🔎 Search Database", key="db_search_btn"):
        results_df = search_candidates_db(search_query, min_score_filter)
        if results_df.empty:
            st.info("No matching candidates found in the database.")
        else:
            st.dataframe(results_df, hide_index=True, use_container_width=True)
            st.caption(f"Found {len(results_df)} matching candidate record(s).")

st.divider()
# ---------------- SIDEBAR: SCORING WEIGHTS ----------------
st.sidebar.header("⚙️ Match Scoring Weights")
st.sidebar.caption("Tell the AI how much each factor should count toward the overall Match Score.")

weight_technical = st.sidebar.slider("Technical Skills Match", 0, 100, 40, key="weight_technical")
weight_experience = st.sidebar.slider("Experience Duration & Relevance", 0, 100, 35, key="weight_experience")
weight_education = st.sidebar.slider("Education & Certifications", 0, 100, 25, key="weight_education")

total_weight = weight_technical + weight_experience + weight_education

if total_weight == 0:
    st.sidebar.warning("At least one weight must be greater than 0.")
    norm_technical, norm_experience, norm_education = 0, 0, 0
else:
    norm_technical = round((weight_technical / total_weight) * 100)
    norm_experience = round((weight_experience / total_weight) * 100)
    norm_education = 100 - norm_technical - norm_experience  # guarantees exact sum to 100

    st.sidebar.caption(
        f"Normalized → Technical: **{norm_technical}%** | "
        f"Experience: **{norm_experience}%** | Education: **{norm_education}%**"
    )
# -----------------------------------------------------------

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {e}")
    return text.strip()

def clean_text(text: str) -> str:
    """Strip non-breaking spaces and other invisible unicode chars that break email headers."""
    if not text:
        return text
    return text.replace("\xa0", " ").replace("\u200b", "").strip()

def send_candidate_email(recipient_email: str, subject: str, body: str) -> tuple[bool, str]:
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return False, "SMTP credentials missing in .env!"

    recipient_email = clean_text(recipient_email)
    subject = clean_text(subject)
    body = clean_text(body)

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = Header(subject, "utf-8")
        msg.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        clean_password = SENDER_PASSWORD.replace(" ", "").replace("\xa0", "").strip()
        server.login(SENDER_EMAIL, clean_password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"Failed to send email: {e}"

def chunk_dict(d: dict, size: int):
    items = list(d.items())
    for i in range(0, len(items), size):
        yield dict(items[i:i + size])

def call_model_for_batch(formatted_cvs: str, job_description: str, weights: dict) -> list:
    """Sends ONE small batch of candidates to Groq (falls back to Gemini) and returns their summaries."""
    prompt = f"""
    You are an expert HR Talent Acquisition Specialist.
    Below are the raw extracted texts from a batch of candidate CVs, followed by the Target Job Description.

    ### Target Job Description:
    {job_description}

    ### Scoring Weight Configuration (you MUST use this to compute Match Score):
    - Technical Skills Match: {weights['technical']}%
    - Experience Duration & Relevance: {weights['experience']}%
    - Education & Certifications: {weights['education']}%

    For each candidate, first internally evaluate three sub-scores (each 0-100):
    technical_score, experience_score, and education_score. Then compute the
    final match_score as a weighted average using the percentages above:
    match_score = (technical_score * {weights['technical']} + experience_score * {weights['experience']} + education_score * {weights['education']}) / 100
    Round match_score to the nearest whole number.

    ### Candidates In This Batch:
    {formatted_cvs}

    ---
    For EVERY candidate above, create a concise summary including:
       - Candidate Name
       - Candidate Email (extract from resume if available, else leave as blank string "")
       - Match Score (0-100)
       - Top Key Strengths (3-4 bullet skills matching the job)
       - Weaknesses / Missing Skills (1-2 bullet points)
       - Quick Verdict (1 sentence summary)

    Return response strictly as valid JSON with this exact structure:
    {{
      "all_candidate_summaries": [
        {{
          "name": "Candidate Name",
          "filename": "Filename.pdf",
          "email": "candidate@email.com",
          "match_score": 85,
          "key_strengths": ["Python Development", "Streamlit UI"],
          "weaknesses": ["Lacks Cloud Deployment"],
          "quick_verdict": "Strong candidate."
        }}
      ]
    }}
    """

    if GROQ_API_KEY:
        active_groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        groq_client = Groq(api_key=GROQ_API_KEY)
        for g_model in active_groq_models:
            try:
                response = groq_client.chat.completions.create(
                    model=g_model,
                    messages=[
                        {"role": "system", "content": "You are an HR assistant that responds strictly in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content).get("all_candidate_summaries", [])
            except Exception as e:
                st.warning(f"Groq model `{g_model}` failed for this batch: {e}")
                continue

    if GEMINI_API_KEY:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        for attempt in range(3):
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"}
                )
                return json.loads(response.text).get("all_candidate_summaries", [])
            except Exception as e:
                st.warning(f"Gemini failed for this batch (attempt {attempt + 1}/3): {e}")
                time.sleep(3 * (attempt + 1))

    raise Exception("All model attempts failed for this batch across both Groq and Gemini.")

def analyze_and_rank_json(candidates_dict: dict, job_description: str, weights: dict, batch_size: int = 5) -> dict:
    """Processes candidates in small batches to stay under provider token limits, then ranks everyone locally."""
    all_summaries = []
    batches = list(chunk_dict(candidates_dict, batch_size))

    for i, batch in enumerate(batches, start=1):
        st.write(f"🔄 Processing batch {i} of {len(batches)} ({len(batch)} candidate(s))...")
        formatted_cvs = ""
        for filename, text in batch.items():
            truncated_text = text[:6000]  # cap a single résumé so it alone can't blow the token budget
            formatted_cvs += f"\n--- CANDIDATE FILE: {filename} ---\n{truncated_text}\n"

        batch_summaries = call_model_for_batch(formatted_cvs, job_description, weights)
        all_summaries.extend(batch_summaries)

        if i < len(batches):
            time.sleep(3)  # small pause so consecutive batches don't stack up against the per-minute token limit

    # Rank everyone locally -- no extra API call needed for this step
    sorted_summaries = sorted(all_summaries, key=lambda s: s.get("match_score", 0), reverse=True)
    top_5 = sorted_summaries[:5]

    top_candidates = []
    for rank, cand in enumerate(top_5, start=1):
        top_candidates.append({
            "rank": rank,
            "name": cand.get("name", ""),
            "filename": cand.get("filename", ""),
            "email": cand.get("email", ""),
            "match_score": cand.get("match_score", 0),
            "key_strengths": ", ".join(cand.get("key_strengths", [])),
            "compatibility_reasons": cand.get("quick_verdict", ""),
            "weaknesses": ", ".join(cand.get("weaknesses", [])),
        })

    return {
        "all_candidate_summaries": all_summaries,
        "top_candidates": top_candidates
    }

def generate_excel_bytes(candidates_data: list) -> bytes:
    df = pd.DataFrame(candidates_data).rename(columns={
        "rank": "Rank",
        "name": "Candidate Name",
        "filename": "File Name",
        "email": "Email Address",
        "match_score": "Match Score (%)",
        "key_strengths": "Key Strengths",
        "compatibility_reasons": "Why They Are Highly Compatible"
    })
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Top Candidates Leaderboard")
        worksheet = writer.sheets["Top Candidates Leaderboard"]
        
        font_family = "Aptos"
        header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
        header_font = Font(name=font_family, size=11, bold=True, color="FFFFFF")
        data_font = Font(name=font_family, size=10, color="000000")
        
        thin_border_side = Side(border_style="thin", color="D9D9D9")
        cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
        
        worksheet.row_dimensions[1].height = 26
        for col_num in range(1, len(df.columns) + 1):
            cell = worksheet.cell(row=1, column=col_num)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = cell_border
        
        for col_num, col_name in enumerate(df.columns, 1):
            col_letter = worksheet.cell(row=1, column=col_num).column_letter
            max_len = len(str(col_name))
            for row_num in range(2, len(df) + 2):
                val = str(worksheet.cell(row=row_num, column=col_num).value or '')
                max_len = max(max_len, len(val))
            
            if col_name in ["Why They Are Highly Compatible"]:
                calculated_width = max(max_len + 5, 55)
            elif col_name in ["Key Strengths"]:
                calculated_width = max(max_len + 5, 42)
            elif col_name in ["Candidate Name", "File Name", "Email Address"]:
                calculated_width = max(max_len + 6, 28)
            else:
                calculated_width = max(max_len + 6, 16)
                
            worksheet.column_dimensions[col_letter].width = calculated_width
            
            for row_num in range(2, len(df) + 2):
                cell = worksheet.cell(row=row_num, column=col_num)
                cell.font = data_font
                cell.border = cell_border
                worksheet.row_dimensions[row_num].height = 32
                
                if col_name in ["Rank", "Match Score (%)"]:
                    cell.alignment = Alignment(horizontal="center", vertical="top")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

    return output.getvalue()

# Streamlit UI
st.subheader("1. Target Job Description")
job_description = st.text_area("Paste job requirements here:", height=150)

st.subheader("2. Upload Candidate Resumes")
uploaded_files = st.file_uploader("Upload candidate PDFs:", type=["pdf"], accept_multiple_files=True)

if st.button("🚀 Analyze All Resumes", type="primary"):
    if not job_description.strip():
        st.warning("Please paste a job description first!")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume!")
    elif total_weight == 0:
        st.warning("Please set at least one scoring weight above 0 in the sidebar before analyzing.")
    else:
        candidates_text = {}
        with st.spinner("Extracting text from uploaded PDFs..."):
            for pdf in uploaded_files:
                raw = extract_text_from_pdf(pdf)
                if raw:
                    candidates_text[pdf.name] = raw

        if candidates_text:
            with st.spinner("Evaluating candidates with AI..."):
            
                try:
                    weights = {
                        "technical": norm_technical,
                        "experience": norm_experience,
                        "education": norm_education
                    }
                    result_json = analyze_and_rank_json(candidates_text, job_description, weights)
                    st.session_state.all_summaries = result_json.get("all_candidate_summaries", [])
                    st.session_state.top_candidates = result_json.get("top_candidates", [])
                    save_candidates_to_db(
                        st.session_state.all_summaries,
                        st.session_state.top_candidates,
                        job_description
                    )
                    st.success("✅ Results saved to database for future reference.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")
            

 
if "top_candidates" in st.session_state and st.session_state.top_candidates:
    st.divider()
    st.subheader("🏆 Top Candidate Leaderboard")
    
    # --- BULK OUTREACH SECTION ---
    with st.expander("⚡ Bulk Outreach: Send Interview Invitations to All Top Candidates"):
        st.write("Automatically dispatch interview invitations to all shortlisted candidates displaying valid emails.")
        bulk_subject = st.text_input("Bulk Subject:", value="Interview Invitation - Shortlisted Candidate", key="bulk_subj")
        
        default_bulk_body = (
            "Dear Candidate,\n\n"
            "We were very impressed by your background and credentials submitted for this role. "
            "We would like to invite you to an initial interview with our hiring team.\n\n"
            "Please let us know your availability over the upcoming week.\n\n"
            "Best regards,\n"
            "Talent Acquisition Team"
        )
         
        bulk_body = st.text_area("Bulk Message Body:", value=default_bulk_body, height=120, key="bulk_body")

        st.markdown("###### 🗓️ Interview Schedule (applies to all recipients)")
        bulk_col1, bulk_col2 = st.columns(2)
        with bulk_col1:
            bulk_interview_date = st.date_input("Interview Date:", key="bulk_date")
        with bulk_col2:
            bulk_interview_time = st.time_input("Interview Time:", key="bulk_time")

        bulk_interview_mode = st.radio("Interview Mode:", ["In-Person", "Online"], horizontal=True, key="bulk_mode")

        if bulk_interview_mode == "In-Person":
            bulk_location_value = st.text_input("Interview Location (address):", key="bulk_loc")
            bulk_location_line = f"📍 Location: {bulk_location_value}" if bulk_location_value else "📍 Location: (to be confirmed)"
        else:
            bulk_location_value = st.text_input("Meeting Link (Zoom / Google Meet):", key="bulk_link")
            bulk_location_line = f"💻 Join Link: {bulk_location_value}" if bulk_location_value else "💻 Join Link: (to be shared)"

        bulk_schedule_block = (
            f"\n\n📅 Date: {bulk_interview_date.strftime('%B %d, %Y')}"
            f"\n⏰ Time: {bulk_interview_time.strftime('%I:%M %p')}"
            f"\n{bulk_location_line}"
        )

        if st.button("🚀 Send Emails to ALL Top Candidates", type="primary"):
            sent_count = 0
            failed_count = 0
            
            with st.spinner("Dispatching bulk outreach emails..."):
                 
                for candidate in st.session_state.top_candidates:
                    cand_email = candidate.get("email", "").strip()
                    cand_name = candidate.get("name", "Candidate")
    
                    if not cand_email:
                        st.warning(f"Skipped {cand_name}: No email address extracted from resume.")
                        continue
        
                    if cand_email in st.session_state.email_tracker:
                        st.info(f"Skipped {cand_name} ({cand_email}): Email already sent previously.")
                        continue

                    # personalized_body = bulk_body.replace("Dear Candidate,", f"Dear {cand_name},")
                    personalized_body = bulk_body.replace("Dear Candidate,", f"Dear {cand_name},") + bulk_schedule_block
                    success, log_msg = send_candidate_email(cand_email, bulk_subject, personalized_body)
    
                     
                    if success:
                        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.email_tracker[cand_email] = {
                            "status": "Sent (Bulk)",
                            "timestamp": now_str,
                            "template": "Bulk Interview Invitation"
                        }
                        log_email_to_db(cand_email, cand_name, bulk_subject, "Sent (Bulk)", "Bulk Interview Invitation", now_str)
                        sent_count += 1
                    else:
                        failed_count += 1
                        st.error(f"Failed sending to {cand_name} ({cand_email}): {log_msg}")
                if sent_count > 0:
                    st.success(f"Successfully sent {sent_count} interview invitation email(s)!")
                if failed_count == 0 and sent_count > 0:
                    st.balloons()
                # --- BULK REJECTION SECTION ---
    with st.expander("❌ Bulk Rejection: Notify Candidates Who Didn't Make the Shortlist"):
        st.write("Send a personalized rejection email — explaining the specific gap — to every analyzed candidate who did NOT make the Top 5.")

        top_filenames = {c.get("filename") for c in st.session_state.top_candidates}
        rejected_candidates = [
            s for s in st.session_state.get("all_summaries", [])
            if s.get("filename") not in top_filenames
        ]

        st.caption(f"{len(rejected_candidates)} candidate(s) did not make the shortlist.")

        rejection_subject = st.text_input(
            "Rejection Subject:",
            value="Application Status Update - Thank You for Applying",
            key="bulk_reject_subj"
        )

        if st.button("📭 Send Rejection Emails to All Non-Shortlisted Candidates", type="secondary"):
            rej_sent = 0
            rej_failed = 0
            rej_skipped = 0

            with st.spinner("Sending personalized rejection emails..."):
                for candidate in rejected_candidates:
                    cand_email = candidate.get("email", "").strip()
                    cand_name = candidate.get("name", "Candidate")
                    weaknesses = candidate.get("weaknesses", [])

                    if not cand_email:
                        st.warning(f"Skipped {cand_name}: No email address extracted from resume.")
                        rej_skipped += 1
                        continue

                    if cand_email in st.session_state.email_tracker:
                        st.info(f"Skipped {cand_name} ({cand_email}): Email already sent previously.")
                        rej_skipped += 1
                        continue

                    if weaknesses:
                        reason_lines = "\n".join(f"- {w}" for w in weaknesses)
                        reason_block = (
                            "\n\nAfter careful review, we found the following areas where your "
                            "profile didn't fully align with what this role requires:\n"
                            f"{reason_lines}\n"
                        )
                    else:
                        reason_block = (
                            "\n\nWhile your background was strong, we ultimately moved forward "
                            "with candidates whose experience more closely matched this specific role.\n"
                        )

                    rejection_body = (
                        f"Dear {cand_name},\n\n"
                        f"Thank you for taking the time to apply and for sharing your background with us. "
                        f"After careful consideration, we have decided not to move forward with your application "
                        f"for this position at this time."
                        f"{reason_block}\n"
                        f"We genuinely appreciate the effort you put into your application and encourage you to "
                        f"apply again for future openings that match your skill set.\n\n"
                        f"Wishing you the best in your job search.\n\n"
                        f"Sincerely,\nTalent Acquisition Team"
                    )

                    success, log_msg = send_candidate_email(cand_email, rejection_subject, rejection_body)

                    if success:
                        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.email_tracker[cand_email] = {
                            "status": "Sent (Rejection)",
                            "timestamp": now_str,
                            "template": "Bulk Rejection Notice"
                        }
                        log_email_to_db(cand_email, cand_name, rejection_subject, "Sent (Rejection)", "Bulk Rejection Notice", now_str)
                        rej_sent += 1
                    else:
                        rej_failed += 1
                        st.error(f"Failed sending to {cand_name} ({cand_email}): {log_msg}")

            if rej_sent > 0:
                st.success(f"Successfully sent {rej_sent} rejection email(s).")
            if rej_skipped > 0:
                st.caption(f"{rej_skipped} candidate(s) skipped (no email on file, or already contacted).")       

    df_display = pd.DataFrame(st.session_state.top_candidates)
     
    st.table(
        df_display[["rank", "name", "filename", "match_score", "key_strengths", "weaknesses", "compatibility_reasons"]]
        .rename(columns={
            "rank": "Rank",
            "name": "Name",
            "filename": "File",
            "match_score": "Match Score",
            "key_strengths": "Key Strengths",
            "weaknesses": "Why Not Compatible",
            "compatibility_reasons": "Why Compatible",
        })
        .set_index("Rank")
    )

    excel_data = generate_excel_bytes(st.session_state.top_candidates)
    st.download_button(
        label="📥 Download Top Candidates Excel Report",
        data=excel_data,
        file_name="Top_Candidates_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()
    st.subheader("📋 Candidate Summaries & Outreach Actions")
    
     
    for idx, summary in enumerate(st.session_state.get("all_summaries", [])):
        cand_name = summary.get("name", "Candidate")
        cand_email = summary.get("email", "")
        cand_score = summary.get("match_score", 0)
        filename = summary.get("filename", "")

        # Check if we've already emailed this candidate
        sent_info = st.session_state.email_tracker.get(cand_email) if cand_email else None
        badge = " ✅ Already Emailed" if sent_info else ""

        with st.expander(f"📄 Candidate: {cand_name} ({filename}) — Score: {cand_score}%{badge}"):
            st.write(f"**Quick Verdict:** {summary.get('quick_verdict', '')}")

            if sent_info:
                st.info(f"📨 Already sent on **{sent_info['timestamp']}** — Template: *{sent_info['template']}*")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🌟 Key Strengths")
                for s in summary.get("key_strengths", []):
                    st.markdown(f"- {s}")
            with col2:
                st.markdown("##### ⚠️ Skill Gaps")
                for w in summary.get("weaknesses", []):
                    st.markdown(f"- {w}")

            st.markdown("---")
            st.markdown("##### ✉️ Direct Outreach")
            
            target_email = st.text_input("Recipient Email:", value=cand_email, key=f"email_{idx}")
            email_type = st.radio("Template:", ["Interview Invitation", "Rejection Notice", "Custom"], horizontal=True, key=f"tpl_{idx}")

            interview_details = ""
            if email_type == "Interview Invitation":
                st.markdown("###### 🗓️ Interview Schedule")
                sched_col1, sched_col2 = st.columns(2)
                with sched_col1:
                    interview_date = st.date_input("Interview Date:", key=f"date_{idx}")
                with sched_col2:
                    interview_time = st.time_input("Interview Time:", key=f"time_{idx}")

                interview_mode = st.radio("Interview Mode:", ["In-Person", "Online"], horizontal=True, key=f"mode_{idx}")

                if interview_mode == "In-Person":
                    location_value = st.text_input("Interview Location (address):", key=f"loc_{idx}")
                    location_line = f"📍 Location: {location_value}" if location_value else "📍 Location: (to be confirmed)"
                else:
                    location_value = st.text_input("Meeting Link (Zoom / Google Meet):", key=f"link_{idx}")
                    location_line = f"💻 Join Link: {location_value}" if location_value else "💻 Join Link: (to be shared)"

                interview_details = (
                    f"\n\n📅 Date: {interview_date.strftime('%B %d, %Y')}"
                    f"\n⏰ Time: {interview_time.strftime('%I:%M %p')}"
                    f"\n{location_line}"
                )

            if email_type == "Interview Invitation":
                subj = "Interview Invitation - Shortlisted for Position"
                body = (
                    f"Dear {cand_name},\n\n"
                    f"We were impressed by your application! We would love to schedule an initial interview with you."
                    f"{interview_details}\n\n"
                    f"Please let us know if this works for you or if you'd like to propose an alternate time.\n\n"
                    f"Best regards,\nTalent Acquisition Team"
                )
            elif email_type == "Rejection Notice":
                subj = f"Application Status Update - {cand_name}"
                body = f"Dear {cand_name},\n\nThank you for applying. We have decided to move forward with other candidates at this time.\n\nSincerely,\nTalent Acquisition Team"
            else:
                subj = f"Application Update - {cand_name}"
                body = f"Dear {cand_name},\n\n\n\nBest regards,\nTalent Acquisition Team"

            subject_val = st.text_input("Subject:", value=subj, key=f"subj_{idx}")
            body_val = st.text_area("Body:", value=body, height=120, key=f"body_{idx}")
            
              
            already_sent_to_current = target_email in st.session_state.email_tracker
            if already_sent_to_current:
                st.caption(f"⚠️ An email was already sent to {target_email}. Sending again will overwrite the log entry.")

            btn_label = f"📧 Resend Email to {cand_name}" if already_sent_to_current else f"📧 Send Email to {cand_name}"
            if st.button(btn_label, key=f"btn_{idx}"):
                if target_email:
                    success, msg = send_candidate_email(target_email, subject_val, body_val)
                     
                    if success:
                        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.email_tracker[target_email] = {
                            "status": f"Sent ({email_type})",
                            "timestamp": now_str,
                            "template": email_type
                        }
                        log_email_to_db(target_email, cand_name, subject_val, f"Sent ({email_type})", email_type, now_str)
                        st.success(f"{msg} — sent at {now_str}")
                        st.rerun()
                    else:
                        st.error(msg)