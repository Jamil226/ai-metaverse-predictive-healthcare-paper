import streamlit as st

# Page setup
st.set_page_config(page_title="Meta Doctor Pro", layout="wide")
st.markdown("<h1 style='text-align: center;'>Meta Doctor Pro</h1>", unsafe_allow_html=True)
st.caption("Beta")

# Main container
with st.container():
    st.subheader("Upload Medical Documents (Optional)")
    st.write("Drag and drop files here or click **Browse files**")
    uploaded_files = st.file_uploader("Upload your medical files", type=["pdf", "jpg", "png"], accept_multiple_files=True, label_visibility="collapsed")
    st.text("Limit 2000MB per file")

    # Display uploaded files in a grid
    if uploaded_files:
        st.write("### Uploaded Files")
        cols = st.columns(3)
        for idx, file in enumerate(uploaded_files):
            with cols[idx % 3]:
                st.write(f"📄 {file.name}")

# Patient Info Section
st.write("---")
with st.container():
    st.subheader("Patient Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", "Muhammad Jamil")
    with col2:
        email = st.text_input("Email", "jamil138.amin@gmail.com")

# Text Input Section
st.write("---")
st.subheader("Your Request")
user_request = st.text_area("",
    """I have uploaded my doctor’s notes from my recent visit, my past medicine prescription, and my latest CBC report. 
Please review these files, explain the findings in simple terms, point out any important changes compared to my previous treatment, 
and recommend a suitable treatment plan along with follow-up questions I should discuss with my doctor. 
Also, consider any potential medicine interactions or precautions based on my history before suggesting the treatment."""
)

# Analyze Button
if st.button("Analyze & Generate Response"):
    st.success("✅ Analysis complete.")
    st.markdown("### **Meta Doctor Response**")
    st.write("""
    **Summary:** Your CBC shows generally normal blood counts, but your hemoglobin is slightly low.
    
    **Recommendations:** Continue iron supplements for 4–6 weeks, add iron-rich foods, stay hydrated, avoid dairy near iron doses.
    
    **Follow-up Questions:**  
    - Should I repeat CBC after therapy?  
    - Can blood thinner be discontinued?  
    - Would vitamin C improve iron absorption?
    """)

# Recent Interactions Section
st.write("---")
st.subheader("Recent Interactions (08)")
interaction_1 = st.expander("📌 What does my CBC report show?")
interaction_1.write("Your CBC looks mostly normal, but HB is slightly low, suggesting mild anemia...")

interaction_2 = st.expander("📌 Are there any medicine risks I should know?")
interaction_2.write("Your iron supplements are generally safe, but they may interact with your past blood thinner...")

# Download Video Button
st.download_button("⬇ Download Video Response", "video_response.mp4", "Download")

