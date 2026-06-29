import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline

# 1. Load your Hugging Face model and cache it so it runs fast
@st.cache_resource
def load_model():
    # This uses a standard toxicity model. 
    # If you used a different specific model before, you can change the name here.
    return pipeline("text-classification", model="unitary/toxic-bert")

classifier = load_model()

# 2. UI Layout
st.title("🚨 Full YouTube Comment Toxicity Analyzer")
st.write("Enter a YouTube link below to scrape and analyze **every available comment** on the video.")

video_url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

# Helper function to fetch ALL comments from the video link
def fetch_all_comments(url):
    downloader = YoutubeCommentDownloader()
    try:
        comments_iter = downloader.get_comments_from_url(url)
        # This reads every single comment until the end of the video stream
        return [c['text'] for c in comments_iter]
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return []

# 3. Analysis Execution
if st.button("Analyze Video"):
    if video_url:
        # Step A: Fetching the data
        with st.spinner("Fetching comments from YouTube... This can take a moment for large videos."):
            all_comments = fetch_all_comments(video_url)
            
        total_comments = len(all_comments)
        
        if total_comments == 0:
            st.error("No comments found. Please check if the video link is correct or if comments are turned off.")
        else:
            st.info(f"Found {total_comments} comments! Running AI toxicity analysis...")
            
            # Setup variables for tracking data
            toxic_count = 0
            non_toxic_count = 0
            
            # Step B: Set up a real-time progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step C: Loop through absolutely every comment found
            for index, text in enumerate(all_comments):
                try:
                    prediction = classifier(text)[0]
                    label = prediction['label'].lower()
                    score = prediction['score']
                    
                    # FIX THE BUG: Only mark as toxic if confidence score is higher than 50%
                    if 'toxic' in label and score > 0.50:
                        toxic_count += 1
                    else:
                        non_toxic_count += 1
                except Exception:
                    # Skips occasional problematic characters safely without breaking the loop
                    continue
                
                # Update progress bar every 5 comments
                if index % 5 == 0 or index == total_comments - 1:
                    progress_percentage = (index + 1) / total_comments
                    progress_bar.progress(progress_percentage)
                    status_text.text(f"Analyzed {index + 1} of {total_comments} comments...")

            # Clear status text once finished
            status_text.empty()
            st.success(f"Successfully completed analysis for all {total_comments} comments!")
            
            # 4. Dashboard Visuals (Side-by-side metric cards)
            col1, col2 = st.columns(2)
            col1.metric("🚨 Toxic Comments", toxic_count)
            col2.metric("😇 Non-Toxic Comments", non_toxic_count)
            
            # Create a simple visual bar chart comparison
            chart_data = {
                "Non-Toxic": non_toxic_count,
                "Toxic": toxic_count
            }
            st.bar_chart(chart_data)
            
    else:
        st.warning("Please paste a valid YouTube URL first.")
