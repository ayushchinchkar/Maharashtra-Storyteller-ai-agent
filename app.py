import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import tempfile
import PyPDF2
import docx
import time
import requests
from typing import Optional
import re

# Load environment variables first
load_dotenv()

# --- Configuration Constants ---
GEMINI_TEXT_MODEL = "gemini-2.5-flash"
GROQ_TEXT_MODEL = "llama3-70b-8192"

# --- Apply user-provided configuration keys directly to environment for immediate use ---
# Only set environment variables if they exist and are not None
google_key = os.getenv('GOOGLE_API_KEY')
groq_key = os.getenv('GROQ_API_KEY')

if google_key:
    os.environ['GOOGLE_API_KEY'] = google_key
if groq_key:
    os.environ['GROQ_API_KEY'] = groq_key

# --- Main Class ---

class MaharashtraStoryteller:
    """
    Core class to handle API configurations, document processing, and
    LLM generation with fallback mechanisms, now focusing on storytelling and recommendations.
    """
    def __init__(self):
        self.setup_api_keys()
        self.uploaded_files_content = ""
        # Set the model names from the global constants
        self.gemini_text_model = GEMINI_TEXT_MODEL
        self.groq_text_model = GROQ_TEXT_MODEL
        
    def setup_api_keys(self):
        """Setup API keys from environment variables"""
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
            except Exception as e:
                st.error(f"Failed to configure Google API: {e}")
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text from uploaded files (PDF, DOCX, TXT)"""
        text_content = ""
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            if uploaded_file.name.endswith('.pdf'):
                with open(tmp_file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() or "" + "\n"
                        
            elif uploaded_file.name.endswith('.docx'):
                doc = docx.Document(tmp_file_path)
                for paragraph in doc.paragraphs:
                    text_content += paragraph.text + "\n"
                    
            elif uploaded_file.name.endswith('.txt'):
                with open(tmp_file_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
            
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {str(e)}")
            
        return text_content
    
    def process_uploaded_files(self, uploaded_files):
        """Process all uploaded files and extract text content"""
        all_content = ""
        if not uploaded_files:
            self.uploaded_files_content = ""
            return ""

        for uploaded_file in uploaded_files:
            content = self.extract_text_from_file(uploaded_file)
            all_content += f"\n--- Content from {uploaded_file.name} ---\n{content}\n"
        
        self.uploaded_files_content = all_content
        return all_content
    
    def generate_with_gemini(self, prompt: str) -> Optional[str]:
        """Generate story using the configured Gemini model."""
        if not self.google_api_key:
            return None
            
        try:
            model = genai.GenerativeModel(self.gemini_text_model)
            
            full_prompt = f"""
            You are a Maharashtra Lok Katha (folk story) storyteller. Tell stories in the traditional Maharashtrian storytelling style.
            
            Context from uploaded documents (if any):
            {self.uploaded_files_content}
            
            User request: {prompt}
            
            Please respond in English but maintain the cultural essence and storytelling style of Maharashtra Lok Katha.
            Include traditional elements, moral lessons, and the rich cultural heritage of Maharashtra.
            Make the story engaging and authentic.
            """
            
            response = model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            return None
    
    def generate_with_groq_direct(self, prompt: str) -> Optional[str]:
        """Generate story using Groq API directly via HTTP requests"""
        if not self.groq_api_key:
            return None
            
        try:
            full_prompt = f"""
            You are a Maharashtra Lok Katha (folk story) storyteller. Tell stories in the traditional Maharashtrian storytelling style.
            
            Context from uploaded documents (if any):
            {self.uploaded_files_content}
            
            User request: {prompt}
            
            Please respond in English but maintain the cultural essence and storytelling style of Maharashtra Lok Katha.
            Include traditional elements, moral lessons, and the rich cultural heritage of Maharashtra.
            Make the story engaging and authentic.
            """
            
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "messages": [{"role": "user", "content": full_prompt}],
                "model": self.groq_text_model,
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return None
            
    def generate_story(self, prompt: str) -> str:
        """Generate story with primary Gemini, and Groq fallback mechanism"""
        st.info("🔄 Weaving your Maharashtra folk tale...")
        
        story = None
        
        # 1. Try Gemini first (Primary)
        if self.google_api_key:
            story = self.generate_with_gemini(prompt)
            if story:
                st.success("✅ Generated successfully with Google Gemini.")
                return story
        
        # 2. If Gemini fails or API key is missing, try Groq (Fallback)
        if not story and self.groq_api_key:
            st.warning("⚠️ Gemini text generation unavailable. Trying Groq...")
            story = self.generate_with_groq_direct(prompt)
            if story:
                st.success("✅ Generated successfully with Groq.")
                return story

        # 3. If both fail, provide a sample story (Final Fallback)
        if not story:
            st.error("Both LLM APIs failed or are not configured. Displaying sample story.")
            return self.get_sample_story(prompt)
        
        return story

    def generate_recommendations(self, story: str) -> Optional[str]:
        """Generate book and video recommendations based on the story content."""
        if not self.google_api_key and not self.groq_api_key:
            return "Cannot generate recommendations as both Google and Groq APIs are unavailable."

        recommendation_prompt = f"""
        Based on the following Maharashtra Lok Katha (folk story) content, suggest:
        1. A list of 3 books (Title and Author/Topic) related to the themes, history, or culture mentioned in the story.
        2. A list of 3 YouTube video topics (Suggested Search Keywords) that would be relevant to the story's setting, characters, or moral.

        Story Content:
        ---
        {story}
        ---

        Format your output clearly using Markdown headings and numbered lists (e.g., '1. ...', '2. ...').
        """
        
        # Try Gemini first (Primary)
        if self.google_api_key:
            try:
                model = genai.GenerativeModel(self.gemini_text_model)
                with st.spinner("Searching for related books and videos..."):
                    response = model.generate_content(recommendation_prompt)
                st.success("✅ Recommendations generated with Google Gemini.")
                return response.text
            except Exception as e:
                st.warning(f"Gemini failed to generate recommendations: {str(e)}. Trying Groq...")

        # Try Groq (Fallback)
        if self.groq_api_key:
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "messages": [{"role": "user", "content": recommendation_prompt}],
                    "model": self.groq_text_model,
                    "temperature": 0.5,
                    "max_tokens": 512,
                    "stream": False
                }
                
                with st.spinner("Searching for related books and videos via Groq..."):
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                st.success("✅ Recommendations generated with Groq.")
                return result['choices'][0]['message']['content']
            except Exception as e:
                st.error(f"Groq also failed to generate recommendations: {str(e)}")
                return "Failed to generate recommendations using both APIs."
        
        return "APIs not configured or failed to respond."
    
    def get_sample_story(self, prompt: str) -> str:
        """Provide a sample story when APIs fail"""
        sample_stories = {
            "bravery": """
**The Brave Shepherd of Sinhagad**

In the shadow of the mighty Sinhagad fort, there lived a young shepherd named Shiva. Every day, he would graze his sheep on the lush green slopes, dreaming of the brave warriors who once defended the fort.

One monsoon evening, as dark clouds gathered, Shiva noticed something unusual - a group of suspicious men sneaking up the back trail of the fort. Remembering the stories his grandmother told him about Shivaji Maharaj's guards, he knew he had to act.

Instead of running away, clever Shiva came up with a plan. He began playing his flute loudly, mimicking the secret signals the fort guards used. Then he made his sheep bleat in a pattern that sounded like warning drums. The suspicious men, thinking the fort was alerted, retreated in confusion.

When the fort commander heard about this, he rewarded Shiva and said, "True bravery isn't just strength, but using wisdom to protect what you love." From that day, Shiva became known as the little guardian of Sinhagad.
""",
            "wisdom": """
**The Wise Potter of Pune**

In old Pune, there lived a potter named Gopal who was known for his beautiful clay pots. One day, a rich merchant came to him and ordered a thousand pots for his daughter's wedding. He paid half the money in advance.

Gopal worked day and night. But as he was carrying the finished pots to the merchant, his cart hit a stone and all the pots broke into pieces.

Instead of crying, Gopal collected the broken pieces and created beautiful mosaic artwork. When the merchant saw this, he was initially angry but then amazed at the creativity. The mosaic art became the talk of the town, and the merchant actually paid double the original price.

The moral: When life breaks your pots, make mosaic art. Every problem carries the seed of an opportunity for those wise enough to see it.
""",
            "mythological": """
**Vithoba's Test of Faith**

In Pandharpur, there lived a poor farmer named Tukaram who worshipped Lord Vithoba with all his heart. One year, there was a terrible drought, and Tukaram's crops failed. He had nothing to offer to the lord during the Ashadhi Ekadashi festival.

With a heavy heart, Tukaram went to the temple empty-handed. As he prayed, he noticed a hungry dog outside the temple. Without thinking, Tukaram took the only roti he had saved for himself and fed the dog.

That night, Lord Vithoba appeared in his dream and said, "Tukya, your roti to that hungry creature was the greatest offering I have ever received. True devotion is not in what you give me, but in how you treat my creations."

The next morning, Tukaram found his fields miraculously green with crops, teaching everyone that God resides in every living being.
"""
        }
        
        prompt_lower = prompt.lower()
        if "brave" in prompt_lower or "courage" in prompt_lower:
            return sample_stories["bravery"]
        elif "wisdom" in prompt_lower or "wise" in prompt_lower or "moral" in prompt_lower:
            return sample_stories["wisdom"]
        elif "god" in prompt_lower or "myth" in prompt_lower or "vithoba" in prompt_lower:
            return sample_stories["mythological"]
        else:
            return sample_stories["wisdom"]

def main():
    st.set_page_config(
        page_title="Maharashtra Lok Katha Storyteller",
        page_icon="📖",
        layout="wide"
    )
    
    # Custom CSS for Maharashtrian theme
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Teko:wght@400;700&family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="stApp"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-family: 'Teko', sans-serif;
        font-size: 3.5rem;
        color: #FF9933;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.25rem;
        color: #138808;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .story-container {
        background-color: #FFFDF5;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #FF993330;
        margin: 1rem 0;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    .recommendations-container {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .api-status {
        padding: 8px;
        border-radius: 6px;
        margin: 5px 0;
        font-weight: 600;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
    }
    .status-ok {
        background-color: #e6f7e9;
        color: #1a5824;
        border: 1px solid #c2ecc9;
    }
    .status-error {
        background-color: #fcebeb;
        color: #721c24;
        border: 1px solid #f9c9c9;
    }
    .stButton>button {
        border-radius: 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-header">📖 Maharashtra Lok Katha Storyteller</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">_Chala, aikoo ek sundar goshta!_ (Come, let\'s hear a beautiful story!)</div>', unsafe_allow_html=True)
    
    if 'storyteller' not in st.session_state:
        st.session_state.storyteller = MaharashtraStoryteller()
    
    storyteller = st.session_state.storyteller
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("🔧 Configuration & Context")
        
        # API Status
        st.subheader("API Status")
        google_status = f"✅ Configured ({storyteller.gemini_text_model})" if storyteller.google_api_key else "❌ Not Configured"
        groq_status = f"✅ Configured ({storyteller.groq_text_model})" if storyteller.groq_api_key else "❌ Not Configured"
        
        st.markdown(f'<div class="api-status {"status-ok" if storyteller.google_api_key else "status-error"}">Google Gemini (Text/Recommendations): {google_status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="api-status {"status-ok" if storyteller.groq_api_key else "status-error"}">Groq (Text Fallback): {groq_status}</div>', unsafe_allow_html=True)
        
        if not storyteller.google_api_key and not storyteller.groq_api_key:
            st.error("Please configure at least one API key (GOOGLE_API_KEY or GROQ_API_KEY) in the environment.")
            
        st.markdown("---")
        st.header("📚 Provide Context")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents to influence the story (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )
        
        # Process files
        current_file_names = {f.name for f in uploaded_files}
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != current_file_names:
            with st.spinner("Reading your documents..."):
                content = storyteller.process_uploaded_files(uploaded_files)
                if content:
                    st.success(f"📄 Processed {len(uploaded_files)} file(s).")
                st.session_state.last_uploaded_files = current_file_names
        
        if uploaded_files:
            content = storyteller.uploaded_files_content 
            if st.checkbox("Show document preview"):
                st.text_area("Document Content Preview", 
                             content[:1000] + "\n..." if len(content) > 1000 else content, 
                             height=200, 
                             disabled=True)
        else:
            if storyteller.uploaded_files_content != "":
                storyteller.uploaded_files_content = ""
                st.session_state.last_uploaded_files = set()

        st.markdown("---")
        st.header("About Lok Katha")
        st.info("The stories are crafted in the style of traditional Maharashtrian folk tales.")
        

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("✨ Create Your Lok Katha")
        
        if 'story_prompt' not in st.session_state:
            st.session_state.story_prompt = ""
            
        story_prompt = st.text_area(
            "What story would you like to hear?",
            value=st.session_state.story_prompt,
            placeholder="e.g., Tell me a story about bravery and wisdom from Maharashtra...",
            height=120,
            key="story_input_area"
        )
        
        def set_prompt(new_prompt):
            st.session_state.story_prompt = new_prompt
        
        # Quick story buttons
        col1_1, col1_2, col1_3 = st.columns(3)
        with col1_1:
            st.button("🧡 Bravery Story", on_click=set_prompt, args=("Tell me a brave story from Maharashtra about courage and honor",), use_container_width=True)
        with col1_2:
            st.button("💚 Wisdom Tale", on_click=set_prompt, args=("Share a wise folk tale from Maharashtra with a moral lesson about kindness",), use_container_width=True)
        with col1_3:
            st.button("💙 Mythological Story", on_click=set_prompt, args=("Tell me a mythological story from Maharashtra about the gods and legends of Pandharpur",), use_container_width=True)
        
        
        st.markdown("---")

        if st.button("📖 Generate Lok Katha", type="primary", use_container_width=True, key="generate_button"):
            final_prompt = st.session_state.story_prompt 
            
            if final_prompt:
                st.session_state.current_story = storyteller.generate_story(final_prompt)
                st.session_state.current_recommendations = None
            else:
                st.warning("Please enter a story prompt or choose a quick story theme.")
                st.session_state.current_story = None

        
        if 'current_story' in st.session_state and st.session_state.current_story:
            story = st.session_state.current_story
            
            st.markdown('<div class="story-container">', unsafe_allow_html=True)
            st.markdown("### 📜 Your Lok Katha")
            st.markdown(story)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Action buttons: Recommendations and Download
            btn_col1, btn_col2 = st.columns([1, 2])
            
            with btn_col1:
                if st.button("💡 Find Related Content", use_container_width=True, key="recommend_button"):
                    if storyteller.google_api_key or storyteller.groq_api_key:
                        st.session_state.current_recommendations = storyteller.generate_recommendations(story)
                    else:
                        st.error("Cannot generate recommendations without a configured API key.")

            with btn_col2:
                st.download_button(
                    "💾 Download Story as Text",
                    story,
                    file_name=f"maharashtra_lok_katha_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Recommendations Display
            if 'current_recommendations' in st.session_state and st.session_state.current_recommendations:
                recommendations_text = st.session_state.current_recommendations
                
                st.markdown('<div class="recommendations-container">', unsafe_allow_html=True)
                st.markdown("### 🔗 Further Exploration: Books & Videos")
                st.markdown(recommendations_text)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Extract and link the most similar YouTube match
                yt_section_match = re.search(r'(YouTube video topics|Suggested Search Keywords)', recommendations_text, re.IGNORECASE)
                
                first_yt_query = None
                if yt_section_match:
                    yt_block = recommendations_text[yt_section_match.end():]
                    first_item_match = re.search(r'[\*\-]\s*(.*)|[1]\.\s*(.*)', yt_block)
                    
                    if first_item_match:
                        query_raw = first_item_match.group(1) or first_item_match.group(2)
                        if query_raw:
                            first_yt_query = query_raw.strip().strip('*').strip('-').strip()
                
                if first_yt_query:
                    search_url = f"https://www.youtube.com/results?search_query={first_yt_query.replace(' ', '+')}"
                    
                    st.markdown("---")
                    st.subheader("Quick Link: Play Most Similar Match")
                    st.link_button(
                        f"▶️ Search YouTube for: '{first_yt_query}'",
                        search_url,
                        help="Opens the most relevant video search result on YouTube in a new tab.",
                        type="secondary",
                        use_container_width=True
                    )

    with col2:
        st.header("🎯 Tips & Elements")
        st.markdown("""
        To get the best folk tale, try to include these **Maharashtrian elements** in your prompt:
        
        - **Locations**: Pune, Mumbai, Nagpur, Kolhapur, Konkan region, Sinhagad fort.
        - **Historical figures**: Chhatrapati Shivaji Maharaj, Jijabai, Sant Dnyaneshwar, Sant Tukaram.
        - **Concepts**: Bhakti (devotion), Dharma (duty), Sanskar (values).
        - **Moral**: Specify the lesson you want the story to convey (e.g., kindness, honesty, perseverance).
        """)
        
        if storyteller.uploaded_files_content:
            st.warning(f"**Context Active!** The story will be influenced by the content of your uploaded document(s).")
        else:
            st.info("Upload PDF/DOCX/TXT files in the sidebar to provide context for your story.")

if __name__ == "__main__":
    main()


