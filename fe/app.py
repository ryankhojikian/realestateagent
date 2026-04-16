import streamlit as st
import requests
import pandas as pd

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        padding: 10px;
        font-size: 16px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #2E8B57;
        text-align: center;
    }
    .hero-section {
        background: linear-gradient(135deg, #4CAF50, #2E8B57);
        color: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px;
        text-align: center;
    }
    .testimonial-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .stats-section {
        background-color: #2E8B57;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #666;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏠", layout="wide")

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1>🏠 Revolutionize Your Real Estate Decisions</h1>
    <p style="font-size: 20px;">Get accurate property valuations in seconds with our cutting-edge AI technology. Trusted by homeowners, agents, and investors worldwide.</p>
    <p style="font-size: 16px;">💡 <strong>Why Choose Us?</strong> Instant results, data-driven insights, and expert-level analysis at your fingertips.</p>
    <a href="#valuation" style="background-color: white; color: #4CAF50; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block; margin-top: 20px;">🚀 Get Valuation</a>
</div>
""", unsafe_allow_html=True)

# Key Features
st.header("🚀 Key Features")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>⚡ Instant Valuation</h3>
        <p>Get property prices in under 30 seconds with AI-powered analysis.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 Accurate Predictions</h3>
        <p>Trained on thousands of real estate transactions for reliable results.</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>📊 Detailed Insights</h3>
        <p>Receive comprehensive feature analysis and expert interpretation.</p>
    </div>
    """, unsafe_allow_html=True)

# Stats Section
st.markdown("""
<div class="stats-section">
    <h2>📈 Our Impact</h2>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
        <div><h3>10,000+</h3><p>Properties Analyzed</p></div>
        <div><h3>95%</h3><p>Accuracy Rate</p></div>
        <div><h3>24/7</h3><p>Available Anytime</p></div>
        <div><h3>50+</h3><p>Happy Customers</p></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Testimonials
st.header("💬 What Our Customers Say")
testimonials = [
    {"name": "Sarah Johnson", "role": "Homeowner", "text": "This AI agent gave me a spot-on valuation for my house. Saved me thousands compared to traditional appraisals!"},
    {"name": "Mike Chen", "role": "Real Estate Agent", "text": "Incredible tool for quick market analysis. My clients love the detailed reports."},
    {"name": "Emily Davis", "role": "Investor", "text": "Fast, accurate, and user-friendly. This has become my go-to for property evaluations."}
]

for testimonial in testimonials:
    st.markdown(f"""
    <div class="testimonial-card">
        <p>"{testimonial['text']}"</p>
        <p><strong>- {testimonial['name']}, {testimonial['role']}</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Why Choose Us
st.header("🌟 Why Choose AI Real Estate Agent?")
benefits = [
    {"icon": "⚡", "title": "Lightning Fast", "desc": "Get results in seconds, not days like traditional appraisals."},
    {"icon": "💰", "title": "Cost Effective", "desc": "Save hundreds on appraisal fees with our affordable AI solution."},
    {"icon": "🎯", "title": "Highly Accurate", "desc": "Powered by machine learning trained on real market data."},
    {"icon": "🔒", "title": "Secure & Private", "desc": "Your property information is encrypted and never shared."},
    {"icon": "📱", "title": "Easy to Use", "desc": "No complex forms - just describe your property in plain English."},
    {"icon": "🌍", "title": "Accessible Anywhere", "desc": "Use from any device, anywhere in the world, 24/7."}
]

cols = st.columns(3)
for i, benefit in enumerate(benefits):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card">
            <h3>{benefit['icon']} {benefit['title']}</h3>
            <p>{benefit['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# Header
st.markdown("<h1>🏠 AI Real Estate Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Get an instant valuation for your property using our advanced AI analysis.</p>", unsafe_allow_html=True)

# How It Works and Tips Section
st.header("ℹ️ How It Works & Tips")
how_cols = st.columns([1, 1])
with how_cols[0]:
    st.subheader("🚀 How It Works")
    st.markdown("""
    1. **Describe your property** in detail
    2. **Click 'Run AI Analysis'** to process
    3. **Get instant prediction** and expert review
    """)
with how_cols[1]:
    st.subheader("💡 Pro Tips")
    st.markdown("""
    - **Location matters**: Include neighborhood codes
    - **Be specific**: Mention bedrooms, bathrooms, quality
    - **Details help**: The more info, the better accuracy
    - **Examples**: "2-story house in CollgCr with excellent quality"
    """)

# Valuation Section Anchor
st.markdown('<div id="valuation"></div>', unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Property Description")
    user_input = st.text_area(
        "Enter your property details:",
        placeholder="e.g., A 2-story house in CollgCr with 3 bedrooms, 2 bathrooms, and good quality...",
        height=150,
        label_visibility="collapsed"
    )
    
    if st.button("🚀 Run AI Analysis", use_container_width=True):
        if user_input:
            with st.spinner("🤖 AI is analyzing your property..."):
                try:
                    res = requests.post("http://localhost:8080/agent", json={"description": user_input}).json()
                    data = res["extracted_data"]
                    
                    # Store results in session state for display in col2
                    st.session_state.results = res
                    st.session_state.data = data
                    st.success("✅ Analysis complete! Check the results on the right.")
                    
                except requests.exceptions.RequestException:
                    st.error("❌ Unable to connect to the backend. Please ensure 'main.py' is running.")
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
        else:
            st.error("⚠️ Please enter a property description.")

with col2:
    st.subheader("📊 Analysis Results")
    if 'results' in st.session_state:
        res = st.session_state.results
        data = st.session_state.data
        
        # Signal if features were missing
        if not data["is_complete"]:
            st.markdown(f"<div class='warning-box'>⚠️ <strong>Agent filled in missing details:</strong> {', '.join(data['missing_features'])}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='success-box'>✅ <strong>Full extraction successful!</strong></div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Predicted Price in a card
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("💰 Predicted Sale Price", f"${res['prediction']:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Agent Review
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown(f"**🤖 Agent Review:** {res['interpretation']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Technical Features
        with st.expander("🔧 Technical Feature Table"):
            st.table(pd.DataFrame([data["features"]]))
    else:
        st.info("👈 Enter a description and run analysis to see results here.")

# Footer
st.markdown("""
<div class="footer">
    <h3>🚀 Ready to Get Started?</h3>
    <p>Join thousands of satisfied customers who trust our AI for accurate property valuations.</p>
    <p><strong>Contact Us:</strong> support@airealestate.com | 📞 (555) 123-4567</p>
    <p>Follow us on: <a href="#" style="color: #4CAF50;">Facebook</a> | <a href="#" style="color: #4CAF50;">Twitter</a> | <a href="#" style="color: #4CAF50;">LinkedIn</a></p>
    <p style="font-size: 12px; margin-top: 20px;">© 2026 AI Real Estate Agent. All rights reserved. | Powered by AI | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)