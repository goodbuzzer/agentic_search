import streamlit as st

# CSS personnalisé pour améliorer l'apparence
def load_css():
    st.markdown("""
        <style>
            /* Variables CSS pour la cohérence des couleurs */
            :root {
                --primary-color: #2E86AB;
                --secondary-color: #A23B72;
                --accent-color: #F18F01;
                --background-light: #F8F9FA;
                --text-dark: #2C3E50;
                --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --shadow-soft: 0 4px 6px rgba(0, 0, 0, 0.07);
                --shadow-medium: 0 8px 25px rgba(0, 0, 0, 0.1);
            }
            
            /* Import Google Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* Style global */
            .main {
                font-family: 'Inter', sans-serif !important;
            }
            
            /* Header principal avec gradient animé */
            .main-header {
                background: var(--gradient-primary);
                padding: 2rem;
                border-radius: 20px;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: var(--shadow-medium);
                position: relative;
                overflow: hidden;
            }
            
            .main-header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
                animation: shine 3s infinite;
            }
            
            @keyframes shine {
                0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
                100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
            }
            
            .main-title {
                color: white;
                font-size: 3.5rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                position: relative;
                z-index: 1;
            }
            
            .main-subtitle {
                color: rgba(255,255,255,0.9);
                font-size: 1.3rem;
                margin-top: 0.5rem;
                font-weight: 300;
                position: relative;
                z-index: 1;
            }
            
            /* Cards avec effets hover améliorés */
            .feature-card {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                margin-bottom: 1.5rem;
                box-shadow: var(--shadow-soft);
                border: 1px solid rgba(0,0,0,0.05);
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                position: relative;
                overflow: hidden;
            }
            
            .feature-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            }
            
            .feature-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 4px;
                background: var(--gradient-primary);
            }
            
            .feature-icon {
                font-size: 3rem;
                margin-bottom: 1rem;
                display: block;
            }
            
            .feature-title {
                font-size: 1.4rem;
                font-weight: 600;
                color: var(--text-dark);
                margin-bottom: 0.8rem;
            }
            
            .feature-description {
                font-size: 1rem;
                color: #6c757d;
                line-height: 1.6;
            }
            
            /* Chat interface styling */
            .chat-container {
                background: white;
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: var(--shadow-soft);
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background: var(--gradient-primary);
            }
            
            /* Buttons styling */
            .stButton > button {
                width: 100%;
                background: var(--gradient-primary);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.7rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: var(--shadow-soft);
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-medium);
            }
            
            /* Search results styling */
            .search-results-header {
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                border-radius: 15px;
                padding: 2rem;
                margin-bottom: 2rem;
                color: white;
                text-align: center;
                box-shadow: var(--shadow-medium);
            }
            
            /* Metrics cards */
            .metric-card {
                background: white;
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                box-shadow: var(--shadow-soft);
                border-left: 4px solid var(--primary-color);
            }
            
            /* Loading spinner customization */
            .stSpinner > div {
                border-top-color: var(--primary-color) !important;
            }
            
            /* DataFrame styling */
            .dataframe {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: var(--shadow-soft);
            }
            
            /* Alert boxes */
            .success-box {
                background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            
            .warning-box {
                background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
                color: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            
            /* Upload area styling */
            .upload-area {
                border: 2px dashed var(--primary-color);
                border-radius: 15px;
                padding: 2rem;
                text-align: center;
                background: rgba(46, 134, 171, 0.05);
                transition: all 0.3s ease;
            }
            
            .upload-area:hover {
                background: rgba(46, 134, 171, 0.1);
            }
            
            /* Footer */
            .footer {
                margin-top: 3rem;
                padding: 2rem;
                text-align: center;
                color: #6c757d;
                border-top: 1px solid #e9ecef;
            }
            
            /* Responsive design */
            @media (max-width: 768px) {
                .main-title {
                    font-size: 2.5rem;
                }
                
                .feature-card {
                    padding: 1.5rem;
                }
            }
        </style>
    """, unsafe_allow_html=True)