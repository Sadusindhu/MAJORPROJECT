"""
SQL Injection Detection & Mitigation Streamlit App
Updated: Fixes TensorFlow 'reduction=auto' load error by loading CNN with compile=False,
and improves the UI and model-loading diagnostics.

To run:
    streamlit run sql_injection_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import os
import traceback

# Page config
st.set_page_config(
    page_title="SQL Injection Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for nicer UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.25rem;
        color: #162447;
        text-align: center;
        margin-bottom: 0.25rem;
        font-weight: 700;
    }
    .sub-header {
        color: #375a7f;
        margin-bottom: 0.75rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .safe-box {
        background: #27ae60;
        padding: 0.75rem;
        border-radius: 6px;
        color: white;
        font-weight: 700;
        text-align: center;
    }
    .malicious-box {
        background: #e74c3c;
        padding: 0.75rem;
        border-radius: 6px;
        color: white;
        font-weight: 700;
        text-align: center;
    }
    .small-muted {
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# Model paths - update them to point to your files
MODEL_PATHS = {
    'nb_pkl': r"C:\Users\JAYA SOORYA\Downloads\Data Leakage\naive_bayes_model.pkl",
    'nb_joblib': r"C:\Users\JAYA SOORYA\Downloads\Data Leakage\naive_bayes_model.joblib",
    'vec_pkl': r"C:\Users\JAYA SOORYA\Downloads\Data Leakage\vectorizer.pkl",
    'vec_joblib': r"C:\Users\JAYA SOORYA\Downloads\Data Leakage\vectorizer.joblib",
    'cnn': r"C:\Users\JAYA SOORYA\Downloads\Data Leakage\cnn_model.h5"
}


@st.cache_resource
def load_models():
    """
    Load saved models (Naive Bayes, Vectorizer, optional CNN).
    Fix for TensorFlow 'reduction=auto' error: load CNN with compile=False.
    Returns: (nb_clf, vectorizer, cnn_model, status_dict)
    status_dict contains messages about what was found/loaded.
    """
    status = {'nb': None, 'vec': None, 'cnn': None, 'errors': []}
    nb_clf = None
    vectorizer = None
    cnn_model = None

    try:
        # Load Naive Bayes classifier (try pickle then joblib)
        if os.path.exists(MODEL_PATHS['nb_pkl']):
            try:
                with open(MODEL_PATHS['nb_pkl'], 'rb') as f:
                    nb_clf = pickle.load(f)
                status['nb'] = f"Loaded from PKL: {MODEL_PATHS['nb_pkl']}"
            except Exception as e:
                status['errors'].append(f"Failed to load NB from pkl: {e}")
                nb_clf = None

        if nb_clf is None and os.path.exists(MODEL_PATHS['nb_joblib']):
            try:
                nb_clf = joblib.load(MODEL_PATHS['nb_joblib'])
                status['nb'] = f"Loaded from joblib: {MODEL_PATHS['nb_joblib']}"
            except Exception as e:
                status['errors'].append(f"Failed to load NB from joblib: {e}")
                nb_clf = None

        if nb_clf is None:
            status['errors'].append("Naive Bayes model not found or failed to load.")
        else:
            # sanity: ensure classifier has predict
            if not hasattr(nb_clf, "predict"):
                status['errors'].append("Loaded NB model does not have 'predict' method.")
        
        # Load vectorizer
        if os.path.exists(MODEL_PATHS['vec_pkl']):
            try:
                with open(MODEL_PATHS['vec_pkl'], 'rb') as f:
                    vectorizer = pickle.load(f)
                status['vec'] = f"Loaded from PKL: {MODEL_PATHS['vec_pkl']}"
            except Exception as e:
                status['errors'].append(f"Failed to load vectorizer from pkl: {e}")
                vectorizer = None

        if vectorizer is None and os.path.exists(MODEL_PATHS['vec_joblib']):
            try:
                vectorizer = joblib.load(MODEL_PATHS['vec_joblib'])
                status['vec'] = f"Loaded from joblib: {MODEL_PATHS['vec_joblib']}"
            except Exception as e:
                status['errors'].append(f"Failed to load vectorizer from joblib: {e}")
                vectorizer = None

        if vectorizer is None:
            status['errors'].append("Vectorizer not found or failed to load.")
        else:
            if not hasattr(vectorizer, "transform"):
                status['errors'].append("Loaded vectorizer does not have 'transform' method.")

        # Load CNN model (optional). Fix common TF load errors by using compile=False.
        if os.path.exists(MODEL_PATHS['cnn']):
            try:
                import tensorflow as tf
                # Load with compile=False to avoid errors related to optimizer/compile-time args
                cnn_model = tf.keras.models.load_model(MODEL_PATHS['cnn'], compile=False)
                status['cnn'] = f"Loaded CNN (compile=False): {MODEL_PATHS['cnn']}"
            except Exception as e:
                # Try to capture common 'reduction=auto' issue and give actionable message
                tb = traceback.format_exc()
                status['errors'].append(f"CNN found but failed to load: {e}")
                status['errors'].append("Traceback: " + tb.splitlines()[-1])
                cnn_model = None
        else:
            status['cnn'] = "No CNN model file found."

    except Exception as e:
        status['errors'].append(f"Unexpected error while loading models: {e}")
        tb = traceback.format_exc()
        status['errors'].append("Traceback: " + tb.splitlines()[-1])

    return nb_clf, vectorizer, cnn_model, status


# Initialize database
def initialize_database():
    """Create in-memory SQLite database with sample employee data."""
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        salary INTEGER,
        ssn TEXT,
        department TEXT
    )
    ''')

    employees = [
        ('John Admin', 'john@company.com', 150000, '123-45-6789', 'IT'),
        ('Jane Manager', 'jane@company.com', 120000, '234-56-7890', 'HR'),
        ('Bob Developer', 'bob@company.com', 95000, '345-67-8901', 'IT'),
        ('Alice Designer', 'alice@company.com', 85000, '456-78-9012', 'Marketing'),
        ('Charlie CEO', 'charlie@company.com', 250000, '567-89-0123', 'Executive'),
        ('Diana Engineer', 'diana@company.com', 110000, '678-90-1234', 'IT'),
        ('Eve Analyst', 'eve@company.com', 80000, '789-01-2345', 'Finance'),
        ('Frank Sales', 'frank@company.com', 90000, '890-12-3456', 'Sales')
    ]

    cursor.executemany('INSERT INTO employees (name, email, salary, ssn, department) VALUES (?,?,?,?,?)', employees)
    conn.commit()

    return conn


# Detection functions
def detect_sql_injection_ml(input_text, model, vectorizer):
    """ML-based detection. Returns (prediction_binary, confidence_percent).
    prediction_binary: 1 => malicious, 0 => safe
    """
    try:
        if model is None or vectorizer is None:
            return 0, 0.0

        # Transform input
        vect = vectorizer.transform([input_text])
        # Some vectorizers return sparse; convert to dense only if necessary
        try:
            input_vector = vect.toarray()
        except Exception:
            input_vector = np.asarray(vect)

        # Prediction
        pred = model.predict(input_vector)
        if isinstance(pred, (list, np.ndarray)):
            prediction = int(pred[0])
        else:
            prediction = int(pred)

        # Confidence: try predict_proba, fallback to decision_function scaled if available
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_vector)[0]
                confidence = float(np.max(proba) * 100)
            except Exception:
                confidence = 0.0
        elif hasattr(model, "decision_function"):
            try:
                score = model.decision_function(input_vector)
                # scale the score into 0-100 for display (rough)
                confidence = float((1.0 / (1.0 + np.exp(-score))) * 100)
                # handle array
                if np.ndim(confidence) > 0:
                    confidence = float(confidence[0])
            except Exception:
                confidence = 0.0
        else:
            confidence = 0.0

        return prediction, round(confidence, 2)

    except Exception as e:
        st.error(f"ML Detection Error: {e}")
        return 0, 0.0


def validate_input(input_text):
    """Rule-based validation: return (is_clean_bool, list_of_detected_patterns)"""
    suspicious_patterns = [
        (r"(\bOR\b|\bAND\b).*[=']", "SQL operators with quotes"),
        (r"(--|#|\/\*|\*\/)", "SQL comments"),
        (r"(\bUNION\b.*\bSELECT\b)", "UNION-based injection"),
        (r"(\bDROP\b|\bINSERT\b|\bDELETE\b|\bUPDATE\b|\bALTER\b|\bTRUNCATE\b)", "Dangerous SQL keywords"),
        (r"[;'\"]+.*(\bOR\b|\bAND\b)", "Quote manipulation"),
        (r"(\bEXEC\b|\bEXECUTE\b)", "Command execution"),
        (r"(xp_|sp_)", "System procedures"),
        (r"([0-9]{1,3}'\s*=\s*'[0-9]{1,3})", "Obfuscated boolean checks")
    ]

    detected_patterns = []
    for pattern, description in suspicious_patterns:
        if re.search(pattern, input_text, re.IGNORECASE):
            detected_patterns.append(description)

    return len(detected_patterns) == 0, detected_patterns


def vulnerable_query(search_term, conn):
    """Simulate vulnerable query (string concatenation)"""
    cursor = conn.cursor()
    query = f"SELECT * FROM employees WHERE name LIKE '%{search_term}%'"
    try:
        cursor.execute(query)
        results = cursor.fetchall()
        return results, query
    except Exception as e:
        return [], f"Query: {query}\nError: {str(e)}"


def secure_query(search_term, conn):
    """Parameterized query to prevent injection"""
    cursor = conn.cursor()
    query = "SELECT * FROM employees WHERE name LIKE ?"
    try:
        cursor.execute(query, (f'%{search_term}%',))
        results = cursor.fetchall()
        return results, query + f" [Parameter: '%{search_term}%']"
    except Exception as e:
        return [], str(e)


# Main app
def main():
    st.markdown('<h1 class="main-header">🔒 SQL Injection Detection & Mitigation System</h1>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Multi-layer demo: ML + rule validation + parameterized queries</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar information & controls
    st.sidebar.title("⚙️ Configuration & Models")
    with st.sidebar.expander("Model Controls", expanded=True):
        st.write("Use the controls below to load/reload models and view status.")
        if st.button("🔄 Reload models (clear cache)", use_container_width=True):
            try:
                # Clear resource cache then rerun — clears all cached resources
                try:
                    st.cache_resource.clear()
                except Exception:
                    # Older Streamlit versions may not have cache_resource.clear; try experimental
                    try:
                        st.experimental_singleton.clear()
                    except Exception:
                        pass
                st.experimental_rerun()
            except Exception:
                st.experimental_rerun()

    # Load models (cached)
    with st.spinner("Loading models..."):
        nb_clf, vectorizer, cnn_model, status = load_models()

    # Show model status
    st.sidebar.markdown("### Model Load Status")
    if status['nb']:
        st.sidebar.success(f"NB: {status['nb']}")
    else:
        st.sidebar.error("NB: Not loaded")

    if status['vec']:
        st.sidebar.success(f"Vectorizer: {status['vec']}")
    else:
        st.sidebar.error("Vectorizer: Not loaded")

    if status['cnn'] and "Loaded" in str(status['cnn']):
        st.sidebar.success(f"CNN: {status['cnn']}")
    else:
        # If CNN not found or failed to load, show warning and last error
        if status['cnn'] == "No CNN model file found.":
            st.sidebar.info("CNN: Not provided (optional)")
        else:
            st.sidebar.warning("CNN: Failed to load (optional). See details below.")

    if status['errors']:
        with st.sidebar.expander("🛠️ Load diagnostics & errors", expanded=False):
            for err in status['errors']:
                st.text(err)

    # If core models are missing, show clear message but keep app usable for demo queries
    core_missing = (nb_clf is None or vectorizer is None)
    if core_missing:
        st.warning("⚠️ ML model or vectorizer not loaded. ML detection will be unavailable. You can still use rule-based checks and demo features below.")

    # Initialize database
    if not st.session_state.db_initialized:
        st.session_state.conn = initialize_database()
        st.session_state.db_initialized = True

    # Main navigation
    mode = st.radio(
        "Choose Mode",
        ["🔍 Real-time Detection", "📊 Vulnerability Demo", "📈 Analytics Dashboard", "💾 Database Manager"],
        index=0,
        horizontal=True
    )

    if mode == "🔍 Real-time Detection":
        st.header("Real-time SQL Injection Detection")
        st.write("Analyze single queries or upload a CSV of queries for batch analysis.")

        col_left, col_right = st.columns([2, 1])

        with col_left:
            input_method = st.radio("Input Method", ["💬 Text Input", "📁 File Upload"])

            if input_method == "💬 Text Input":
                user_input = st.text_area(
                    "Enter query or user input to analyze:",
                    placeholder="e.g., admin' OR '1'='1",
                    height=140,
                    help="This input will be checked by ML (if available) and rule-based validators."
                )

                btn_col1, btn_col2 = st.columns([1, 1])
                with btn_col1:
                    analyze_btn = st.button("🔍 Analyze Query", type="primary", use_container_width=True)
                with btn_col2:
                    clear_btn = st.button("🗑 Clear Input", use_container_width=True)

                if analyze_btn:
                    if user_input and (nb_clf is not None and vectorizer is not None):
                        analyze_query(user_input, nb_clf, vectorizer)
                    elif user_input:
                        # ML missing: run rule-based only
                        analyze_query(user_input, nb_clf, vectorizer)
                    else:
                        st.warning("⚠️ Please enter a query to analyze.")

                if clear_btn:
                    # simple clear by rerunning
                    st.experimental_rerun()

            else:
                uploaded_file = st.file_uploader("Upload CSV file with queries (first row as header)", type=['csv'])
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded file:")
                    st.dataframe(df.head(), use_container_width=True)

                    col_name = st.selectbox("Select the column containing queries:", df.columns)
                    if st.button("🔍 Analyze All", type="primary"):
                        analyze_batch(df, col_name, nb_clf, vectorizer)

        with col_right:
            st.subheader("🧪 Quick Test Samples")
            test_cases = {
                "✅ Safe Query": "John Admin",
                "⚠️ Basic OR Injection": "' OR '1'='1",
                "⚠️ Union Attack": "' UNION SELECT * FROM employees--",
                "⚠️ Comment Attack": "admin'--",
                "⚠️ Stacked Query": "'; DROP TABLE employees--",
                "⚠️ Boolean-based": "' OR 1=1#"
            }
            for label, sample in test_cases.items():
                if st.button(label, use_container_width=True):
                    analyze_query(sample, nb_clf, vectorizer)

    elif mode == "📊 Vulnerability Demo":
        st.header("Vulnerability Demonstration")
        st.info("Showcases vulnerable string-concatenation queries vs multi-layer defenses.")

        tab1, tab2, tab3 = st.tabs(["🚨 Vulnerable System", "🔒 Secure System", "📊 Comparison"])

        with tab1:
            st.subheader("🚨 Vulnerable Implementation (String Concatenation)")
            st.code("query = f\"SELECT * FROM employees WHERE name LIKE '%{user_input}%'\"", language="python")
            vuln_input = st.text_input("Enter search term (vulnerable):", "John", key="vuln_input")
            if st.button("▶️ Execute Vulnerable", key="vuln_exec"):
                results, query = vulnerable_query(vuln_input, st.session_state.conn)
                st.markdown("**Executed Query:**")
                st.code(query, language="sql")
                if results:
                    df_results = pd.DataFrame(results, columns=['ID', 'Name', 'Email', 'Salary', 'SSN', 'Department'])
                    st.error(f"🚨 DATA LEAK: {len(results)} records exposed!")
                    st.dataframe(df_results, use_container_width=True)
                    if len(results) > 1:
                        st.warning(f"💰 **Total salaries exposed:** ${df_results['Salary'].sum():,}")
                        st.warning(f"🔐 **SSN records leaked:** {len(df_results)}")
                else:
                    st.info("No results or query failed.")

        with tab2:
            st.subheader("🔒 Secure Implementation (Multi-Layer Defense)")
            st.code("cursor.execute('SELECT * FROM employees WHERE name LIKE ?', (search_term,))", language="python")
            secure_input = st.text_input("Enter search term (secure):", "John", key="secure_input")
            if st.button("▶️ Execute Secure", key="secure_exec"):
                with st.spinner("Analyzing input across defense layers..."):
                    # Layer 1: ML Detection (if available)
                    ml_pred, confidence = detect_sql_injection_ml(secure_input, nb_clf, vectorizer)
                    st.markdown("**🛡️ Security Layers:**")
                    if nb_clf is not None and vectorizer is not None:
                        if ml_pred == 1:
                            st.error(f"❌ Layer 1 - ML Detection: Blocked (Confidence: {confidence:.1f}%)")
                            # log to history
                            st.session_state.detection_history.append({
                                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'Query': secure_input,
                                'Prediction': 'Malicious',
                                'Confidence': f"{confidence:.2f}%",
                                'Patterns': 'ML detected'
                            })
                            st.stop()
                        else:
                            st.success(f"✅ Layer 1 - ML Detection: Passed (Confidence: {confidence:.1f}%)")
                    else:
                        st.info("ℹ️ ML model unavailable, skipping Layer 1.")

                    # Layer 2: Rule Validation
                    is_valid, patterns = validate_input(secure_input)
                    if not is_valid:
                        st.error(f"❌ Layer 2 - Rule Validation: Blocked - Detected: {', '.join(patterns)}")
                        # log to history
                        st.session_state.detection_history.append({
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Query': secure_input,
                            'Prediction': 'Malicious (Rule)',
                            'Confidence': 'N/A',
                            'Patterns': ', '.join(patterns)
                        })
                        st.stop()
                    else:
                        st.success("✅ Layer 2 - Rule Validation: Passed")

                    # Layer 3: Parameterized Query
                    results, query = secure_query(secure_input, st.session_state.conn)
                    st.success("✅ Layer 3 - Parameterized Query: Executed safely")
                    st.markdown("**Executed Query:**")
                    st.code(query, language="sql")
                    if results:
                        df_results = pd.DataFrame(results, columns=['ID', 'Name', 'Email', 'Salary', 'SSN', 'Department'])
                        st.success(f"✅ Legitimate query: {len(results)} record(s) returned")
                        st.dataframe(df_results, use_container_width=True)
                    else:
                        st.info("No results found.")

        with tab3:
            st.subheader("📊 Comparison Analysis")
            attack_payloads = [
                ("Normal Query", "John"),
                ("OR-based Injection", "' OR '1'='1"),
                ("OR 1=1 Injection", "' OR 1=1--"),
                ("UNION Attack", "' UNION SELECT * FROM employees--"),
                ("DROP TABLE Attack", "'; DROP TABLE employees--"),
                ("Comment Bypass", "admin'#")
            ]

            comparison_data = []
            with st.spinner("Testing payloads..."):
                for attack_name, payload in attack_payloads:
                    vuln_results, _ = vulnerable_query(payload, st.session_state.conn)
                    ml_pred, confidence = detect_sql_injection_ml(payload, nb_clf, vectorizer)
                    is_valid, patterns = validate_input(payload)
                    secure_blocked = False
                    if nb_clf is not None and vectorizer is not None and ml_pred == 1:
                        secure_blocked = True
                    if not is_valid:
                        secure_blocked = True

                    comparison_data.append({
                        'Attack Type': attack_name,
                        'Payload': payload if len(payload) <= 80 else payload[:80] + '...',
                        'Vulnerable (Records Leaked)': len(vuln_results),
                        'Secure (Records Leaked)': 0 if secure_blocked else len(vuln_results),
                        'Status': '🛡️ Blocked' if secure_blocked else '⚠️ Passed'
                    })

            df_comp = pd.DataFrame(comparison_data)
            st.dataframe(df_comp, use_container_width=True)

            # Visualization
            fig = go.Figure(data=[
                go.Bar(name='Vulnerable', x=df_comp['Attack Type'], y=df_comp['Vulnerable (Records Leaked)'],
                       marker_color='#e74c3c', text=df_comp['Vulnerable (Records Leaked)']),
                go.Bar(name='Secure', x=df_comp['Attack Type'], y=df_comp['Secure (Records Leaked)'],
                       marker_color='#27ae60', text=df_comp['Secure (Records Leaked)'])
            ])
            fig.update_layout(title='Records Leaked: Vulnerable vs Secure System',
                              xaxis_title='Attack Type', yaxis_title='Records Exposed', barmode='group', height=520)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                total_vuln = df_comp['Vulnerable (Records Leaked)'].sum()
                st.metric("Total Vulnerable Leaks", total_vuln)
            with col2:
                total_secure = df_comp['Secure (Records Leaked)'].sum()
                st.metric("Total Secure Leaks", total_secure)
            with col3:
                prevention_rate = 100 - (total_secure / total_vuln * 100) if total_vuln > 0 else 100
                st.metric("Prevention Rate", f"{prevention_rate:.1f}%")

    elif mode == "📈 Analytics Dashboard":
        st.header("Analytics Dashboard")
        st.write("Historical detection results aggregated from current session.")

        if st.session_state.detection_history:
            df_history = pd.DataFrame(st.session_state.detection_history)
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Queries", len(df_history))
            with col2:
                malicious_count = (df_history['Prediction'].str.contains('Malicious')).sum()
                st.metric("🚨 Malicious Detected", malicious_count)
            with col3:
                safe_count = len(df_history) - malicious_count
                st.metric("✅ Safe Queries", safe_count)
            with col4:
                detection_rate = (malicious_count / len(df_history) * 100) if len(df_history) > 0 else 0
                st.metric("🎯 Detection Rate", f"{detection_rate:.1f}%")

            st.markdown("---")
            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(df_history, names='Prediction', title='Query Distribution',
                                color_discrete_map={'Safe': '#27ae60', 'Malicious': '#e74c3c'}, hole=0.4)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                # Confidence numeric
                def parse_conf(c):
                    try:
                        if c is None:
                            return np.nan
                        return float(str(c).rstrip('%'))
                    except Exception:
                        return np.nan
                df_history['Confidence_Numeric'] = df_history['Confidence'].apply(parse_conf)
                fig_bar = px.histogram(df_history, x='Confidence_Numeric', color='Prediction',
                                      title='Confidence Distribution',
                                      color_discrete_map={'Safe': '#27ae60', 'Malicious': '#e74c3c'},
                                      nbins=20)
                fig_bar.update_layout(xaxis_title='Confidence (%)', yaxis_title='Count')
                st.plotly_chart(fig_bar, use_container_width=True)

            # Timeline
            st.subheader("Detection Timeline")
            df_history['Timestamp_dt'] = pd.to_datetime(df_history['Timestamp'])
            timeline_counts = df_history.groupby([df_history['Timestamp_dt'].dt.floor('T'), 'Prediction']).size().reset_index(name='Count')
            if not timeline_counts.empty:
                fig_timeline = px.line(timeline_counts, x='Timestamp_dt', y='Count', color='Prediction',
                                      color_discrete_map={'Safe': '#27ae60', 'Malicious': '#e74c3c'})
                fig_timeline.update_layout(xaxis_title='Time', yaxis_title='Queries')
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No timeline data yet.")

            # Recent history
            st.subheader("Recent Detection History")
            display_df = df_history[['Timestamp', 'Query', 'Prediction', 'Confidence', 'Patterns']].tail(50).sort_index(ascending=False)
            st.dataframe(display_df, use_container_width=True)

            # Export and clear
            col1, col2 = st.columns([1, 1])
            with col1:
                csv = df_history.to_csv(index=False)
                st.download_button("📥 Download CSV", csv, "detection_history.csv", "text/csv")
            with col2:
                if st.button("🗑 Clear History"):
                    st.session_state.detection_history = []
                    st.experimental_rerun()
        else:
            st.info("📊 No detection history yet. Analyze some queries to populate the dashboard.")
            st.image("https://via.placeholder.com/800x300?text=Run+an+analysis+to+see+analytics", use_container_width=True)

    else:  # Database Manager
        st.header("💾 Database Manager")
        st.info("Manage the simulated employee database (in-memory). For demo purposes only.")

        tab1, tab2, tab3 = st.tabs(["👥 View Data", "➕ Add Records", "⚙️ Custom Query"])

        with tab1:
            st.subheader("Current Database Contents")
            cursor = st.session_state.conn.cursor()
            cursor.execute("SELECT * FROM employees")
            results = cursor.fetchall()
            df_db = pd.DataFrame(results, columns=['ID', 'Name', 'Email', 'Salary', 'SSN', 'Department'])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Employees", len(df_db))
            with col2:
                st.metric("Departments", df_db['Department'].nunique())
            with col3:
                st.metric("Avg Salary", f"${df_db['Salary'].mean():,.0f}")
            with col4:
                st.metric("Total Payroll", f"${df_db['Salary'].sum():,.0f}")

            st.dataframe(df_db, use_container_width=True)

            st.subheader("Department Breakdown")
            dept_summary = df_db.groupby('Department').agg({
                'ID': 'count',
                'Salary': ['mean', 'sum']
            }).round(0)
            dept_summary.columns = ['Employees', 'Avg Salary', 'Total Salary']
            st.dataframe(dept_summary, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download Database CSV", df_db.to_csv(index=False), "database_export.csv", "text/csv", use_container_width=True)
            with col2:
                if st.button("🔄 Reset Database", use_container_width=True):
                    st.session_state.conn = initialize_database()
                    st.success("✅ Database reset!")
                    st.experimental_rerun()

        with tab2:
            st.subheader("Add New Employee")
            with st.form("add_employee_form"):
                c1, c2 = st.columns(2)
                with c1:
                    name = st.text_input("Name *", placeholder="John Doe")
                    email = st.text_input("Email *", placeholder="john@company.com")
                    department = st.selectbox("Department *", ['IT', 'HR', 'Finance', 'Sales', 'Marketing', 'Executive'])
                with c2:
                    salary = st.number_input("Salary *", min_value=0, step=1000, value=80000)
                    ssn = st.text_input("SSN *", placeholder="123-45-6789")
                submitted = st.form_submit_button("➕ Add Employee")
                if submitted:
                    if name and email and ssn:
                        try:
                            cursor = st.session_state.conn.cursor()
                            cursor.execute("INSERT INTO employees (name, email, salary, ssn, department) VALUES (?,?,?,?,?)",
                                           (name, email, salary, ssn, department))
                            st.session_state.conn.commit()
                            st.success(f"✅ Employee '{name}' added.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error adding employee: {e}")
                    else:
                        st.warning("Please fill all required fields.")

        with tab3:
            st.subheader("Execute Custom SQL Query")
            st.warning("⚠️ This executes queries directly against the demo DB. Do not use with production data.")
            query_templates = {
                "Select All": "SELECT * FROM employees",
                "Count by Department": "SELECT department, COUNT(*) as count FROM employees GROUP BY department",
                "High Earners": "SELECT * FROM employees WHERE salary > 100000 ORDER BY salary DESC",
                "IT Department": "SELECT * FROM employees WHERE department = 'IT'"
            }

            template = st.selectbox("Choose template:", ["Custom"] + list(query_templates.keys()))
            if template == "Custom":
                custom_query = st.text_area("Enter SQL query:", "SELECT * FROM employees", height=100)
            else:
                custom_query = st.text_area("Enter SQL query:", query_templates[template], height=100)

            if st.button("▶️ Execute Query"):
                try:
                    cursor = st.session_state.conn.cursor()
                    cursor.execute(custom_query)
                    results = cursor.fetchall()
                    if results:
                        columns = [description[0] for description in cursor.description]
                        df_results = pd.DataFrame(results, columns=columns)
                        st.success(f"✅ Query executed: {len(df_results)} row(s) returned")
                        st.dataframe(df_results, use_container_width=True)
                    else:
                        st.info("✅ Query executed (no rows returned)")
                except Exception as e:
                    st.error(f"❌ Error executing query: {e}")


def analyze_query(query, model, vectorizer):
    """Analyze a single query and append to session history."""
    st.markdown("---")
    st.subheader("🔍 Analysis Results")

    # ML Detection
    ml_pred, confidence = detect_sql_injection_ml(query, model, vectorizer)

    # Rule-based validation
    is_valid, patterns = validate_input(query)

    # Display summary cards
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if ml_pred == 1:
            st.markdown('<div class="malicious-box">🚨 MALICIOUS</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">✅ SAFE</div>', unsafe_allow_html=True)
    with col2:
        if confidence is None:
            st.metric("ML Confidence", "N/A")
        else:
            st.metric("ML Confidence", f"{confidence:.2f}%")
    with col3:
        st.write("**Rule-based Validation**")
        if is_valid:
            st.success("✅ No suspicious patterns detected")
        else:
            st.error("⚠️ Detected patterns:")
            for p in patterns:
                st.write(f" • {p}")

    # Query details
    with st.expander("📋 Query Details"):
        st.code(query, language="sql")
        st.json({
            "Length": len(query),
            "Has Special Chars": bool(re.search(r"[';\"#-]", query)),
            "Has SQL Keywords": bool(re.search(r"\b(SELECT|UNION|DROP|INSERT|DELETE|UPDATE|ALTER)\b", query, re.I))
        })

    # Save to history
    st.session_state.detection_history.append({
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Query': query,
        'Prediction': 'Malicious' if ml_pred == 1 else 'Safe',
        'Confidence': f"{confidence:.2f}%" if confidence is not None else 'N/A',
        'Patterns': ', '.join(patterns) if patterns else 'None'
    })


def analyze_batch(df, col_name, model, vectorizer):
    """Analyze a DataFrame column of queries in batch."""
    st.markdown("---")
    st.subheader("📊 Batch Analysis Results")

    results = []
    total = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, row in df.iterrows():
        query = str(row[col_name])
        ml_pred, confidence = detect_sql_injection_ml(query, model, vectorizer)
        is_valid, patterns = validate_input(query)

        results.append({
            'Query': query if len(query) <= 200 else query[:200] + "...",
            'Prediction': 'Malicious' if ml_pred == 1 else 'Safe',
            'Confidence': f"{confidence:.2f}%" if confidence is not None else 'N/A',
            'Patterns': ', '.join(patterns) if patterns else 'None'
        })

        # Append to session history
        st.session_state.detection_history.append({
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Query': query,
            'Prediction': 'Malicious' if ml_pred == 1 else 'Safe',
            'Confidence': f"{confidence:.2f}%" if confidence is not None else 'N/A',
            'Patterns': ', '.join(patterns) if patterns else 'None'
        })

        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Analyzed {idx + 1}/{total}")

    df_results = pd.DataFrame(results)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    malicious_count = len(df_results[df_results['Prediction'] == 'Malicious'])
    with col1:
        st.metric("Total Analyzed", len(df_results))
    with col2:
        st.metric("Malicious", malicious_count, delta=f"{(malicious_count/len(df_results)*100):.1f}%")
    with col3:
        st.metric("Safe", len(df_results) - malicious_count)
    with col4:
        try:
            avg_conf = df_results['Confidence'].str.rstrip('%').astype(float).mean()
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        except Exception:
            st.metric("Avg Confidence", "N/A")

    st.dataframe(df_results, use_container_width=True)

    # Download results
    csv = df_results.to_csv(index=False)
    st.download_button("📥 Download Results", csv, "batch_analysis_results.csv", "text/csv")


if __name__ == "__main__":
    main()