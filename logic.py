import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import time
import hashlib

# --- Cấu hình trang và API ---
# Thiết lập cấu hình ban đầu cho trang Streamlit
st.set_page_config(
    page_title="Phân tích Rủi ro Tín dụng",
    page_icon="🤖",
    layout="wide"
)

# --- Quản lý State của ứng dụng ---
# Sử dụng st.session_state để lưu trữ trạng thái giữa các lần tương tác
# Khởi tạo các giá trị nếu chúng chưa tồn tại
if 'trained_model' not in st.session_state:
    st.session_state['trained_model'] = None
if 'model_accuracy' not in st.session_state:
    st.session_state['model_accuracy'] = None
if 'model_cm' not in st.session_state:
    st.session_state['model_cm'] = None
if 'feature_names' not in st.session_state:
    st.session_state['feature_names'] = None
if 'current_customer_data' not in st.session_state:
    st.session_state['current_customer_data'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'current_customer_id' not in st.session_state:
    st.session_state['current_customer_id'] = None


# Lấy API key từ Streamlit secrets (khi deploy) hoặc từ input của người dùng (khi chạy local)
try:
    # Ưu tiên lấy key từ secrets để bảo mật
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (FileNotFoundError, KeyError):
    st.sidebar.warning("Không tìm thấy API Key trong secrets.", icon="🔑")
    GOOGLE_API_KEY = st.sidebar.text_input("Nhập Gemini API Key của bạn:", type="password")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)


# --- Các hàm xử lý logic (Functions) ---

def train_model(df):
    """
    Hàm để huấn luyện mô hình Logistic Regression từ dataframe đầu vào.
    Trả về model đã huấn luyện, độ chính xác, ma trận nhầm lẫn và danh sách tên các đặc trưng.
    """
    # Kiểm tra xem cột biến mục tiêu 'y' có tồn tại không
    if 'y' not in df.columns:
        st.error("Dữ liệu huấn luyện phải có cột 'y' là biến mục tiêu.")
        return None, None, None, None
    X = df.drop(columns=['y'])
    y = df['y']
    # Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Khởi tạo và huấn luyện mô hình
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    # Đánh giá mô hình
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    return model, accuracy, cm, X.columns.tolist()

def get_initial_prompt(customer_data, pd_score):
    """
    Tạo prompt (câu lệnh) khởi đầu cho chatbot với đầy đủ ngữ cảnh về khách hàng.
    Điều này giúp AI hiểu rõ vai trò và thông tin cần phân tích.
    """
    # Chuyển đổi dict dữ liệu khách hàng thành chuỗi có định dạng
    data_string = "\n".join([f"- {key}: {value}" for key, value in customer_data.items()])
    prompt = f"""
    **BỐI CẢNH:**
    Bạn là một Trợ lý AI chuyên về phân tích rủi ro tín dụng, đang trò chuyện với một chuyên viên tín dụng. Bạn vừa nhận được thông tin về một khách hàng cụ thể.

    **Dữ liệu hồ sơ khách hàng:**
    {data_string}

    **Kết quả chấm điểm rủi ro từ mô hình:**
    - Xác suất vỡ nợ (PD): {pd_score:.2%}

    **NHIỆM VỤ CỦA BẠN:**
    1.  Bắt đầu cuộc trò chuyện bằng cách chào chuyên viên tín dụng và xác nhận bạn đã sẵn sàng phân tích hồ sơ này.
    2.  Chờ câu hỏi từ chuyên viên và trả lời một cách chuyên sâu, tập trung vào khách hàng này.
    3.  Các chủ đề bạn có thể thảo luận bao gồm:
        - Phân tích sâu hơn về các yếu tố rủi ro.
        - Đề xuất các câu hỏi cần phỏng vấn khách hàng.
        - Xây dựng các phương án cho vay (ví dụ: điều kiện, tài sản đảm bảo bổ sung).
        - Lên kế hoạch thu hồi nợ nếu có rủi ro.
        - So sánh (một cách giả định) với các hồ sơ rủi ro/an toàn điển hình.
    4.  Luôn giữ vai trò là một trợ lý chuyên nghiệp, đưa ra các phân tích dựa trên dữ liệu được cung cấp.

    **Bắt đầu ngay bây giờ.** Hãy gửi lời chào đầu tiên.
    """
    return prompt

# --- Giao diện ứng dụng Streamlit (UI) ---

st.title("🤖 Phần mềm Phân tích Rủi ro Tín dụng & Trợ lý AI")

# Tạo thanh điều hướng bên trái (sidebar)
menu = ["Giới thiệu & Huấn luyện Mô hình", "Dự báo Vỡ nợ cho Khách hàng", "Chatbot Phân tích Rủi ro"]
choice = st.sidebar.selectbox('Chọn chức năng', menu)

# ==============================================================================
# --- Trang 1: Giới thiệu & Huấn luyện Mô hình ---
# ==============================================================================
if choice == 'Giới thiệu & Huấn luyện Mô hình':
    st.header("1. Mục tiêu của Mô hình")
    st.markdown("""
    Mô hình này được xây dựng để hỗ trợ các chuyên viên tín dụng trong việc đưa ra quyết định cho vay bằng cách:
    - **Dự báo Xác suất Vỡ nợ (Probability of Default - PD)** của khách hàng dựa trên các đặc điểm kinh tế - xã hội.
    - **Sử dụng mô hình Hồi quy Logistic (Logistic Regression)**, một thuật toán phổ biến và diễn giải được trong ngành tài chính.
    - **Tích hợp Trí tuệ nhân tạo (AI)** để cung cấp các phân tích sâu hơn, giúp hiểu rõ "tại sao" đằng sau mỗi con số.
    Bên dưới, bạn có thể tải lên tập dữ liệu lịch sử (`.csv`, `.xlsx`) để huấn luyện hoặc kiểm tra lại mô hình.
    """)
    st.info("Lưu ý: Dữ liệu huấn luyện cần có cột `y` làm biến mục tiêu, trong đó `1` là 'vỡ nợ' và `0` là 'không vỡ nợ'.", icon="ℹ️")

    st.header("2. Huấn luyện và Đánh giá Mô hình")
    uploaded_file = st.file_uploader("Tải lên tệp dữ liệu huấn luyện (CSV hoặc Excel)", type=['csv', 'xlsx'])
    df = None
    # Xử lý việc tải file hoặc sử dụng file mặc định
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")
    else:
        st.markdown("Sử dụng dữ liệu mẫu `credit_access.csv` để huấn luyện.")
        try:
            df = pd.read_csv('credit_access.csv', encoding='latin-1')
        except FileNotFoundError:
            st.error("Không tìm thấy tệp `credit_access.csv`. Vui lòng tải lên một tệp dữ liệu.")

    if df is not None:
        st.dataframe(df.head())
        if st.button("Huấn luyện Mô hình"):
            with st.spinner("Đang huấn luyện mô hình..."):
                # Gọi hàm huấn luyện và lưu kết quả vào session state
                model, accuracy, cm, feature_names = train_model(df)
                st.session_state['trained_model'] = model
                st.session_state['model_accuracy'] = accuracy
                st.session_state['model_cm'] = cm
                st.session_state['feature_names'] = feature_names
                time.sleep(1) # Tạm dừng để người dùng cảm nhận quá trình
            st.success(f"Huấn luyện thành công! Độ chính xác trên tập kiểm tra: **{accuracy:.2%}**")
            st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Không vỡ nợ', 'Vỡ nợ'], yticklabels=['Không vỡ nợ', 'Vỡ nợ'])
            plt.xlabel('Dự đoán'); plt.ylabel('Thực tế')
            st.pyplot(fig)

# ==============================================================================
# --- Trang 2: Dự báo Vỡ nợ cho Khách hàng ---
# ==============================================================================
elif choice == 'Dự báo Vỡ nợ cho Khách hàng':
    st.header("Nhập thông tin để dự báo")
    # Kiểm tra xem mô hình đã được huấn luyện chưa
    if not st.session_state['trained_model']:
        st.warning("Mô hình chưa được huấn luyện. Vui lòng quay lại trang 'Giới thiệu & Huấn luyện Mô hình' để huấn luyện trước.", icon="⚠️")
        st.stop() # Dừng thực thi trang nếu chưa có mô hình

    input_method = st.radio("Chọn phương thức nhập liệu:", ("Nhập thủ công", "Tải lên tệp"))
    customer_data_df = None
    customer_data_dict = {}

    if input_method == "Nhập thủ công":
        st.subheader("Thông tin khách hàng")
        # Sử dụng cột để giao diện gọn gàng hơn
        col1, col2 = st.columns(2)
        with col1:
            dien_tich_dat = st.number_input('Diện tích đất sở hữu (m²)', min_value=0, value=100)
            thu_nhap_nam = st.number_input('Thu nhập một năm của hộ (triệu VNĐ)', min_value=0, value=150)
            tuoi_chu_ho = st.number_input('Tuổi chủ hộ', min_value=18, max_value=100, value=40)
            gioi_tinh = st.selectbox('Giới tính', ['Nam', 'Nữ'])
            dia_vi_chu_ho = st.selectbox('Địa vị chủ hộ', ['Chủ hộ', 'Thành viên khác'])
        with col2:
            so_nguoi_phu_thuoc = st.number_input('Số người phụ thuộc', min_value=0, value=2)
            lich_su_tin_dung = st.selectbox('Lịch sử tín dụng', ['Tốt', 'Chưa có thông tin', 'Có nợ xấu'])
            gia_tri_the_chap = st.number_input('Giá trị tài sản thế chấp (triệu VNĐ)', min_value=0, value=200)
            vay_phi_chinh_thuc = st.selectbox('Vay thị trường phi chính thức?', ['Có', 'Không'])
            so_nam_den_truong = st.number_input('Số năm đến trường của chủ hộ', min_value=0, value=12)

        if st.button("Dự báo"):
            # Chuyển đổi dữ liệu nhập từ form thành định dạng số mà model hiểu được
            customer_data_dict = {
                'DT': dien_tich_dat, 'TN': thu_nhap_nam, 'TCH': tuoi_chu_ho, 'GT': 1 if gioi_tinh == 'Nam' else 0,
                'DV': 1 if dia_vi_chu_ho == 'Chủ hộ' else 0, 'SPT': so_nguoi_phu_thuoc,
                'LS': {'Tốt': 1, 'Chưa có thông tin': 0, 'Có nợ xấu': -1}[lich_su_tin_dung],
                'GTC': gia_tri_the_chap, 'VPCT': 1 if vay_phi_chinh_thuc == 'Có' else 0, 'GD': so_nam_den_truong,
            }
            # Tạo DataFrame từ dict và đảm bảo thứ tự cột đúng như lúc huấn luyện
            customer_data_df = pd.DataFrame([customer_data_dict])[st.session_state['feature_names']]

    # (Phần code tải file lên có thể được thêm vào đây nếu cần)

    # Nếu có dữ liệu khách hàng để dự báo
    if customer_data_df is not None:
        st.subheader("Kết quả Dự báo")
        model = st.session_state['trained_model']
        probabilities = model.predict_proba(customer_data_df)
        pd_scores = probabilities[:, 1] # Lấy xác suất của lớp 1 (vỡ nợ)
        results_df = customer_data_df.copy()
        results_df['Xác suất Vỡ nợ (PD)'] = [f"{score:.2%}" for score in pd_scores]
        st.dataframe(results_df)

        # Xử lý riêng cho trường hợp dự báo 1 khách hàng để kích hoạt chatbot
        if len(customer_data_df) == 1:
            pd_score = pd_scores[0]
            # Lưu thông tin khách hàng hiện tại vào session_state để chatbot sử dụng
            st.session_state['current_customer_data'] = {
                "Dữ liệu gốc": customer_data_dict,
                "Xác suất vỡ nợ": pd_score
            }
            # Tạo ID duy nhất cho khách hàng để biết khi nào cần reset cuộc trò chuyện
            customer_id = hashlib.md5(str(customer_data_dict).encode()).hexdigest()
            # Nếu ID khách hàng thay đổi, reset lịch sử chat
            if st.session_state['current_customer_id'] != customer_id:
                st.session_state['messages'] = []
                st.session_state['current_customer_id'] = customer_id

            st.success("Dữ liệu khách hàng đã được ghi nhận. Hãy chuyển qua trang **'Chatbot Phân tích Rủi ro'** để bắt đầu thảo luận sâu hơn với Trợ lý AI.", icon="👉")


# ==============================================================================
# --- Trang 3: Chatbot Phân tích Rủi ro ---
# ==============================================================================
elif choice == 'Chatbot Phân tích Rủi ro':
    st.header("💬 Chatbot Phân tích Rủi ro")

    # Kiểm tra xem đã có dữ liệu khách hàng để phân tích chưa
    if not st.session_state['current_customer_data']:
        st.info("Vui lòng thực hiện dự báo cho một khách hàng ở trang 'Dự báo Vỡ nợ' trước khi sử dụng chatbot.", icon="ℹ️")
        st.stop()

    # Kiểm tra xem API key đã được cung cấp chưa
    if not GOOGLE_API_KEY:
        st.error("Vui lòng nhập Gemini API Key ở thanh bên để kích hoạt chatbot.")
        st.stop()

    # Khởi tạo model chat của Gemini
    model = genai.GenerativeModel('gemini-pro')

    # Nếu chưa có tin nhắn, khởi tạo cuộc trò chuyện với prompt hệ thống
    if not st.session_state.messages:
        initial_prompt = get_initial_prompt(
            st.session_state['current_customer_data']['Dữ liệu gốc'],
            st.session_state['current_customer_data']['Xác suất vỡ nợ']
        )
        with st.spinner("Trợ lý AI đang phân tích hồ sơ..."):
            # Bắt đầu cuộc trò chuyện với Gemini
            chat = model.start_chat(history=[])
            response = chat.send_message(initial_prompt)
            initial_message = response.text
        # Thêm tin nhắn chào mừng của AI vào lịch sử
        st.session_state.messages.append({"role": "assistant", "content": initial_message})


    # Hiển thị lịch sử tin nhắn đã có
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý input (câu hỏi) từ người dùng
    if prompt := st.chat_input("Đặt câu hỏi về khách hàng này..."):
        # Thêm tin nhắn của người dùng vào lịch sử và hiển thị lên giao diện
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Tạo lại lịch sử chat theo định dạng mà API của Gemini yêu cầu
        chat_history = []
        for msg in st.session_state.messages:
             chat_history.append({"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]})

        # Gửi tin nhắn đến Gemini và nhận phản hồi
        with st.spinner("AI đang suy nghĩ..."):
             chat = model.start_chat(history=chat_history[:-1]) # Gửi toàn bộ lịch sử trừ tin nhắn cuối cùng của user
             response = chat.send_message(prompt)
             response_text = response.text

        # Hiển thị và lưu lại phản hồi của AI
        with st.chat_message("assistant"):
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
