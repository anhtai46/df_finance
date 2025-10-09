import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import time

# --- Cấu hình trang và API ---
st.set_page_config(
    page_title="Phân tích Rủi ro Tín dụng",
    page_icon="🏦",
    layout="wide"
)

# Lấy API key từ Streamlit secrets (cho deploy) hoặc input (cho local)
try:
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (FileNotFoundError, KeyError):
    st.sidebar.warning("Không tìm thấy API Key trong secrets.", icon="🔑")
    GOOGLE_API_KEY = st.sidebar.text_input("Nhập Gemini API Key của bạn:", type="password")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.info("Ứng dụng cần Gemini API Key để sử dụng tính năng phân tích của AI.")
        # Không dừng ứng dụng, vẫn cho phép tính toán xác suất
        # st.stop()

# --- Các hàm xử lý logic ---

def train_model(df):
    """
    Hàm huấn luyện mô hình Logistic Regression từ dữ liệu được cung cấp.
    'y' là biến mục tiêu (1: vỡ nợ, 0: không vỡ nợ).
    """
    if 'y' not in df.columns:
        st.error("Dữ liệu huấn luyện phải có cột 'y' là biến mục tiêu.")
        return None, None, None, None

    X = df.drop(columns=['y'])
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)

    return model, accuracy, cm, X.columns.tolist()

def get_gemini_analysis(customer_data, pd_score):
    """
    Hàm gọi API Gemini để phân tích, giải thích kết quả và đưa ra khuyến nghị.
    """
    if not GOOGLE_API_KEY:
        return "Vui lòng cung cấp API Key để sử dụng tính năng này."

    model = genai.GenerativeModel('gemini-pro')

    # Chuyển dữ liệu khách hàng thành chuỗi dễ đọc
    data_string = "\n".join([f"- {key}: {value}" for key, value in customer_data.items()])

    prompt = f"""
    Bạn là một chuyên gia quản trị rủi ro tín dụng cao cấp tại một tổ chức tài chính.
    Dựa trên thông tin khách hàng và xác suất vỡ nợ (PD) được tính toán, hãy đưa ra một bản phân tích chuyên sâu.

    **Thông tin khách hàng:**
    {data_string}

    **Kết quả mô hình:**
    - Xác suất vỡ nợ (PD): {pd_score:.2%}

    **Yêu cầu phân tích:**
    1.  **Giải thích mức độ rủi ro:** Dựa vào chỉ số PD, hãy giải thích ngắn gọn mức độ rủi ro của khách hàng này (Thấp, Trung bình, Cao, Rất cao).
    2.  **Yếu tố ảnh hưởng chính:** Từ dữ liệu khách hàng, chỉ ra 2-3 yếu tố có khả năng ảnh hưởng lớn nhất đến kết quả PD này (ví dụ: thu nhập thấp, lịch sử tín dụng không tốt, v.v.).
    3.  **Đề xuất hành động:** Dựa trên phân tích, đưa ra một đề xuất rõ ràng cho cán bộ tín dụng (Ví dụ: Phê duyệt khoản vay, Yêu cầu thêm tài sản thế chấp, Từ chối, Phỏng vấn sâu hơn,...).
    4.  **Phương án thu hồi nợ (nếu rủi ro cao):** Nếu khách hàng có rủi ro từ trung bình đến cao, hãy đề xuất 2-3 phương án/chiến lược phòng ngừa và thu hồi nợ tiềm năng.

    Trình bày kết quả một cách chuyên nghiệp, có cấu trúc và dễ hiểu.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Đã có lỗi xảy ra khi kết nối đến AI: {e}"

# --- Giao diện ứng dụng Streamlit ---

st.title("🏦 Phần mềm Phân tích và Dự báo Xác suất Vỡ nợ")

menu = ["Giới thiệu & Huấn luyện Mô hình", "Dự báo Vỡ nợ cho Khách hàng"]
choice = st.sidebar.selectbox('Chọn chức năng', menu)

# --- Trang 1: Giới thiệu & Huấn luyện Mô hình ---
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

    # Tải lên file dữ liệu huấn luyện
    uploaded_file = st.file_uploader("Tải lên tệp dữ liệu huấn luyện (CSV hoặc Excel)", type=['csv', 'xlsx'])
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            else:
                df = pd.read_excel(uploaded_file)
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
                model, accuracy, cm, feature_names = train_model(df)
                st.session_state['trained_model'] = model
                st.session_state['model_accuracy'] = accuracy
                st.session_state['model_cm'] = cm
                st.session_state['feature_names'] = feature_names # Lưu lại thứ tự các cột
                time.sleep(1) # Giả lập thời gian huấn luyện
            st.success(f"Huấn luyện thành công! Độ chính xác trên tập kiểm tra: **{accuracy:.2%}**")

            st.subheader("Ma trận nhầm lẫn (Confusion Matrix)")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Không vỡ nợ', 'Vỡ nợ'],
                        yticklabels=['Không vỡ nợ', 'Vỡ nợ'])
            plt.xlabel('Dự đoán')
            plt.ylabel('Thực tế')
            st.pyplot(fig)


# --- Trang 2: Dự báo Vỡ nợ cho Khách hàng ---
elif choice == 'Dự báo Vỡ nợ cho Khách hàng':
    st.header("Nhập thông tin để dự báo")

    if 'trained_model' not in st.session_state:
        st.warning("Mô hình chưa được huấn luyện. Vui lòng quay lại trang 'Giới thiệu & Huấn luyện Mô hình' để huấn luyện trước.", icon="⚠️")
        st.stop()

    input_method = st.radio("Chọn phương thức nhập liệu:", ("Nhập thủ công", "Tải lên tệp"))

    customer_data_df = None
    customer_data_dict = {}

    if input_method == "Nhập thủ công":
        st.subheader("Thông tin khách hàng")
        # Sử dụng 2 cột để giao diện gọn gàng hơn
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
            # Chuyển đổi dữ liệu sang dạng số để đưa vào mô hình
            customer_data_dict = {
                'DT': dien_tich_dat,
                'TN': thu_nhap_nam,
                'TCH': tuoi_chu_ho,
                'GT': 1 if gioi_tinh == 'Nam' else 0,
                'DV': 1 if dia_vi_chu_ho == 'Chủ hộ' else 0,
                'SPT': so_nguoi_phu_thuoc,
                'LS': {'Tốt': 1, 'Chưa có thông tin': 0, 'Có nợ xấu': -1}[lich_su_tin_dung],
                'GTC': gia_tri_the_chap,
                'VPCT': 1 if vay_phi_chinh_thuc == 'Có' else 0,
                'GD': so_nam_den_truong,
            }
            # Tạo DataFrame với thứ tự cột chính xác như khi huấn luyện
            customer_data_df = pd.DataFrame([customer_data_dict])[st.session_state['feature_names']]


    elif input_method == "Tải lên tệp":
        uploaded_predict_file = st.file_uploader("Tải lên tệp khách hàng cần dự báo (CSV hoặc Excel)", type=['csv', 'xlsx'])
        if uploaded_predict_file:
            try:
                if uploaded_predict_file.name.endswith('.csv'):
                    customer_data_df = pd.read_csv(uploaded_predict_file)
                else:
                    customer_data_df = pd.read_excel(uploaded_predict_file)

                # Đảm bảo các cột trong file tải lên khớp với mô hình
                if not all(col in customer_data_df.columns for col in st.session_state['feature_names']):
                     st.error(f"Tệp tải lên thiếu các cột cần thiết. Yêu cầu có đủ các cột: {st.session_state['feature_names']}")
                     customer_data_df = None
                else:
                    # Sắp xếp lại các cột cho đúng thứ tự
                    customer_data_df = customer_data_df[st.session_state['feature_names']]

            except Exception as e:
                st.error(f"Lỗi đọc file: {e}")
                customer_data_df = None


    # --- Hiển thị kết quả dự báo và phân tích AI ---
    if customer_data_df is not None:
        st.subheader("Kết quả Dự báo")
        model = st.session_state['trained_model']
        probabilities = model.predict_proba(customer_data_df)
        pd_scores = probabilities[:, 1] # Lấy xác suất của lớp 1 (vỡ nợ)

        results_df = customer_data_df.copy()
        results_df['Xác suất Vỡ nợ (PD)'] = [f"{score:.2%}" for score in pd_scores]

        st.dataframe(results_df)

        # Chỉ thực hiện phân tích AI cho trường hợp nhập tay (1 khách hàng)
        if len(customer_data_df) == 1:
            pd_score = pd_scores[0]
            if pd_score > 0.5:
                st.error(f"**XÁC SUẤT VỠ NỢ: {pd_score:.2%} (Rủi ro cao)**", icon="🚨")
            elif pd_score > 0.2:
                st.warning(f"**XÁC SUẤT VỠ NỢ: {pd_score:.2%} (Rủi ro trung bình)**", icon="⚠️")
            else:
                st.success(f"**XÁC SUẤT VỠ NỢ: {pd_score:.2%} (Rủi ro thấp)**", icon="✅")


            if st.button("Yêu cầu AI Phân tích Chuyên sâu"):
                with st.spinner("AI đang phân tích, vui lòng chờ..."):
                    ai_result = get_gemini_analysis(customer_data_dict, pd_score)
                    st.session_state['ai_analysis'] = ai_result

                if 'ai_analysis' in st.session_state:
                    st.subheader("📝 Phân tích và Khuyến nghị từ AI")
                    st.markdown(st.session_state['ai_analysis'])
