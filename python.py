# python.py

import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError
from streamlit_modal import Modal # <-- Nhập thư viện Modal

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

# --- Mã CSS để tạo nút bấm nổi (Floating Action Button) ---
# Icon, màu sắc và vị trí của nút bấm được định nghĩa ở đây.
st.markdown("""
<style>
    /* CSS để định vị container của nút bấm */
    .st-emotion-cache-1v0mbdj > button {
        position: fixed;
        bottom: 35px;
        right: 40px;
        z-index: 1000;
        border-radius: 50%;
        width: 65px;
        height: 65px;
        background-color: #9A242B; /* Màu đỏ của Agribank */
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.2s ease;
    }
    .st-emotion-cache-1v0mbdj > button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .st-emotion-cache-1v0mbdj > button > span {
        font-size: 32px; /* Kích thước icon */
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'. Vui lòng kiểm tra lại tên chỉ tiêu trong file Excel.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Giao diện chính ---
st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")
st.markdown("Tải lên Bảng cân đối kế toán của bạn để xem các phân tích và tương tác với Trợ lý AI.")

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Định dạng: Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls'],
    key="file_uploader" # Thêm key để quản lý state tốt hơn
)

# Xóa state cũ khi tải file mới, trừ state của modal
if 'file_uploader' in st.session_state and not st.session_state.get('last_uploaded_file') == uploaded_file.name:
    for key in list(st.session_state.keys()):
        if key not in ['modal_is_open', 'file_uploader']:
             del st.session_state[key]
    st.session_state.last_uploaded_file = uploaded_file.name


if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        if len(df_raw.columns) >= 3:
            df_raw = df_raw.iloc[:, :3]
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        else:
            raise ValueError("File Excel cần ít nhất 3 cột: Chỉ tiêu, Năm trước, Năm sau.")

        df_processed = process_financial_data(df_raw.copy())
        st.session_state.df_processed = df_processed

        # --- Chức năng 2 & 3: Hiển thị Kết quả ---
        st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
        st.dataframe(df_processed.style.format({
            'Năm trước': '{:,.0f}',
            'Năm sau': '{:,.0f}',
            'Tốc độ tăng trưởng (%)': '{:.2f}%',
            'Tỷ trọng Năm trước (%)': '{:.2f}%',
            'Tỷ trọng Năm sau (%)': '{:.2f}%'
        }), use_container_width=True)
        
        # --- Chức năng 4: Tính Chỉ số Tài chính ---
        st.subheader("4. Các Chỉ số Tài chính Cơ bản")
        try:
            tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
            tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
            no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
            no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
            thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else 0
            thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 0
            st.session_state.financial_ratios = {"current_ratio_n": thanh_toan_hien_hanh_N, "current_ratio_n_1": thanh_toan_hien_hanh_N_1}
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Chỉ số Thanh toán Hiện hành (Năm trước)", value=f"{thanh_toan_hien_hanh_N_1:.2f} lần")
            with col2:
                st.metric(label="Chỉ số Thanh toán Hiện hành (Năm sau)", value=f"{thanh_toan_hien_hanh_N:.2f} lần", delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}")
        except IndexError:
            st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số thanh toán hiện hành.")
            st.session_state.financial_ratios = None

    except ValueError as ve: st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e: st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# --- Chức năng 6: POPUP CHAT VỚI AI ---
# Chỉ hiển thị nút chat khi đã có dữ liệu
if 'df_processed' in st.session_state:
    
    # 1. Khai báo Modal (khung popup)
    chat_modal = Modal(
        "Trợ lý Tài chính AI",
        key="chat_modal",
        # Cấu hình để popup rộng hơn
        max_width=700
    )

    # 2. Tạo nút bấm nổi để mở Modal
    open_modal_button = st.button("💬")
    if open_modal_button:
        chat_modal.open()
        
    # 3. Định nghĩa nội dung bên trong Modal
    if chat_modal.is_open():
        with chat_modal.container():
            st.markdown("#### Trò chuyện với AI") # Thêm tiêu đề trong popup

            api_key = st.secrets.get("GEMINI_API_KEY")

            if not api_key:
                st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình 'GEMINI_API_KEY' trong Streamlit Secrets.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    
                    if 'chat_session' not in st.session_state:
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        st.session_state.chat_session = model.start_chat(history=[])
                        st.session_state.messages = []

                        initial_prompt = f"""
                        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Nhiệm vụ của bạn là phân tích dữ liệu từ bảng cân đối kế toán được cung cấp và trả lời các câu hỏi của người dùng một cách ngắn gọn, chuyên nghiệp.
                        Dữ liệu đã được xử lý: {st.session_state.df_processed.to_markdown(index=False)}
                        Các chỉ số tài chính quan trọng:
                        - Chỉ số thanh toán hiện hành năm sau: {st.session_state.financial_ratios['current_ratio_n']:.2f}
                        - Chỉ số thanh toán hiện hành năm trước: {st.session_state.financial_ratios['current_ratio_n_1']:.2f}
                        Hãy bắt đầu bằng cách đưa ra một nhận xét tổng quan (3-4 câu) về tình hình tài chính của doanh nghiệp. Sau đó, hãy sẵn sàng trả lời các câu hỏi chi tiết hơn.
                        """
                        
                        with st.spinner('Trợ lý AI đang phân tích dữ liệu...'):
                            initial_response = st.session_state.chat_session.send_message(initial_prompt)
                            st.session_state.messages.append({"role": "model", "parts": [initial_response.text]})

                    # Hiển thị lịch sử chat
                    for message in st.session_state.messages:
                        with st.chat_message(name="assistant" if message["role"] == "model" else "user"):
                            st.markdown(message["parts"][0])
                    
                    # Input chat
                    if prompt := st.chat_input("Hỏi AI về báo cáo này..."):
                        st.session_state.messages.append({"role": "user", "parts": [prompt]})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.spinner('...'):
                            response = st.session_state.chat_session.send_message(prompt)
                            response_text = response.text
                            st.session_state.messages.append({"role": "model", "parts": [response_text]})
                            with st.chat_message("assistant"):
                                st.markdown(response_text)
                        
                        # Tự động chạy lại để hiển thị tin nhắn mới nhất
                        st.rerun()

                except GoogleAPICallError as e:
                    st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra lại Khóa API. Chi tiết lỗi: {e}")
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi không xác định khi tương tác với AI: {e}")
