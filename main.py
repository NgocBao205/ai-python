from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI(title="AI Microservice - Đặt Vé Phim")

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"Lỗi khởi tạo API Key: {e}")

class ReviewRequest(BaseModel):
    content: str

def get_bulletproof_model():
    """Chỉ sử dụng 4 mô hình đã được user kiểm chứng là hoạt động 100%"""
    try:
        models_response = client.models.list()
        available_models = [m.id for m in models_response.data]
        
        # Danh sách 4 "chàng lính ngự lâm" mà bạn đã test thành công
        proven_models = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "allam-2-7b"
        ]
        
        # Lọc xem trong 4 anh này, anh nào đang online thì bốc ra xài ngay
        for model in proven_models:
            if model in available_models:
                return model
                
        # Nếu cả 4 đều lặn mất tăm (rất khó xảy ra), lấy đại mô hình đầu tiên không bắt accept terms
        for m in available_models:
            if "guard" not in m.lower() and "orpheus" not in m.lower():
                return m
                
        return proven_models[0]
    except Exception as e:
        print(f"Lỗi danh sách model: {e}")
        # Lỗi mạng cục bộ thì nhắm mắt trả về model mạnh nhất trong 4 anh
        return "openai/gpt-oss-120b" 

@app.post("/api/ai/check-spoiler")
async def check_spoiler(request: ReviewRequest):
    try:
        # Lấy "bộ não" tốt nhất hiện có
        active_model = get_bulletproof_model()
        if not active_model:
            raise ValueError("Groq hiện đang sập toàn bộ hệ thống model.")
            
        prompt = f"""
        Bạn là hệ thống kiểm duyệt bình luận phim.
        Xác định xem bình luận sau có chứa spoiler (tiết lộ nội dung, kết phim) hay không. 
        CHỈ TRẢ LỜI "True" HOẶC "False". Tuyệt đối không nói thêm gì khác.
        
        Bình luận: "{request.content}"
        """
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=active_model,
            temperature=0, 
        )
        
        result_text = chat_completion.choices[0].message.content.strip().lower()
        is_spoiler = True if "true" in result_text else False
        
        return {"is_spoiler": is_spoiler, "source": f"groq_{active_model}"}

    except Exception as e:
        print(f"Lỗi AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi AI: {str(e)}")

# Định nghĩa dữ liệu Node.js sẽ gửi sang
class ChatRequest(BaseModel):
    question: str
    context_data: str # Đây là dữ liệu thật từ DB do Node.js mớm cho AI

@app.post("/api/ai/chat")
async def chat_cskh(request: ChatRequest):
    try:
        active_model = get_bulletproof_model()
        if not active_model:
            raise ValueError("Hệ thống AI đang bảo trì.")
            
        prompt = f"""
        Bạn là nhân viên chăm sóc khách hàng cực kỳ thân thiện của rạp chiếu phim.
        Nhiệm vụ: Trả lời câu hỏi của khách hàng CHỈ DỰA VÀO phần "Dữ liệu rạp phim" bên dưới.
        Nếu khách hỏi ngoài lề hoặc dữ liệu không có, hãy lịch sự xin lỗi và nói không biết.
        TUYỆT ĐỐI KHÔNG tự bịa ra hoặc lấy dữ liệu trên mạng. Trả lời ngắn gọn, súc tích.
        
        [Dữ liệu rạp phim]:
        {request.context_data}
        
        [Câu hỏi của khách]: "{request.question}"
        """
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=active_model,
            temperature=0.3, # Tăng một chút sáng tạo để AI nói chuyện tự nhiên, có cảm xúc hơn
        )
        
        answer = chat_completion.choices[0].message.content.strip()
        
        return {"answer": answer, "source": f"groq_{active_model}"}

    except Exception as e:
        print(f"Lỗi AI Chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi AI Chat: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "🚀 AI Microservice (Chống đạn) đang hoạt động!"}