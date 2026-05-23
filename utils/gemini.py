import google.generativeai as genai

genai.configure(
    api_key="AIzaSyB94C7_2bexTI60_SWBCnsWGFx56LK02n0"
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def generate_explanation(disease):

    prompt = f"""
    Jelaskan penyakit daun porang {disease} yang terdeteksi.

    Berikan penjelasan dalam bentuk 1 paragraf meliputi:
    - nama penyakit
    - ciri-ciri penyakit
    - penyebab penyakit
    - solusi penanganan

    Paragraf tidak boleh terlalu panjang, harus efektif, dan efisien. Gunakan bahasa Indonesia yang mudah dipahami petani.
    """

    try:

        response = model.generate_content(prompt)

        if response.text:
            return response.text

        return "AI tidak memberikan respon."

    except Exception as e:

        print("ERROR GEMINI:", e)

        return f"""
        Penjelasan AI sementara tidak tersedia.

        Penyakit terdeteksi: {disease}

        Error:
        {str(e)}
        """