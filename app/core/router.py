def detect_file_type(filename: str):
    ext = filename.split(".")[-1].lower()
    
    if ext in ["pdf"]:
        return "pdf"
    elif ext in ["mp3", "wav", "m4a", "flac", "ogg"]:
        return "audio"
    elif ext in ["mp4", "avi", "mov", "mkv", "webm"]:
        return "video"
    elif ext in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]:
        return "image"
    elif ext in ["docx"]:
        return "docx"
    elif ext in ["xlsx", "xls"]:
        return "xlsx"
    elif ext in ["pptx", "ppt"]:
        return "pptx"
    else:
        return "unknown"