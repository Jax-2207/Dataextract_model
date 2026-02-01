def detect_file_type(filename: str):
    ext = filename.split(".")[-1].lower()
    
    if ext in ["pdf"]:
        return "pdf"
    elif ext in ["mp3", "wav"]:
        return "audio"
    elif ext in ["mp4"]:
        return "video"
    elif ext in ["jpg", "jpeg", "png"]:
        return "image"
    else:
        return "unknown"