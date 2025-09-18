def close_all():
    """Try to flush and close all file handles"""
    global FILEHANDLES
    for fh in FILEHANDLES:
        try:
            fh.flush()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass