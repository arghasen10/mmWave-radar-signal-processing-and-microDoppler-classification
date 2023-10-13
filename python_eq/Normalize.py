def normalize(Xcube, max_val):
    Xcube = Xcube / max_val
    Angdata = Xcube.astype('float32')  # Assuming you want single precision (32-bit) floating point
    return Angdata
