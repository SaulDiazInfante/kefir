import qrcode

url = "https://slides.com/sauldiazinfantevelasco/seminario-proba-unam-2026"

# Configure the QR code for optimal cell phone scanning
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H, # 30% error correction
    box_size=15, # Larger box size for higher resolution
    border=4,
)

qr.add_data(url)
qr.make(fit=True)

# Generate and save the image
img = qr.make_image(fill_color="black", back_color="white")
filename = "UNAM_Seminar_2026_QR_Optimized.png"
img.save(filename)

print(f"High-resolution QR code saved successfully as {filename}")