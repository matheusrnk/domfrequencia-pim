import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import median_filter

def butterworth_lowpass_filter(dft_shift, cutoff, order):
    rows, cols = dft_shift.shape
    crow, ccol = rows // 2 , cols // 2
    x, y = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - crow)**2 + (y - ccol)**2)
    filter = 1 / (1 + (distance / cutoff)**(2 * order))
    return filter

# Carregar a imagem reticulada
img_reticulada = cv2.imread('folhas1_Reticulada.jpg', 0)

# Aplicar a Transformada de Fourier
dft = np.fft.fft2(img_reticulada)
dft_shift = np.fft.fftshift(dft)

# Visualizar o espectro de frequência
magnitude_spectrum = np.log(np.abs(dft_shift))

plt.figure(figsize=(6, 6))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Espectro de Frequência')
plt.show()

# Ajustar parâmetros do filtro
cutoff = 45  # ajuste conforme necessário
order = 5  # ajuste conforme necessário

# Criar e aplicar o filtro Butterworth
butterworth_filter = butterworth_lowpass_filter(dft_shift, cutoff, order)
fshift = dft_shift * butterworth_filter

# Transformada inversa para obter a imagem filtrada
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Aplicar suavização espacial adicional
img_back_smoothed = median_filter(img_back, size=3)  # ou outro filtro adequado

# Realce de bordas pós-processamento (opcional)
img_back_edges = cv2.Laplacian(img_back_smoothed, cv2.CV_64F)
img_back_final = img_back_smoothed + img_back_edges * 50  # ajuste o peso conforme necessário

# Calcular o SSIM entre a imagem original e a filtrada
img_original = cv2.imread('folhas1.jpg', 0)
ssim_value = ssim(img_original, img_back_final, data_range=img_back_final.max()-img_back_final.min())

print(f'SSIM: {ssim_value}')

# Visualizar a imagem filtrada
plt.figure(figsize=(6, 6))
plt.imshow(img_back_final, cmap='gray')
plt.title('Imagem Filtrada')
plt.show()
