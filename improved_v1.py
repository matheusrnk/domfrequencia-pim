import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

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
cutoff = 45  # pode ajustar este valor
order = 5  # pode ajustar este valor

# Criar e aplicar o filtro Butterworth
butterworth_filter = butterworth_lowpass_filter(dft_shift, cutoff, order)
fshift = dft_shift * butterworth_filter

# Transformada inversa para obter a imagem filtrada
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# Calcular o SSIM entre a imagem original e a filtrada
img_original = cv2.imread('folhas1.jpg', 0)
ssim_value = ssim(img_original, img_back, data_range=img_back.max()-img_back.min())

print(f'SSIM: {ssim_value}')

# Visualizar a imagem filtrada
plt.figure(figsize=(6, 6))
plt.imshow(img_back, cmap='gray')
plt.title('Imagem Filtrada')
plt.show()
