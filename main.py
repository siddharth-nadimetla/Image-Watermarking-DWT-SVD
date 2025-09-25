import cv2
import numpy as np
import pywt
import hashlib
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def hash_to_binary(text_to_hash):
    """Hashes a string using SHA-256 and returns its binary representation."""
    hashed = hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()
    return bin(int(hashed, 16))[2:].zfill(256)

def apply_dwt_svd(image, wavelet='haar'):
    coeffs = pywt.wavedec2(image.astype(np.float64), wavelet, level=2)
    LL2, _, _ = coeffs
    ll2_shape = LL2.shape
    U, S, V = np.linalg.svd(LL2)
    return (coeffs, U, S, V, ll2_shape)

def embed_watermark(S, watermark_binary, alpha, method):
    S_modified = np.copy(S)
    watermark_len = len(watermark_binary)
    if watermark_len > len(S):
        raise ValueError("Image resolution is too low to embed a 256-bit hash.")
    for i in range(watermark_len):
        bit = int(watermark_binary[i])
        if method == 'additive':
            S_modified[i] += alpha * bit
        elif method == 'multiplicative':
            S_modified[i] *= (1 + alpha * bit)
    return S_modified

def reconstruct_channel(coeffs, U, S_modified, V, ll2_shape, wavelet='haar'):
    Sigma_modified = np.zeros(ll2_shape)
    k = min(ll2_shape)
    Sigma_modified[:k, :k] = np.diag(S_modified[:k])
    LL2_modified = U @ Sigma_modified @ V
    coeffs_modified = [LL2_modified, coeffs[1], coeffs[2]]
    reconstructed_channel_float = pywt.waverec2(coeffs_modified, wavelet)
    reconstructed_channel_uint8 = np.clip(reconstructed_channel_float, 0, 255).astype(np.uint8)
    return reconstructed_channel_float, reconstructed_channel_uint8

if __name__ == "__main__":
    print("--- Watermark Embedding (Hash-based) ---")
    secret_message = input("Enter the secret message to embed as a watermark: ")
    
    embedding_method = 'additive'
    alpha = float(input("Enter the alpha for additive method (e.g., 30): "))
    wavelet = 'haar'

    try:
        cover_image_original = cv2.imread('cover_image.png')
        h, w, _ = cover_image_original.shape
        cover_image = cover_image_original[:h - h % 4, :w - w % 4]
        blue_channel, green_channel, red_channel = cv2.split(cover_image)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    watermark_binary_hash = hash_to_binary(secret_message)
    print(f"\nEmbedding a 256-bit SHA-256 hash of your message.")

    coeffs, U_orig, S_orig, V_orig, ll2_shape = apply_dwt_svd(blue_channel, wavelet=wavelet)
    S_watermarked = embed_watermark(S_orig, watermark_binary_hash, alpha, method=embedding_method)
    watermarked_blue_float, watermarked_blue_uint8 = reconstruct_channel(
        coeffs, U_orig, S_watermarked, V_orig, ll2_shape, wavelet=wavelet
    )
    
    watermarked_image = cv2.merge([watermarked_blue_uint8, green_channel, red_channel])
    
    cv2.imwrite('watermarked_image.png', watermarked_image)
    print("\nSaved 'watermarked_image.png' for viewing.")
    
    np.savez('watermark_data.npz', 
             watermarked_channel=watermarked_blue_float,
             U_orig=U_orig,
             V_orig=V_orig,
             wavelet=wavelet)
    print("Saved 'watermark_data.npz' for verification.")

    # --- NEW: Calculate and Display Performance Metrics ---
    print("\n--- Performance Metrics ---")
    # Calculate PSNR
    psnr_score = psnr(cover_image, watermarked_image, data_range=255)
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr_score:.2f} dB")
    
    # Calculate SSIM
    # For color images, ssim needs the channel_axis parameter
    ssim_score = ssim(cover_image, watermarked_image, channel_axis=2, data_range=255)
    print(f"SSIM (Structural Similarity Index): {ssim_score:.4f}")
    
    cv2.imshow('Original', cover_image)
    cv2.imshow('Watermarked', watermarked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()