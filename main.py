import cv2
import numpy as np
import pywt
import hashlib
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def text_to_binary(text):
    """(Helper) Converts a string of text to a binary string."""
    return ''.join(format(ord(char), '08b') for char in text)

def hash_to_binary(text_to_hash):
    """Hashes a string using SHA-256 and returns its 256-bit binary representation."""
    hashed = hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()
    # zfill(256) ensures the binary string is 256 bits long
    return bin(int(hashed, 16))[2:].zfill(256)

def apply_dwt_svd(image, wavelet='haar'):
    """Applies a 2-level DWT and then SVD to the LL2 sub-band."""
    coeffs = pywt.wavedec2(image.astype(np.float64), wavelet, level=2)
    LL2, _, _ = coeffs
    ll2_shape = LL2.shape
    U, S, V = np.linalg.svd(LL2)
    return (coeffs, U, S, V, ll2_shape)

def embed_watermark(S, watermark_binary, alpha, method):
    """Embeds the watermark using the chosen method (additive or multiplicative)."""
    S_modified = np.copy(S)
    watermark_len = len(watermark_binary)
    if watermark_len > len(S):
        raise ValueError(f"Image resolution is too low. Need {watermark_len} slots, but only found {len(S)}.")
    
    for i in range(watermark_len):
        bit = int(watermark_binary[i])
        if method == 'additive':
            # Adds a fixed, robust signal
            S_modified[i] += alpha * bit
        elif method == 'multiplicative':
            # Adds a proportional, more subtle signal
            S_modified[i] *= (1 + alpha * bit)
            
    return S_modified

def reconstruct_channel(coeffs, U, S_modified, V, ll2_shape, wavelet='haar'):
    """Reconstructs the image channel from modified DWT-SVD components."""
    # Create the full Sigma matrix from the modified S vector
    Sigma_modified = np.zeros(ll2_shape)
    k = min(ll2_shape)
    Sigma_modified[:k, :k] = np.diag(S_modified[:k])
    
    # Rebuild the LL2 sub-band
    LL2_modified = U @ Sigma_modified @ V
    
    # Reconstruct the full channel using the inverse DWT
    coeffs_modified = [LL2_modified, coeffs[1], coeffs[2]]
    reconstructed_channel_float = pywt.waverec2(coeffs_modified, wavelet)
    
    # Clip and convert to 8-bit integer format for saving as an image
    reconstructed_channel_uint8 = np.clip(reconstructed_channel_float, 0, 255).astype(np.uint8)
    
    return reconstructed_channel_float, reconstructed_channel_uint8

if __name__ == "__main__":
    print("--- DWT-SVD Watermark Embedding ---")
    secret_message = input("Enter the secret message to embed as a watermark: ")

    # --- 1. Get User Input for Parameters ---
    print("\nChoose an embedding method:")
    print("  1: Additive (Robust, use alpha like 30 or higher)")
    print("  2: Multiplicative (Subtle, use alpha like 0.05)")
    while True:
        try:
            choice = int(input("Enter choice (1 or 2): "))
            if choice == 1:
                embedding_method = 'additive'
                alpha_prompt = "Enter the alpha (e.g., 30): "
                break
            elif choice == 2:
                embedding_method = 'multiplicative'
                alpha_prompt = "Enter the alpha (e.g., 0.05): "
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    alpha = float(input(alpha_prompt))

    available_wavelets = ['haar', 'db2', 'sym4', 'bior1.3']
    print("\nChoose a wavelet:")
    for i, w_name in enumerate(available_wavelets):
        print(f"  {i+1}: {w_name}")
    while True:
        try:
            choice = int(input(f"Enter choice (1-{len(available_wavelets)}): "))
            if 1 <= choice <= len(available_wavelets):
                wavelet = available_wavelets[choice - 1]
                break
            else: print("Invalid choice.")
        except ValueError: print("Invalid input.")
    
    print(f"\nConfiguration: Method='{embedding_method}', Alpha={alpha}, Wavelet='{wavelet}'")

    # --- 2. Load and Prepare Image ---
    try:
        cover_image_original = cv2.imread('cover_image.png')
        h, w, _ = cover_image_original.shape
        # Crop image to be divisible by 4 for 2-level DWT
        cover_image = cover_image_original[:h - h % 4, :w - w % 4]
        blue_channel, green_channel, red_channel = cv2.split(cover_image)
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    # --- 3. Generate and Embed Watermark ---
    watermark_binary_hash = hash_to_binary(secret_message)
    print(f"Embedding a 256-bit SHA-256 hash of your message.")

    coeffs, U_orig, S_orig, V_orig, ll2_shape = apply_dwt_svd(blue_channel, wavelet=wavelet)
    
    try:
        S_watermarked = embed_watermark(S_orig, watermark_binary_hash, alpha, method=embedding_method)
    except ValueError as e:
        print(f"\nError: {e}")
        print("Your image is too small to embed a 256-bit hash. Use a 1024x1024 or larger image.")
        exit()

    watermarked_blue_float, watermarked_blue_uint8 = reconstruct_channel(
        coeffs, U_orig, S_watermarked, V_orig, ll2_shape, wavelet=wavelet
    )
    
    # --- 4. Save Outputs ---
    watermarked_image = cv2.merge([watermarked_blue_uint8, green_channel, red_channel])
    
    cv2.imwrite('watermarked_color_image.png', watermarked_image)
    print("\nSaved 'watermarked_color_image.png' for viewing.")
    
    # Save the "key" file needed for extraction
    np.savez_compressed('watermark_data.npz', 
             watermarked_channel_float=watermarked_blue_float,
             green_channel=green_channel, 
             red_channel=red_channel,   
             U_orig=U_orig,
             V_orig=V_orig,
             wavelet=wavelet)
    print("Saved 'watermark_data.npz' for verification.")

    # --- 5. Calculate Metrics and Display ---
    print("\n--- Performance Metrics ---")
    psnr_score = psnr(cover_image, watermarked_image, data_range=255)
    print(f"PSNR (Image Quality): {psnr_score:.2f} dB")
    ssim_score = ssim(cover_image, watermarked_image, channel_axis=2, data_range=255)
    print(f"SSIM (Structural Similarity): {ssim_score:.4f}")
    
    cv2.imshow('Original', cover_image)
    cv2.imshow('Watermarked', watermarked_image)
    
    print("\nDisplaying images. Close windows or press any key to exit.")
    # Robust killswitch loop
    while cv2.getWindowProperty('Original', cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break
    
    cv2.destroyAllWindows()
    print("Script terminated.")