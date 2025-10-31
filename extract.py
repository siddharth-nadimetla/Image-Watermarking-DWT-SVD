import cv2
import numpy as np
import pywt
import hashlib

def hash_to_binary(text_to_hash):
    """Hashes a string using SHA-256 and returns its 256-bit binary representation."""
    hashed = hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()
    return bin(int(hashed, 16))[2:].zfill(256)

def binary_to_text(binary_str):
    """(Helper) Converts a binary string back to a text string."""
    try:
        byte_chunks = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
        text = ''.join([chr(int(chunk, 2)) for chunk in byte_chunks])
        return text
    except (ValueError, TypeError):
        return "Error: Could not decode binary string."

def extract_watermark_hash(watermarked_channel, U_orig, V_orig, wavelet, alpha, method):
    """
    Extracts the embedded hash from the watermarked channel.
    This is NON-BLIND as it requires the original image to get S_orig.
    """
    
    # 1. Get S_orig from the original cover image
    try:
        original_full = cv2.imread('cover_image.png')
        h, w, _ = original_full.shape
        original_cropped = original_full[:h - h % 4, :w - w % 4]
        original_blue, _, _ = cv2.split(original_cropped)
        
        coeffs_orig = pywt.wavedec2(original_blue.astype(np.float64), wavelet, level=2)
        LL2_orig, _, _ = coeffs_orig
        _, S_orig, _ = np.linalg.svd(LL2_orig)
    except Exception as e:
        print(f"Could not load/process original cover image. Error: {e}")
        return None

    # 2. Get S_watermarked_extracted from the watermarked channel
    coeffs_watermarked = pywt.wavedec2(watermarked_channel.astype(np.float64), wavelet, level=2)
    LL2_watermarked, _, _ = coeffs_watermarked
    
    # Use the original U and V (the "key") to isolate the modified S values
    Sigma_watermarked_extracted = U_orig.T @ LL2_watermarked @ V_orig.T
    S_watermarked_extracted = np.diag(Sigma_watermarked_extracted)

    # 3. Compare S_orig and S_watermarked_extracted to decode the bits
    extracted_bits = ""
    for i in range(256): # A SHA-256 hash is always 256 bits
        if method == 'additive':
            threshold = S_orig[i] + alpha / 2
        elif method == 'multiplicative':
            threshold = S_orig[i] * (1 + alpha / 2)
        else:
            raise ValueError("Unknown embedding method.")
        
        if S_watermarked_extracted[i] > threshold:
            extracted_bits += '1'
        else:
            extracted_bits += '0'
            
    return extracted_bits

if __name__ == "__main__":
    print("--- DWT-SVD Watermark Verification ---")
    secret_message_to_verify = input("Enter the original SECRET message to verify: ")
    
    # --- 1. Get User Input for Parameters ---
    print("\nChoose the embedding method that was used:")
    print("  1: Additive")
    print("  2: Multiplicative")
    while True:
        try:
            choice = int(input("Enter choice (1 or 2): "))
            if choice == 1:
                embedding_method = 'additive'
                alpha_prompt = "Enter the alpha used (e.g., 30): "
                break
            elif choice == 2:
                embedding_method = 'multiplicative'
                alpha_prompt = "Enter the alpha used (e.g., 0.05): "
                break
            else: print("Invalid choice.")
        except ValueError: print("Invalid input.")

    alpha = float(input(alpha_prompt))
    
    # --- 2. Load Verification Data ---
    try:
        # Load the "key" file
        data = np.load('watermark_data.npz')
        watermarked_channel = data['watermarked_channel_float']
        U_orig = data['U_orig']
        V_orig = data['V_orig']
        wavelet = str(data['wavelet'])
        print(f"\nLoaded verification data. (Wavelet: '{wavelet}')")
    except FileNotFoundError:
        print("Error: 'watermark_data.npz' not found. Run main.py first.")
        exit()

    # --- 3. Extract and Verify Hash ---
    extracted_binary_hash = extract_watermark_hash(
        watermarked_channel, U_orig, V_orig, wavelet, alpha, method=embedding_method
    )
    
    if extracted_binary_hash:
        # Generate the hash of the secret message the user provided
        expected_binary_hash = hash_to_binary(secret_message_to_verify)
        
        print("\n--- Verification Result ---")
        if extracted_binary_hash == expected_binary_hash:
            print("✅ Success! The watermark is authentic and valid.")
        else:
            print("❌ Failure! The watermark is invalid or corrupt.")