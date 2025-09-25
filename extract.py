import cv2
import numpy as np
import pywt
import hashlib

def hash_to_binary(text_to_hash):
    """Hashes a string using SHA-256 and returns its binary representation."""
    hashed = hashlib.sha256(text_to_hash.encode('utf-8')).hexdigest()
    return bin(int(hashed, 16))[2:].zfill(256)

def extract_watermark_hash(watermarked_channel, U_orig, V_orig, wavelet, alpha, method):
    """Extracts the embedded hash from the watermarked channel."""
    # Apply DWT to the watermarked channel
    coeffs_watermarked = pywt.wavedec2(watermarked_channel, wavelet, level=2)
    LL2_watermarked, _, _ = coeffs_watermarked
    
    # Re-calculate the original singular values
    # S_orig = U_orig.T @ LL2_orig @ V_orig.T -> This is the non-blind part
    # For extraction, we need to isolate the modified S values
    Sigma_watermarked_extracted = U_orig.T @ LL2_watermarked @ V_orig.T
    S_watermarked_extracted = np.diag(Sigma_watermarked_extracted)
    
    # To compare, we need to re-generate the original S from a clean image
    # This is still a non-blind method
    # NOTE: A truly blind system would not need this step
    # For now, we will re-calculate it for comparison
    
    # We need to load the original image to get the original S
    original_full = cv2.imread('cover_image.png')
    h, w, _ = original_full.shape
    original_cropped = original_full[:h - h % 4, :w - w % 4]
    original_blue, _, _ = cv2.split(original_cropped)
    
    coeffs_orig = pywt.wavedec2(original_blue.astype(np.float64), wavelet, level=2)
    LL2_orig, _, _ = coeffs_orig
    _, S_orig, _ = np.linalg.svd(LL2_orig)

    # Now extract the bits by comparing S_orig and S_watermarked_extracted
    extracted_bits = ""
    for i in range(256): # A SHA-256 hash is always 256 bits
        if method == 'additive':
            threshold = S_orig[i] + alpha / 2
        elif method == 'multiplicative':
            # This logic must be present if you re-add multiplicative choice
            threshold = S_orig[i] * (1 + alpha / 2)
        
        if S_watermarked_extracted[i] > threshold:
            extracted_bits += '1'
        else:
            extracted_bits += '0'
            
    return extracted_bits

if __name__ == "__main__":
    print("--- Watermark Verification (Hash-based) ---")
    secret_message_to_verify = input("Enter the original SECRET message to verify the watermark: ")
    
    embedding_method = 'additive' # Must match the method used in main.py
    alpha = float(input("Enter the alpha used for embedding (e.g., 30): "))
    
    try:
        data = np.load('watermark_data.npz')
        watermarked_channel = data['watermarked_channel']
        U_orig = data['U_orig']
        V_orig = data['V_orig']
        wavelet = str(data['wavelet'])
    except FileNotFoundError:
        print("Error: 'watermark_data.npz' not found. Run main.py first.")
        exit()

    # Extract the embedded binary string (which should be a hash)
    extracted_binary_hash = extract_watermark_hash(
        watermarked_channel, U_orig, V_orig, wavelet, alpha, method=embedding_method
    )
    
    # Generate the hash of the secret message the user provided
    expected_binary_hash = hash_to_binary(secret_message_to_verify)
    
    print("\n--- Verification Result ---")
    if extracted_binary_hash == expected_binary_hash:
        print("✅ Success! The watermark is authentic and valid.")
    else:
        print("❌ Failure! The watermark is invalid or corrupt.")
        print(f"Expected hash: {expected_binary_hash[:32]}...")
        print(f"Extracted hash: {extracted_binary_hash[:32]}...")